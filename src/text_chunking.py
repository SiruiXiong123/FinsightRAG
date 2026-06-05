import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

try:
    from .paddleocr_blocks import get_block_content, get_block_label, get_parsing_res_list, load_ocr_json
    from .text_md_filter import (
        DEFAULT_DROP_LABELS,
        format_block,
        join_markdown_chunks,
        normalize_content,
        should_keep_block,
    )
except ImportError:
    try:
        from paddleocr_blocks import get_block_content, get_block_label, get_parsing_res_list, load_ocr_json
        from text_md_filter import (
            DEFAULT_DROP_LABELS,
            format_block,
            join_markdown_chunks,
            normalize_content,
            should_keep_block,
        )
    except ImportError:
        get_block_content = get_block_label = get_parsing_res_list = load_ocr_json = None
        DEFAULT_DROP_LABELS = set()
        format_block = join_markdown_chunks = normalize_content = should_keep_block = None


DEFAULT_OUTPUT_SUFFIX = "sentence_split"
DEFAULT_INITIAL_CHUNK_OUTPUT_SUFFIX = "initial_chunks"
DEFAULT_CHUNK_OUTPUT_SUFFIX = "chunk_merge"
DEFAULT_LANGUAGE = "english"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_WINDOW = 6
DEFAULT_OVERLAP = 2
DEFAULT_BREAKPOINT_PERCENTILE = 95.0
DEFAULT_MERGE_SIMILARITY_THRESHOLD = 0.85
DEFAULT_MAX_EMBEDDING_TOKENS = 512
DEFAULT_MAX_TOKENS = DEFAULT_MAX_EMBEDDING_TOKENS
DEFAULT_MAX_SENTENCE_GAP = 5
DEFAULT_MERGE_SAFETY_RATIO = 0.40
DEFAULT_EMBED_BATCH_SIZE = 32
DEFAULT_HIGH_OVERLAP_THRESHOLD = 0.80
DEFAULT_SMALL_CHUNK_REPAIR = True
DEFAULT_SMALL_CHUNK_SENTENCE_COUNT = 1
DEFAULT_SHORT_PREVIOUS_MERGE = True
DEFAULT_SHORT_PREVIOUS_MAX_TOKENS = 20
DEFAULT_DUPLICATE_OVERLAP_THRESHOLD = 0.80


@dataclass(frozen=True)
class SentenceSplitResult:
    source_path: Path
    sentence_output_path: Path
    initial_chunk_output_path: Optional[Path]
    chunk_output_path: Optional[Path]
    document_name: str
    sentence_count: int
    initial_chunk_count: int
    final_chunk_count: int
    merge_count: int
    validation_passed: bool
    containment_issue_count: int
    duplicate_overlap_issue_count: int


@dataclass(frozen=True)
class SemanticChunkingConfig:
    window: int = DEFAULT_WINDOW
    overlap: int = DEFAULT_OVERLAP
    step: Optional[int] = None
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    breakpoint_percentile: float = DEFAULT_BREAKPOINT_PERCENTILE
    merge_similarity_threshold: float = DEFAULT_MERGE_SIMILARITY_THRESHOLD
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_embedding_tokens: Optional[int] = None
    max_sentence_gap: int = DEFAULT_MAX_SENTENCE_GAP
    merge_safety_ratio: float = DEFAULT_MERGE_SAFETY_RATIO
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    high_overlap_threshold: float = DEFAULT_HIGH_OVERLAP_THRESHOLD
    small_chunk_repair: bool = DEFAULT_SMALL_CHUNK_REPAIR
    small_chunk_sentence_count: int = DEFAULT_SMALL_CHUNK_SENTENCE_COUNT
    short_previous_merge: bool = DEFAULT_SHORT_PREVIOUS_MERGE
    short_previous_max_tokens: int = DEFAULT_SHORT_PREVIOUS_MAX_TOKENS
    duplicate_overlap_threshold: float = DEFAULT_DUPLICATE_OVERLAP_THRESHOLD
    device: Optional[str] = None

    def resolved_step(self) -> int:
        if self.step is not None:
            return self.step
        return self.window - self.overlap

    def resolved_max_embedding_tokens(self) -> int:
        if self.max_embedding_tokens is not None:
            return self.max_embedding_tokens
        return self.max_tokens

    def validate(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be greater than 0.")
        if self.overlap < 0:
            raise ValueError("overlap must be greater than or equal to 0.")
        if self.overlap >= self.window:
            raise ValueError("overlap must be smaller than window.")
        if self.resolved_step() <= 0:
            raise ValueError("step must be greater than 0.")
        if not 0 <= self.breakpoint_percentile <= 100:
            raise ValueError("breakpoint_percentile must be between 0 and 100.")
        if self.merge_similarity_threshold < -1 or self.merge_similarity_threshold > 1:
            raise ValueError("merge_similarity_threshold must be between -1 and 1.")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0.")
        if self.resolved_max_embedding_tokens() <= 0:
            raise ValueError("max_embedding_tokens must be greater than 0.")
        if self.max_sentence_gap < 0:
            raise ValueError("max_sentence_gap must be greater than or equal to 0.")
        if self.merge_safety_ratio < 0:
            raise ValueError("merge_safety_ratio must be greater than or equal to 0.")
        if self.embed_batch_size <= 0:
            raise ValueError("embed_batch_size must be greater than 0.")
        if not 0 <= self.high_overlap_threshold <= 1:
            raise ValueError("high_overlap_threshold must be between 0 and 1.")
        if self.small_chunk_sentence_count < 0:
            raise ValueError("small_chunk_sentence_count must be greater than or equal to 0.")
        if self.short_previous_max_tokens < 0:
            raise ValueError("short_previous_max_tokens must be greater than or equal to 0.")
        if not 0 <= self.duplicate_overlap_threshold <= 1:
            raise ValueError("duplicate_overlap_threshold must be between 0 and 1.")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    sentence_ids: tuple[int, ...]
    text: str
    source_block_ids: tuple[int, ...]
    source_chunk_ids: tuple[str, ...]
    page_numbers: tuple[int, ...] = ()


def split_text_md_to_sentences(
    text_md_path: Path,
    output_dir: Optional[Path] = None,
    chunk_output_dir: Optional[Path] = None,
    language: str = DEFAULT_LANGUAGE,
    output_suffix: str = DEFAULT_OUTPUT_SUFFIX,
    initial_chunk_output_suffix: str = DEFAULT_INITIAL_CHUNK_OUTPUT_SUFFIX,
    chunk_output_suffix: str = DEFAULT_CHUNK_OUTPUT_SUFFIX,
    semantic_config: Optional[SemanticChunkingConfig] = None,
    semantic_chunking: bool = True,
    write_initial_chunks: bool = False,
    sentence_items: Optional[list[dict]] = None,
    overwrite: bool = True,
) -> SentenceSplitResult:
    text_md_path = Path(text_md_path).resolve()
    if not text_md_path.exists():
        raise FileNotFoundError(f"Text-only Markdown file not found: {text_md_path}")

    output_base = Path(output_dir).resolve() if output_dir else text_md_path.parent
    output_base.mkdir(parents=True, exist_ok=True)
    chunk_output_base = (
        Path(chunk_output_dir).resolve()
        if chunk_output_dir
        else output_base
    )
    chunk_output_base.mkdir(parents=True, exist_ok=True)
    document_name = infer_document_name(text_md_path)
    sentence_output_path = output_base / f"{document_name}_{output_suffix}.json"
    initial_chunk_output_path = (
        chunk_output_base / f"{document_name}_{initial_chunk_output_suffix}.json"
        if semantic_chunking and write_initial_chunks
        else None
    )
    chunk_output_path = (
        chunk_output_base / f"{document_name}_{chunk_output_suffix}.json"
        if semantic_chunking
        else None
    )
    for path in (sentence_output_path, initial_chunk_output_path, chunk_output_path):
        if path and path.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {path}")

    if sentence_items is None:
        text = text_md_path.read_text(encoding="utf-8")
        sentences = split_text_into_sentences(text, language=language)
        sentence_items = [
            {
                "sentence_id": index,
                "text": sentence,
            }
            for index, sentence in enumerate(sentences, start=1)
        ]
    else:
        sentence_items = normalize_sentence_items(sentence_items)
        sentences = [item["text"] for item in sentence_items]
    page_metadata_available = any(get_sentence_page_number(item) is not None for item in sentence_items)
    initial_chunk_count = 0
    final_chunk_count = 0
    merge_count = 0
    validation_passed = True
    containment_issue_count = 0
    duplicate_overlap_issue_count = 0
    chunking_payload = None
    if semantic_chunking:
        semantic_config = semantic_config or SemanticChunkingConfig()
        semantic_config.validate()
        chunking_payload = build_semantic_chunk_payload(
            sentence_items=sentence_items,
            config=semantic_config,
        )
        initial_chunk_count = len(chunking_payload["initial_chunks"])
        final_chunk_count = len(chunking_payload["chunks"])
        merge_count = chunking_payload["merge_count"]
        validation_passed = chunking_payload["final_validation"]["passed"]
        containment_issue_count = chunking_payload["final_validation"]["contained_pair_count"]
        duplicate_overlap_issue_count = chunking_payload["final_validation"]["duplicate_overlap_pair_count"]

    sentence_payload = {
        "document_name": document_name,
        "source_path": str(text_md_path),
        "splitter": "nltk.sent_tokenize",
        "language": language,
        "page_metadata_available": page_metadata_available,
        "sentence_count": len(sentences),
        "sentences": sentence_items,
    }
    sentence_output_path.write_text(
        json.dumps(sentence_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if initial_chunk_output_path and chunking_payload:
        initial_chunk_payload = {
            "document_name": document_name,
            "source_path": str(text_md_path),
            "sentence_split_path": str(sentence_output_path),
            "sentence_count": len(sentences),
            "page_metadata_available": chunking_payload["page_metadata_available"],
            "page_metadata_sentence_count": chunking_payload["page_metadata_sentence_count"],
            "config": chunking_payload["config"],
            "blocks": chunking_payload["blocks"],
            "initial_chunk_count": chunking_payload["initial_chunk_count"],
            "initial_chunks": chunking_payload["initial_chunks"],
        }
        initial_chunk_output_path.write_text(
            json.dumps(initial_chunk_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if chunk_output_path and chunking_payload:
        chunk_merge_payload = {
            "document_name": document_name,
            "source_path": str(text_md_path),
            "sentence_split_path": str(sentence_output_path),
            "initial_chunks_path": str(initial_chunk_output_path) if initial_chunk_output_path else None,
            "sentence_count": len(sentences),
            "page_metadata_available": chunking_payload["page_metadata_available"],
            "page_metadata_sentence_count": chunking_payload["page_metadata_sentence_count"],
            "merge_config": {
                "merge_similarity_threshold": chunking_payload["config"]["merge_similarity_threshold"],
                "max_tokens": chunking_payload["config"]["max_tokens"],
                "max_embedding_tokens": chunking_payload["config"]["max_embedding_tokens"],
                "max_sentence_gap": chunking_payload["config"]["max_sentence_gap"],
                "merge_safety_ratio": chunking_payload["config"]["merge_safety_ratio"],
                "high_overlap_threshold": chunking_payload["config"]["high_overlap_threshold"],
                "small_chunk_repair": chunking_payload["config"]["small_chunk_repair"],
                "small_chunk_sentence_count": chunking_payload["config"]["small_chunk_sentence_count"],
                "short_previous_merge": chunking_payload["config"]["short_previous_merge"],
                "short_previous_max_tokens": chunking_payload["config"]["short_previous_max_tokens"],
                "duplicate_overlap_threshold": chunking_payload["config"]["duplicate_overlap_threshold"],
                "embedding_model": chunking_payload["config"]["embedding_model"],
                "embed_batch_size": chunking_payload["config"]["embed_batch_size"],
                "device": chunking_payload["config"]["device"],
            },
            "initial_chunk_count": chunking_payload["initial_chunk_count"],
            "merge_count": chunking_payload["merge_count"],
            "merge_phase_counts": chunking_payload["merge_phase_counts"],
            "merge_events": chunking_payload["merge_events"],
            "final_validation": chunking_payload["final_validation"],
            "chunk_count": chunking_payload["chunk_count"],
            "chunks": chunking_payload["chunks"],
        }
        chunk_output_path.write_text(
            json.dumps(chunk_merge_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return SentenceSplitResult(
        source_path=text_md_path,
        sentence_output_path=sentence_output_path,
        initial_chunk_output_path=initial_chunk_output_path,
        chunk_output_path=chunk_output_path,
        document_name=document_name,
        sentence_count=len(sentences),
        initial_chunk_count=initial_chunk_count,
        final_chunk_count=final_chunk_count,
        merge_count=merge_count,
        validation_passed=validation_passed,
        containment_issue_count=containment_issue_count,
        duplicate_overlap_issue_count=duplicate_overlap_issue_count,
    )


def split_text_into_sentences(text: str, language: str = DEFAULT_LANGUAGE) -> list[str]:
    sent_tokenize = load_nltk_sentence_tokenizer()
    sentences: list[str] = []
    for paragraph in iter_markdown_paragraphs(text):
        heading = normalize_heading(paragraph)
        if heading:
            sentences.append(heading)
            continue
        for sentence in sent_tokenize(paragraph, language=language):
            normalized = normalize_sentence(sentence)
            if normalized:
                sentences.append(normalized)
    return sentences


def build_sentence_items_from_page_json_dir(page_json_dir: Path, language: str = DEFAULT_LANGUAGE) -> list[dict]:
    if load_ocr_json is None:
        raise RuntimeError("PaddleOCR block utilities are required for --page-json-dir.")
    sentence_items = []
    sentence_id = 0
    for page_json in sorted(Path(page_json_dir).glob("page_*.json")):
        page_index = page_index_from_page_json(page_json)
        page_text = text_only_markdown_from_page_json(page_json)
        for sentence in split_text_into_sentences(page_text, language=language):
            sentence_id += 1
            sentence_items.append(
                {
                    "sentence_id": sentence_id,
                    "text": sentence,
                    "page": page_index + 1,
                    "page_index": page_index,
                }
            )
    return sentence_items


def text_only_markdown_from_page_json(page_json: Path) -> str:
    data = load_ocr_json(page_json)
    chunks = []
    for block in get_parsing_res_list(data):
        label = get_block_label(block)
        content = normalize_content(get_block_content(block))
        if should_keep_block(label, content, None, set(DEFAULT_DROP_LABELS)):
            chunks.append(format_block(label, content, format_titles=True))
    return join_markdown_chunks(chunks)


def page_index_from_page_json(page_json: Path) -> int:
    data = load_ocr_json(page_json)
    if isinstance(data.get("page_index"), int):
        return data["page_index"]
    match = re.search(r"page_(\d+)", page_json.stem)
    if match:
        return max(0, int(match.group(1)) - 1)
    return 0


def normalize_sentence_items(sentence_items: list[dict]) -> list[dict]:
    normalized = []
    for index, item in enumerate(sentence_items, start=1):
        sentence_id = int(item.get("sentence_id") or index)
        text = normalize_sentence(item.get("text") or "")
        if not text:
            continue
        payload = {
            "sentence_id": sentence_id,
            "text": text,
        }
        for key in ("page", "page_num", "page_number", "page_index", "page_id"):
            if key in item and item[key] not in (None, ""):
                payload[key] = item[key]
        normalized.append(payload)
    return normalized


def load_nltk_sentence_tokenizer():
    try:
        from nltk import data, download, sent_tokenize
    except ImportError as exc:
        raise RuntimeError(
            "NLTK is required for sentence splitting. Install it in llm_env with: "
            "python -m pip install nltk"
        ) from exc

    missing_resources = []
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            data.find(resource)
        except LookupError:
            missing_resources.append(resource.rsplit("/", 1)[-1])
    for resource_name in missing_resources:
        download(resource_name, quiet=True)
    return sent_tokenize


def iter_markdown_paragraphs(text: str):
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for paragraph in re.split(r"\n\s*\n+", normalized):
        cleaned = normalize_sentence(paragraph)
        if cleaned:
            yield cleaned


def normalize_heading(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return ""
    match = re.match(r"^#{1,6}\s+(.+)$", lines[0])
    if not match:
        return ""
    return normalize_sentence(match.group(1))


def normalize_sentence(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def infer_document_name(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_text"):
        return stem[: -len("_text")]
    if stem.endswith("_text_only"):
        return stem[: -len("_text_only")]
    return stem


def build_semantic_chunk_payload(
    sentence_items: list[dict],
    config: SemanticChunkingConfig,
) -> dict:
    config_payload = {
        **asdict(config),
        "step": config.resolved_step(),
        "max_embedding_tokens": config.resolved_max_embedding_tokens(),
    }
    if not sentence_items:
        return build_empty_semantic_chunk_payload(config_payload, config)

    model, util, np = load_embedding_runtime(config)
    blocks = build_sentence_blocks(sentence_items, config)
    sentence_embeddings = encode_texts(
        model=model,
        texts=[item["text"] for item in sentence_items],
        batch_size=config.embed_batch_size,
    )
    sentence_embedding_by_id = {
        item["sentence_id"]: sentence_embeddings[index]
        for index, item in enumerate(sentence_items)
    }
    sentence_text_by_id = {
        item["sentence_id"]: item["text"]
        for item in sentence_items
    }
    sentence_page_by_id = {
        item["sentence_id"]: page_number
        for item in sentence_items
        for page_number in [get_sentence_page_number(item)]
        if page_number is not None
    }
    page_metadata_available = bool(sentence_page_by_id)

    initial_chunks: list[Chunk] = []
    chunk_counter = 0
    block_payloads = []
    for block in blocks:
        block_sentence_ids = block["sentence_ids"]
        block_embeddings = stack_embeddings(
            [sentence_embedding_by_id[sentence_id] for sentence_id in block_sentence_ids]
        )
        chunk_ranges, distances, threshold = split_block_by_sentence_distances(
            block_sentence_ids=block_sentence_ids,
            block_embeddings=block_embeddings,
            util=util,
            np=np,
            percentile=config.breakpoint_percentile,
        )
        block_chunk_ids = []
        for chunk_sentence_ids in chunk_ranges:
            chunk_counter += 1
            chunk_id = f"chunk_{chunk_counter:06d}"
            chunk = make_chunk(
                chunk_id=chunk_id,
                sentence_ids=chunk_sentence_ids,
                source_block_ids=(block["block_id"],),
                source_chunk_ids=(chunk_id,),
                sentence_text_by_id=sentence_text_by_id,
                sentence_page_by_id=sentence_page_by_id,
            )
            initial_chunks.append(chunk)
            block_chunk_ids.append(chunk.chunk_id)
        block_payloads.append(
            {
                "block_id": block["block_id"],
                "sentence_ids": block_sentence_ids,
                "distance_count": len(distances),
                "breakpoint_threshold": threshold,
                "distances": distances,
                "chunk_ids": block_chunk_ids,
            }
        )

    final_chunks, merge_events = merge_chunks_iteratively(
        chunks=initial_chunks,
        model=model,
        util=util,
        config=config,
        sentence_text_by_id=sentence_text_by_id,
        sentence_page_by_id=sentence_page_by_id,
        next_chunk_number=chunk_counter + 1,
    )
    final_chunks = sorted(final_chunks, key=lambda chunk: (min(chunk.sentence_ids), chunk.chunk_id))
    final_validation = validate_final_chunks(final_chunks, config)
    return {
        "enabled": True,
        "config": config_payload,
        "page_metadata_available": page_metadata_available,
        "page_metadata_sentence_count": len(sentence_page_by_id),
        "blocks": block_payloads,
        "initial_chunk_count": len(initial_chunks),
        "initial_chunks": [chunk_to_payload(chunk) for chunk in initial_chunks],
        "merge_count": len(merge_events),
        "merge_phase_counts": count_merge_events_by_phase(merge_events),
        "merge_events": merge_events,
        "final_validation": final_validation,
        "chunk_count": len(final_chunks),
        "chunks": [chunk_to_payload(chunk) for chunk in final_chunks],
    }


def build_empty_semantic_chunk_payload(config_payload: dict, config: SemanticChunkingConfig) -> dict:
    return {
        "enabled": True,
        "config": config_payload,
        "page_metadata_available": False,
        "page_metadata_sentence_count": 0,
        "blocks": [],
        "initial_chunk_count": 0,
        "initial_chunks": [],
        "merge_count": 0,
        "merge_phase_counts": {},
        "merge_events": [],
        "final_validation": {
            "passed": True,
            "contained_pair_count": 0,
            "duplicate_overlap_pair_count": 0,
            "duplicate_overlap_threshold": config.duplicate_overlap_threshold,
            "contained_pair_examples": [],
            "duplicate_overlap_pair_examples": [],
        },
        "chunk_count": 0,
        "chunks": [],
    }


def load_embedding_runtime(config: SemanticChunkingConfig):
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer, util
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers and numpy are required for semantic chunking. "
            "Install them in llm_env with: python -m pip install sentence-transformers"
        ) from exc

    if config.device:
        model = SentenceTransformer(config.embedding_model, device=config.device)
    else:
        model = SentenceTransformer(config.embedding_model)
    return model, util, np


def build_sentence_blocks(sentence_items: list[dict], config: SemanticChunkingConfig) -> list[dict]:
    sentence_ids = [item["sentence_id"] for item in sentence_items]
    step = config.resolved_step()
    blocks = []
    for block_id, start in enumerate(range(0, len(sentence_ids), step), start=1):
        block_sentence_ids = sentence_ids[start : start + config.window]
        if not block_sentence_ids:
            continue
        blocks.append(
            {
                "block_id": block_id,
                "sentence_ids": block_sentence_ids,
            }
        )
        if start + config.window >= len(sentence_ids):
            break
    return blocks


def encode_texts(model, texts: list[str], batch_size: int):
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def stack_embeddings(embeddings):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required by sentence-transformers.") from exc
    return torch.stack(list(embeddings))


def split_block_by_sentence_distances(
    block_sentence_ids: list[int],
    block_embeddings,
    util,
    np,
    percentile: float,
) -> tuple[list[tuple[int, ...]], list[float], Optional[float]]:
    if len(block_sentence_ids) <= 1:
        return [tuple(block_sentence_ids)], [], None

    similarities = util.cos_sim(block_embeddings[:-1], block_embeddings[1:])
    distances_array = 1 - similarities.diag().cpu().numpy()
    distances = [float(distance) for distance in distances_array.tolist()]
    threshold = float(np.percentile(distances_array, percentile))
    break_indexes = [
        index
        for index, distance in enumerate(distances)
        if distance > threshold
    ]
    chunks = []
    start = 0
    for break_index in break_indexes:
        chunks.append(tuple(block_sentence_ids[start : break_index + 1]))
        start = break_index + 1
    chunks.append(tuple(block_sentence_ids[start:]))
    return [chunk for chunk in chunks if chunk], distances, threshold


def make_chunk(
    chunk_id: str,
    sentence_ids: tuple[int, ...] | list[int],
    source_block_ids: tuple[int, ...],
    source_chunk_ids: tuple[str, ...],
    sentence_text_by_id: dict[int, str],
    sentence_page_by_id: Optional[dict[int, int]] = None,
) -> Chunk:
    unique_sentence_ids = tuple(sorted(set(sentence_ids)))
    text = build_chunk_text(unique_sentence_ids, sentence_text_by_id)
    sentence_page_by_id = sentence_page_by_id or {}
    page_numbers = tuple(
        sorted(
            {
                sentence_page_by_id[sentence_id]
                for sentence_id in unique_sentence_ids
                if sentence_id in sentence_page_by_id
            }
        )
    )
    return Chunk(
        chunk_id=chunk_id,
        sentence_ids=unique_sentence_ids,
        text=text,
        source_block_ids=tuple(sorted(set(source_block_ids))),
        source_chunk_ids=tuple(sorted(set(source_chunk_ids), key=natural_chunk_sort_key)),
        page_numbers=page_numbers,
    )


def merge_chunks_iteratively(
    chunks: list[Chunk],
    model,
    util,
    config: SemanticChunkingConfig,
    sentence_text_by_id: dict[int, str],
    sentence_page_by_id: dict[int, int],
    next_chunk_number: int,
) -> tuple[list[Chunk], list[dict]]:
    if len(chunks) <= 1:
        return chunks, []

    merge_events = []
    max_embedding_merges = int(len(chunks) * config.merge_safety_ratio)
    current_chunks = list(chunks)
    token_count_cache: dict[tuple[int, ...], int] = {}

    if config.small_chunk_repair and config.short_previous_merge:
        current_chunks, short_previous_events, next_chunk_number = repair_small_chunks(
            chunks=current_chunks,
            model=model,
            config=config,
            sentence_text_by_id=sentence_text_by_id,
            sentence_page_by_id=sentence_page_by_id,
            token_count_cache=token_count_cache,
            next_chunk_number=next_chunk_number,
        )
        append_merge_events(merge_events, short_previous_events)

    current_chunks, cleanup_events = cleanup_contained_chunks(
        chunks=current_chunks,
        phase="containment_cleanup",
    )
    append_merge_events(merge_events, cleanup_events)

    current_chunks, overlap_events, next_chunk_number = merge_high_overlap_chunks(
        chunks=current_chunks,
        model=model,
        config=config,
        sentence_text_by_id=sentence_text_by_id,
        sentence_page_by_id=sentence_page_by_id,
        token_count_cache=token_count_cache,
        next_chunk_number=next_chunk_number,
    )
    append_merge_events(merge_events, overlap_events)

    current_chunks, embedding_events, next_chunk_number = merge_embedding_greedy_chunks(
        chunks=current_chunks,
        model=model,
        util=util,
        config=config,
        sentence_text_by_id=sentence_text_by_id,
        sentence_page_by_id=sentence_page_by_id,
        token_count_cache=token_count_cache,
        next_chunk_number=next_chunk_number,
        max_embedding_merges=max_embedding_merges,
    )
    append_merge_events(merge_events, embedding_events)

    current_chunks, final_cleanup_events = cleanup_contained_chunks(
        chunks=current_chunks,
        phase="final_containment_cleanup",
    )
    append_merge_events(merge_events, final_cleanup_events)

    return current_chunks, merge_events


def merge_embedding_greedy_chunks(
    chunks: list[Chunk],
    model,
    util,
    config: SemanticChunkingConfig,
    sentence_text_by_id: dict[int, str],
    sentence_page_by_id: dict[int, int],
    token_count_cache: dict[tuple[int, ...], int],
    next_chunk_number: int,
    max_embedding_merges: int,
) -> tuple[list[Chunk], list[dict], int]:
    if len(chunks) <= 1 or max_embedding_merges <= 0:
        return chunks, [], next_chunk_number

    current_chunks = list(chunks)
    merge_events = []
    embeddings_by_id = encode_chunks_by_id(
        chunks=current_chunks,
        model=model,
        batch_size=config.embed_batch_size,
    )

    while len(current_chunks) > 1 and len(merge_events) < max_embedding_merges:
        best_pair = find_best_embedding_legal_pair(
            chunks=current_chunks,
            embeddings_by_id=embeddings_by_id,
            model=model,
            util=util,
            config=config,
            sentence_text_by_id=sentence_text_by_id,
            token_count_cache=token_count_cache,
        )
        if best_pair is None:
            break

        first, second, similarity, merged_sentence_ids, merged_token_count, position_reasons = best_pair
        merged_chunk_id = f"chunk_{next_chunk_number:06d}"
        merged_chunk = merge_two_chunks(
            chunk_id=merged_chunk_id,
            first=first,
            second=second,
            sentence_text_by_id=sentence_text_by_id,
            sentence_page_by_id=sentence_page_by_id,
        )
        next_chunk_number += 1
        current_chunks = [
            chunk
            for chunk in current_chunks
            if chunk.chunk_id not in {first.chunk_id, second.chunk_id}
        ]
        current_chunks.append(merged_chunk)
        del embeddings_by_id[first.chunk_id]
        del embeddings_by_id[second.chunk_id]
        embeddings_by_id[merged_chunk.chunk_id] = encode_texts(
            model=model,
            texts=[merged_chunk.text],
            batch_size=1,
        )[0]
        merge_events.append(
            {
                "phase": "embedding_greedy_merge",
                "source_chunk_ids": list(merged_chunk.source_chunk_ids),
                "direct_chunk_ids": [first.chunk_id, second.chunk_id],
                "merged_chunk_id": merged_chunk.chunk_id,
                "similarity": float(similarity),
                "position_reasons": position_reasons,
                "sentence_ids": list(merged_chunk.sentence_ids),
                "token_count": merged_token_count,
            }
        )

    return current_chunks, merge_events, next_chunk_number


def find_best_embedding_legal_pair(
    chunks: list[Chunk],
    embeddings_by_id: dict[str, object],
    model,
    util,
    config: SemanticChunkingConfig,
    sentence_text_by_id: dict[int, str],
    token_count_cache: dict[tuple[int, ...], int],
) -> Optional[tuple[Chunk, Chunk, float, tuple[int, ...], int, list[str]]]:
    import torch

    ordered_chunks = sort_chunks(chunks)
    ordered_embeddings = torch.stack([embeddings_by_id[chunk.chunk_id] for chunk in ordered_chunks])
    similarity_matrix = util.cos_sim(ordered_embeddings, ordered_embeddings)
    best = None

    for first_index, first in enumerate(ordered_chunks):
        first_max = max(first.sentence_ids)
        for second_index in range(first_index + 1, len(ordered_chunks)):
            second = ordered_chunks[second_index]
            second_min = min(second.sentence_ids)
            if (
                second_min - first_max > config.max_sentence_gap
                and not chunks_are_same_or_adjacent_page(first, second)
            ):
                break
            position_reasons = chunk_position_reasons(first, second, config.max_sentence_gap)
            if not position_reasons:
                continue
            similarity = float(similarity_matrix[first_index, second_index].cpu().item())
            if similarity <= config.merge_similarity_threshold:
                continue
            merged_sentence_ids = tuple(sorted(set(first.sentence_ids + second.sentence_ids)))
            token_count = get_chunk_token_count(
                sentence_ids=merged_sentence_ids,
                sentence_text_by_id=sentence_text_by_id,
                model=model,
                cache=token_count_cache,
            )
            if token_count > config.resolved_max_embedding_tokens():
                continue
            if best is None or similarity > best[2]:
                best = (
                    first,
                    second,
                    similarity,
                    merged_sentence_ids,
                    token_count,
                    position_reasons,
                )
    return best


def cleanup_contained_chunks(chunks: list[Chunk], phase: str) -> tuple[list[Chunk], list[dict]]:
    current_chunks = sort_chunks(chunks)
    events = []
    changed = True
    while changed and len(current_chunks) > 1:
        changed = False
        sentence_sets = {chunk.chunk_id: set(chunk.sentence_ids) for chunk in current_chunks}
        for small in reversed(current_chunks):
            small_set = sentence_sets[small.chunk_id]
            candidates = []
            for candidate in current_chunks:
                if candidate.chunk_id == small.chunk_id:
                    continue
                candidate_set = sentence_sets[candidate.chunk_id]
                if not small_set.issubset(candidate_set):
                    continue
                if len(candidate_set) > len(small_set) or chunk_sort_key(candidate) < chunk_sort_key(small):
                    candidates.append(candidate)
            if not candidates:
                continue
            target = sorted(
                candidates,
                key=lambda chunk: (-len(chunk.sentence_ids), chunk_sort_key(chunk)),
            )[0]
            updated_target = merge_chunk_metadata(target, small)
            current_chunks = [
                updated_target if chunk.chunk_id == target.chunk_id else chunk
                for chunk in current_chunks
                if chunk.chunk_id != small.chunk_id
            ]
            current_chunks = sort_chunks(current_chunks)
            events.append(
                {
                    "phase": phase,
                    "removed_chunk_id": small.chunk_id,
                    "target_chunk_id": updated_target.chunk_id,
                    "source_chunk_ids": list(updated_target.source_chunk_ids),
                    "sentence_ids": list(updated_target.sentence_ids),
                    "contained_sentence_ids": list(small.sentence_ids),
                    "overlap_ratio": 1.0,
                }
            )
            changed = True
            break
    return current_chunks, events


def merge_high_overlap_chunks(
    chunks: list[Chunk],
    model,
    config: SemanticChunkingConfig,
    sentence_text_by_id: dict[int, str],
    sentence_page_by_id: dict[int, int],
    token_count_cache: dict[tuple[int, ...], int],
    next_chunk_number: int,
) -> tuple[list[Chunk], list[dict], int]:
    current_chunks = sort_chunks(chunks)
    events = []
    while len(current_chunks) > 1:
        best_pair = find_best_high_overlap_pair(
            chunks=current_chunks,
            model=model,
            config=config,
            sentence_text_by_id=sentence_text_by_id,
            token_count_cache=token_count_cache,
        )
        if best_pair is None:
            break
        first, second, overlap_ratio, merged_token_count = best_pair
        merged_chunk_id = f"chunk_{next_chunk_number:06d}"
        merged_chunk = merge_two_chunks(
            chunk_id=merged_chunk_id,
            first=first,
            second=second,
            sentence_text_by_id=sentence_text_by_id,
            sentence_page_by_id=sentence_page_by_id,
        )
        next_chunk_number += 1
        current_chunks = [
            chunk
            for chunk in current_chunks
            if chunk.chunk_id not in {first.chunk_id, second.chunk_id}
        ]
        current_chunks.append(merged_chunk)
        current_chunks = sort_chunks(current_chunks)
        events.append(
            {
                "phase": "high_overlap_merge",
                "source_chunk_ids": list(merged_chunk.source_chunk_ids),
                "direct_chunk_ids": [first.chunk_id, second.chunk_id],
                "merged_chunk_id": merged_chunk.chunk_id,
                "overlap_ratio": overlap_ratio,
                "sentence_ids": list(merged_chunk.sentence_ids),
                "token_count": merged_token_count,
            }
        )
    return current_chunks, events, next_chunk_number


def find_best_high_overlap_pair(
    chunks: list[Chunk],
    model,
    config: SemanticChunkingConfig,
    sentence_text_by_id: dict[int, str],
    token_count_cache: dict[tuple[int, ...], int],
) -> Optional[tuple[Chunk, Chunk, float, int]]:
    ordered_chunks = sort_chunks(chunks)
    best = None
    for first_index, first in enumerate(ordered_chunks):
        first_max = max(first.sentence_ids)
        for second in ordered_chunks[first_index + 1 :]:
            if min(second.sentence_ids) > first_max:
                break
            overlap_ratio = sentence_overlap_ratio(first.sentence_ids, second.sentence_ids)
            if overlap_ratio < config.high_overlap_threshold:
                continue
            merged_sentence_ids = tuple(sorted(set(first.sentence_ids + second.sentence_ids)))
            token_count = get_chunk_token_count(
                sentence_ids=merged_sentence_ids,
                sentence_text_by_id=sentence_text_by_id,
                model=model,
                cache=token_count_cache,
            )
            if token_count > config.resolved_max_embedding_tokens():
                continue
            candidate = (first, second, overlap_ratio, token_count)
            if best is None or high_overlap_pair_sort_key(candidate) > high_overlap_pair_sort_key(best):
                best = candidate
    return best


def repair_small_chunks(
    chunks: list[Chunk],
    model,
    config: SemanticChunkingConfig,
    sentence_text_by_id: dict[int, str],
    sentence_page_by_id: dict[int, int],
    token_count_cache: dict[tuple[int, ...], int],
    next_chunk_number: int,
) -> tuple[list[Chunk], list[dict], int]:
    current_chunks = sort_chunks(chunks)
    events = []
    while len(current_chunks) > 1:
        repaired = False
        for chunk in current_chunks:
            if len(chunk.sentence_ids) > config.small_chunk_sentence_count:
                continue
            previous_chunk = find_best_previous_continuation_chunk(current_chunks, chunk)
            if previous_chunk is None:
                continue
            chunk_token_count = get_chunk_token_count(
                sentence_ids=chunk.sentence_ids,
                sentence_text_by_id=sentence_text_by_id,
                model=model,
                cache=token_count_cache,
            )
            if chunk_token_count > config.short_previous_max_tokens:
                continue
            content_reasons = short_answer_like_reasons(chunk.text)
            if not content_reasons:
                continue
            merged_token_count = get_merged_token_count(
                first=previous_chunk,
                second=chunk,
                model=model,
                sentence_text_by_id=sentence_text_by_id,
                token_count_cache=token_count_cache,
            )
            if merged_token_count > config.resolved_max_embedding_tokens():
                continue

            merged_chunk_id = f"chunk_{next_chunk_number:06d}"
            merged_chunk = merge_two_chunks(
                chunk_id=merged_chunk_id,
                first=previous_chunk,
                second=chunk,
                sentence_text_by_id=sentence_text_by_id,
                sentence_page_by_id=sentence_page_by_id,
            )
            next_chunk_number += 1
            current_chunks = [
                item
                for item in current_chunks
                if item.chunk_id not in {chunk.chunk_id, previous_chunk.chunk_id}
            ]
            current_chunks.append(merged_chunk)
            current_chunks = sort_chunks(current_chunks)
            events.append(
                {
                    "phase": "short_previous_repair",
                    "source_chunk_ids": list(merged_chunk.source_chunk_ids),
                    "direct_chunk_ids": [previous_chunk.chunk_id, chunk.chunk_id],
                    "merged_chunk_id": merged_chunk.chunk_id,
                    "repair_direction": "previous",
                    "short_chunk_id": chunk.chunk_id,
                    "short_chunk_token_count": chunk_token_count,
                    "content_reasons": content_reasons,
                    "sentence_ids": list(merged_chunk.sentence_ids),
                    "token_count": merged_token_count,
                }
            )
            repaired = True
            break
        if not repaired:
            break
    return current_chunks, events, next_chunk_number


def find_best_previous_continuation_chunk(chunks: list[Chunk], chunk: Chunk) -> Optional[Chunk]:
    candidates = [
        candidate
        for candidate in chunks
        if candidate.chunk_id != chunk.chunk_id
        and chunk_continues_previous(candidate, chunk)
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda candidate: (
            -len(candidate.sentence_ids),
            min(candidate.sentence_ids),
            chunk_sort_key(candidate),
        ),
    )[0]


def merge_two_chunks(
    chunk_id: str,
    first: Chunk,
    second: Chunk,
    sentence_text_by_id: dict[int, str],
    sentence_page_by_id: dict[int, int],
) -> Chunk:
    return make_chunk(
        chunk_id=chunk_id,
        sentence_ids=first.sentence_ids + second.sentence_ids,
        source_block_ids=first.source_block_ids + second.source_block_ids,
        source_chunk_ids=first.source_chunk_ids + second.source_chunk_ids,
        sentence_text_by_id=sentence_text_by_id,
        sentence_page_by_id=sentence_page_by_id,
    )


def chunk_continues_previous(previous_chunk: Chunk, chunk: Chunk) -> bool:
    return min(chunk.sentence_ids) == max(previous_chunk.sentence_ids) + 1


def short_answer_like_reasons(text: str) -> list[str]:
    normalized = normalize_short_candidate_text(text)
    if not normalized:
        return []

    reasons = []
    lower = normalized.lower().strip(".;:")
    if contains_checkbox_marker(normalized):
        reasons.append("checkbox")
    if is_numeric_like_short_text(normalized):
        reasons.append("numeric_like")
    if is_short_status_text(lower):
        reasons.append("short_status")
    if is_yes_no_answer_text(lower):
        reasons.append("yes_no_answer")
    return reasons


def normalize_short_candidate_text(text: str) -> str:
    value = normalize_sentence(text)
    value = re.sub(r"^[>\-\*\u2022\s]+", "", value)
    value = re.sub(r"^\(?[a-zA-Z0-9]{1,3}\)?[\.)]\s+", "", value)
    return value.strip()


def contains_checkbox_marker(text: str) -> bool:
    checkbox_markers = (
        "[x]",
        "[ ]",
        "(x)",
        "( )",
        "\u2610",
        "\u2611",
        "\u2612",
        "\u25a1",
        "\u25a2",
        "\u25a3",
        "\u25a0",
        "\u2713",
        "\u2714",
        "\u2715",
        "\u2717",
        "\u2718",
    )
    lower = text.lower()
    return any(marker in lower for marker in checkbox_markers)


def is_numeric_like_short_text(text: str) -> bool:
    compact = text.strip()
    if not re.search(r"\d", compact):
        return False
    token_count = len(compact.split())
    if token_count > 8:
        return False
    allowed = re.sub(r"[\d\s,.\-+/$%():'’A-Za-z]", "", compact)
    if allowed:
        return False
    non_space = re.sub(r"\s+", "", compact)
    digit_symbol_count = len(re.findall(r"[\d,.\-+/$%():'’]", non_space))
    return digit_symbol_count / max(len(non_space), 1) >= 0.45


def is_short_status_text(lower_text: str) -> bool:
    status_values = {
        "n/a",
        "na",
        "not applicable",
        "none",
        "nil",
        "true",
        "false",
        "ok",
        "okay",
        "done",
        "complete",
        "completed",
        "incomplete",
        "pending",
        "approved",
        "denied",
        "selected",
        "unselected",
        "checked",
        "unchecked",
        "pass",
        "passed",
        "fail",
        "failed",
        "valid",
        "invalid",
        "met",
        "not met",
    }
    return lower_text in status_values


def is_yes_no_answer_text(lower_text: str) -> bool:
    return bool(
        re.fullmatch(
            r"(yes|no)(\s+[\u2610\u2611\u2612])?(\s+(yes|no)\s+[\u2610\u2611\u2612])?",
            lower_text,
        )
    )


def merge_chunk_metadata(target: Chunk, absorbed: Chunk) -> Chunk:
    return Chunk(
        chunk_id=target.chunk_id,
        sentence_ids=target.sentence_ids,
        text=target.text,
        source_block_ids=tuple(sorted(set(target.source_block_ids + absorbed.source_block_ids))),
        source_chunk_ids=tuple(
            sorted(
                set(target.source_chunk_ids + absorbed.source_chunk_ids),
                key=natural_chunk_sort_key,
            )
        ),
        page_numbers=tuple(sorted(set(target.page_numbers + absorbed.page_numbers))),
    )


def encode_chunks_by_id(chunks: list[Chunk], model, batch_size: int) -> dict[str, object]:
    return {
        chunk.chunk_id: embedding
        for chunk, embedding in zip(
            chunks,
            encode_texts(
                model=model,
                texts=[chunk.text for chunk in chunks],
                batch_size=batch_size,
            ),
        )
    }


def get_merged_token_count(
    first: Chunk,
    second: Chunk,
    model,
    sentence_text_by_id: dict[int, str],
    token_count_cache: dict[tuple[int, ...], int],
) -> int:
    return get_chunk_token_count(
        sentence_ids=tuple(sorted(set(first.sentence_ids + second.sentence_ids))),
        sentence_text_by_id=sentence_text_by_id,
        model=model,
        cache=token_count_cache,
    )


def sentence_overlap_ratio(first_sentence_ids: tuple[int, ...], second_sentence_ids: tuple[int, ...]) -> float:
    first_set = set(first_sentence_ids)
    second_set = set(second_sentence_ids)
    if not first_set or not second_set:
        return 0.0
    return len(first_set.intersection(second_set)) / min(len(first_set), len(second_set))


def high_overlap_pair_sort_key(candidate: tuple[Chunk, Chunk, float, int]) -> tuple[float, int, int]:
    first, second, overlap_ratio, token_count = candidate
    return (
        overlap_ratio,
        -token_count,
        -sentence_id_gap(first.sentence_ids, second.sentence_ids),
    )


def chunk_position_reasons(first: Chunk, second: Chunk, max_sentence_gap: int) -> list[str]:
    reasons = []
    if set(first.sentence_ids).intersection(second.sentence_ids):
        reasons.append("sentence_overlap")
    if sentence_id_gap(first.sentence_ids, second.sentence_ids) <= max_sentence_gap:
        reasons.append("sentence_gap")
    if chunks_are_same_or_adjacent_page(first, second):
        reasons.append("same_or_adjacent_page")
    return reasons


def chunks_are_same_or_adjacent_page(first: Chunk, second: Chunk) -> bool:
    if not first.page_numbers or not second.page_numbers:
        return False
    return min(
        abs(first_page - second_page)
        for first_page in first.page_numbers
        for second_page in second.page_numbers
    ) <= 1


def sort_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return sorted(chunks, key=chunk_sort_key)


def chunk_sort_key(chunk: Chunk) -> tuple[int, int, int, tuple]:
    return (
        min(chunk.sentence_ids),
        max(chunk.sentence_ids),
        len(chunk.sentence_ids),
        natural_chunk_sort_key(chunk.chunk_id),
    )


def natural_chunk_sort_key(value: str) -> tuple[str, int, str]:
    match = re.search(r"(\d+)$", value)
    if match:
        return (value[: match.start()], int(match.group(1)), "")
    return (value, -1, value)


def append_merge_events(target: list[dict], events: list[dict]) -> None:
    for event in events:
        event = dict(event)
        event["merge_id"] = len(target) + 1
        target.append(event)


def count_merge_events_by_phase(events: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        phase = event.get("phase", "unknown")
        counts[phase] = counts.get(phase, 0) + 1
    return counts


def validate_final_chunks(chunks: list[Chunk], config: SemanticChunkingConfig) -> dict:
    contained_pairs = find_contained_chunk_pairs(chunks)
    duplicate_overlap_pairs = find_duplicate_overlap_pairs(
        chunks=chunks,
        threshold=config.duplicate_overlap_threshold,
    )
    return {
        "passed": not contained_pairs and not duplicate_overlap_pairs,
        "contained_pair_count": len(contained_pairs),
        "duplicate_overlap_pair_count": len(duplicate_overlap_pairs),
        "duplicate_overlap_threshold": config.duplicate_overlap_threshold,
        "contained_pair_examples": contained_pairs[:20],
        "duplicate_overlap_pair_examples": duplicate_overlap_pairs[:20],
    }


def find_contained_chunk_pairs(chunks: list[Chunk]) -> list[dict]:
    ordered_chunks = sort_chunks(chunks)
    pairs = []
    sentence_sets = {chunk.chunk_id: set(chunk.sentence_ids) for chunk in ordered_chunks}
    for first_index, first in enumerate(ordered_chunks):
        first_set = sentence_sets[first.chunk_id]
        for second in ordered_chunks[first_index + 1 :]:
            if min(second.sentence_ids) > max(first.sentence_ids):
                break
            second_set = sentence_sets[second.chunk_id]
            if first_set.issubset(second_set):
                pairs.append(
                    {
                        "contained_chunk_id": first.chunk_id,
                        "container_chunk_id": second.chunk_id,
                        "contained_sentence_ids": list(first.sentence_ids),
                    }
                )
            elif second_set.issubset(first_set):
                pairs.append(
                    {
                        "contained_chunk_id": second.chunk_id,
                        "container_chunk_id": first.chunk_id,
                        "contained_sentence_ids": list(second.sentence_ids),
                    }
                )
    return pairs


def find_duplicate_overlap_pairs(chunks: list[Chunk], threshold: float) -> list[dict]:
    ordered_chunks = sort_chunks(chunks)
    pairs = []
    for first_index, first in enumerate(ordered_chunks):
        for second in ordered_chunks[first_index + 1 :]:
            if min(second.sentence_ids) > max(first.sentence_ids):
                break
            overlap_ratio = sentence_overlap_ratio(first.sentence_ids, second.sentence_ids)
            if overlap_ratio >= threshold:
                pairs.append(
                    {
                        "chunk_ids": [first.chunk_id, second.chunk_id],
                        "overlap_ratio": overlap_ratio,
                        "intersection_sentence_ids": sorted(
                            set(first.sentence_ids).intersection(second.sentence_ids)
                        ),
                    }
                )
    return pairs


def get_sentence_page_number(item: dict) -> Optional[int]:
    for key in ("page", "page_num", "page_number", "page_id"):
        value = item.get(key)
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    page_index = item.get("page_index")
    if page_index not in (None, ""):
        try:
            return int(page_index) + 1
        except (TypeError, ValueError):
            pass
    return None


def sentence_id_gap(first_sentence_ids: tuple[int, ...], second_sentence_ids: tuple[int, ...]) -> int:
    return min(
        abs(first_id - second_id)
        for first_id in first_sentence_ids
        for second_id in second_sentence_ids
    )


def get_chunk_token_count(
    sentence_ids: tuple[int, ...],
    sentence_text_by_id: dict[int, str],
    model,
    cache: dict[tuple[int, ...], int],
) -> int:
    if sentence_ids in cache:
        return cache[sentence_ids]
    text = build_chunk_text(sentence_ids, sentence_text_by_id)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        token_count = len(text.split())
    else:
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
    cache[sentence_ids] = token_count
    return token_count


def build_chunk_text(sentence_ids: tuple[int, ...], sentence_text_by_id: dict[int, str]) -> str:
    return " ".join(sentence_text_by_id[sentence_id] for sentence_id in sentence_ids)


def chunk_to_payload(chunk: Chunk) -> dict:
    payload = {
        "chunk_id": chunk.chunk_id,
        "sentence_ids": list(chunk.sentence_ids),
        "source_block_ids": list(chunk.source_block_ids),
        "source_chunk_ids": list(chunk.source_chunk_ids),
        "page_span": [min(chunk.page_numbers), max(chunk.page_numbers)] if chunk.page_numbers else None,
        "text": chunk.text,
    }
    if chunk.page_numbers:
        payload["page_numbers"] = list(chunk.page_numbers)
    return payload
