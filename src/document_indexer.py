import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

try:
    from .rag_config import RagConfig
except ImportError:
    from rag_config import RagConfig


DEFAULT_TEXT_CHUNK_SUFFIX = "chunk_merge"
DEFAULT_TABLE_ENRICHMENT_SUFFIX = "table_enrichment"
DEFAULT_IMAGE_ENRICHMENT_SUFFIX = "image_enrichment"
DEFAULT_EMBED_BATCH_SIZE = 32


@dataclass(frozen=True)
class AssetBlockInfo:
    filename: str
    page: int
    bbox: list[float]
    asset_path: Path


@dataclass(frozen=True)
class DocumentIndexConfig:
    document_id: str
    source_file: str
    text_chunks_path: Path
    table_enrichment_path: Path
    image_enrichment_path: Path
    table_asset_dir: Path
    image_asset_dir: Path
    table_asset_manifest_path: Path
    image_asset_manifest_path: Path
    index_root: Path
    project_root: Path
    embedding_model: str
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    device: Optional[str] = None
    overwrite: bool = True
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def index_dir(self) -> Path:
        return self.index_root / self.document_id


def build_document_indexes(config: DocumentIndexConfig) -> dict:
    config = normalize_config_paths(config)
    config.index_dir.mkdir(parents=True, exist_ok=True)

    table_asset_info = load_asset_manifest(config.table_asset_manifest_path, config.table_asset_dir)
    image_asset_info = load_asset_manifest(config.image_asset_manifest_path, config.image_asset_dir)

    text_records = build_text_records(
        text_chunks_path=config.text_chunks_path,
        source_file=config.source_file,
    )
    table_records = build_asset_records(
        enrichment_path=config.table_enrichment_path,
        source_file=config.source_file,
        modality="table",
        asset_info_by_filename=table_asset_info,
        project_root=config.project_root,
    )
    image_records = build_asset_records(
        enrichment_path=config.image_enrichment_path,
        source_file=config.source_file,
        modality="image",
        asset_info_by_filename=image_asset_info,
        project_root=config.project_root,
    )

    model = load_embedding_model(config.embedding_model, config.device)
    modalities = {}
    modalities["text"] = write_modality_index(
        records=text_records,
        faiss_path=config.index_dir / "text.faiss",
        metadata_path=config.index_dir / "text_metadata.jsonl",
        model=model,
        batch_size=config.embed_batch_size,
        overwrite=config.overwrite,
        project_root=config.project_root,
    )
    modalities["table"] = write_modality_index(
        records=table_records,
        faiss_path=config.index_dir / "table.faiss",
        metadata_path=config.index_dir / "table_metadata.jsonl",
        model=model,
        batch_size=config.embed_batch_size,
        overwrite=config.overwrite,
        project_root=config.project_root,
    )
    modalities["image"] = write_modality_index(
        records=image_records,
        faiss_path=config.index_dir / "image.faiss",
        metadata_path=config.index_dir / "image_metadata.jsonl",
        model=model,
        batch_size=config.embed_batch_size,
        overwrite=config.overwrite,
        project_root=config.project_root,
    )

    manifest = {
        "document_id": config.document_id,
        "source_file": normalize_manifest_path(config.source_file, config.project_root),
        "index_dir": make_metadata_path(config.index_dir, config.project_root),
        "embedding_model": config.embedding_model,
        "modalities": modalities,
        "metadata": dict(config.metadata),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "text_chunks_path": make_metadata_path(config.text_chunks_path, config.project_root),
            "table_enrichment_path": make_metadata_path(config.table_enrichment_path, config.project_root),
            "image_enrichment_path": make_metadata_path(config.image_enrichment_path, config.project_root),
            "table_asset_dir": make_metadata_path(config.table_asset_dir, config.project_root),
            "image_asset_dir": make_metadata_path(config.image_asset_dir, config.project_root),
            "table_asset_manifest_path": make_metadata_path(config.table_asset_manifest_path, config.project_root),
            "image_asset_manifest_path": make_metadata_path(config.image_asset_manifest_path, config.project_root),
        },
    }
    (config.index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def normalize_config_paths(config: DocumentIndexConfig) -> DocumentIndexConfig:
    return DocumentIndexConfig(
        document_id=config.document_id,
        source_file=config.source_file,
        text_chunks_path=config.text_chunks_path.resolve(),
        table_enrichment_path=config.table_enrichment_path.resolve(),
        image_enrichment_path=config.image_enrichment_path.resolve(),
        table_asset_dir=config.table_asset_dir.resolve(),
        image_asset_dir=config.image_asset_dir.resolve(),
        table_asset_manifest_path=config.table_asset_manifest_path.resolve(),
        image_asset_manifest_path=config.image_asset_manifest_path.resolve(),
        index_root=config.index_root.resolve(),
        project_root=config.project_root.resolve(),
        embedding_model=config.embedding_model,
        embed_batch_size=config.embed_batch_size,
        device=config.device,
        overwrite=config.overwrite,
        metadata=dict(config.metadata),
    )


def build_text_records(
    text_chunks_path: Path,
    source_file: str,
) -> list[dict]:
    data = read_json(text_chunks_path)
    records = []
    for index, chunk in enumerate(data.get("chunks", []), start=1):
        content = str(chunk.get("text") or "").strip()
        if not content:
            continue
        page_span = chunk.get("page_span")
        if page_span is None:
            raise ValueError(
                f"Text chunk {chunk.get('chunk_id') or index} has no page_span. "
                "Run text chunking with --page-json-dir or --sentence-split-file first."
            )
        records.append(
            {
                "id": f"text_{index:06d}",
                "source_file": source_file,
                "modality": "text",
                "page_span": page_span,
                "content": content,
                "hash": md5_text(content),
            }
        )
    return records


def build_asset_records(
    enrichment_path: Path,
    source_file: str,
    modality: str,
    asset_info_by_filename: dict[str, AssetBlockInfo],
    project_root: Path,
) -> list[dict]:
    data = read_json(enrichment_path)
    records = []
    for index, item in enumerate(data.get("results", []), start=1):
        filename = str(item.get("filename") or "").strip()
        content = str(item.get("content") or "").strip()
        if not filename or not content:
            continue
        asset_info = asset_info_by_filename.get(filename)
        if asset_info is None:
            raise ValueError(
                f"{modality} enrichment filename is missing from asset manifest: {filename}"
            )
        records.append(
            {
                "id": f"{modality}_{index:06d}",
                "source_file": source_file,
                "modality": modality,
                "page": asset_info.page,
                "bbox": asset_info.bbox,
                "asset_path": make_metadata_path(asset_info.asset_path, project_root),
                "title": str(item.get("title") or "").strip(),
                "content": content,
                "hash": md5_text(content),
            }
        )
    return records


def load_asset_manifest(manifest_path: Path, asset_dir: Path) -> dict[str, AssetBlockInfo]:
    records = {}
    for line_number, line in enumerate(Path(manifest_path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        filename = str(item.get("filename") or "").strip()
        if not filename:
            raise ValueError(f"Missing filename in asset manifest line {line_number}: {manifest_path}")
        raw_asset_path = Path(str(item.get("asset_path") or filename))
        asset_path = raw_asset_path if raw_asset_path.is_absolute() else asset_dir / raw_asset_path
        bbox = item.get("bbox") or item.get("asset_bbox")
        if bbox is None:
            raise ValueError(f"Missing bbox for {filename} in asset manifest: {manifest_path}")
        page = item.get("page")
        if page is None:
            raise ValueError(f"Missing page for {filename} in asset manifest: {manifest_path}")
        records[filename] = AssetBlockInfo(
            filename=filename,
            page=int(page),
            bbox=[float(value) for value in bbox],
            asset_path=asset_path,
        )
    return records


def write_modality_index(
    records: list[dict],
    faiss_path: Path,
    metadata_path: Path,
    model,
    batch_size: int,
    overwrite: bool,
    project_root: Path,
) -> dict:
    for path in (faiss_path, metadata_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {path}")

    records = with_faiss_ids(records)
    embeddings = encode_records(records, model, batch_size)
    write_faiss_index(embeddings, faiss_path)
    write_jsonl(records, metadata_path)
    return {
        "count": len(records),
        "faiss_path": make_metadata_path(faiss_path, project_root),
        "metadata_path": make_metadata_path(metadata_path, project_root),
        "dimension": int(embeddings.shape[1]) if len(embeddings.shape) == 2 else 0,
    }


def with_faiss_ids(records: list[dict]) -> list[dict]:
    return [
        {
            **record,
            "faiss_id": index,
        }
        for index, record in enumerate(records, start=1)
    ]


def encode_records(records: list[dict], model, batch_size: int):
    import numpy as np

    texts = [embedding_text_for_record(record) for record in records]
    if not texts:
        dimension = int(model.get_sentence_embedding_dimension() or 0)
        return np.zeros((0, dimension), dtype="float32")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.astype("float32")


def embedding_text_for_record(record: dict) -> str:
    if record.get("modality") in {"table", "image"}:
        title = str(record.get("title") or "").strip()
        content = str(record.get("content") or "").strip()
        return f"{title}\n{content}".strip()
    return str(record.get("content") or "").strip()


def write_faiss_index(embeddings, faiss_path: Path) -> None:
    import faiss
    import numpy as np

    dimension = int(embeddings.shape[1]) if len(embeddings.shape) == 2 else 0
    if dimension <= 0:
        raise ValueError("Cannot write a FAISS index with unknown embedding dimension.")
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
    if embeddings.shape[0] > 0:
        vector_ids = np.arange(1, embeddings.shape[0] + 1, dtype="int64")
        index.add_with_ids(embeddings, vector_ids)
    faiss.write_index(index, str(faiss_path))


def write_jsonl(records: list[dict], metadata_path: Path) -> None:
    metadata_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def load_embedding_model(embedding_model: str, device: Optional[str]):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for index building. "
            "Install it in llm_env with: python -m pip install sentence-transformers"
        ) from exc
    if device:
        return SentenceTransformer(embedding_model, device=device)
    return SentenceTransformer(embedding_model)


def md5_text(text: str) -> str:
    return hashlib.md5(str(text or "").encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_metadata_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def normalize_manifest_path(path: str, project_root: Path) -> str:
    raw_path = Path(str(path))
    if raw_path.is_absolute():
        return make_metadata_path(raw_path, project_root)
    return str(raw_path).replace("\\", "/")


def infer_document_id(source_file: Optional[Path], ocr_output_dir: Optional[Path]) -> str:
    if source_file:
        return Path(source_file).stem
    if ocr_output_dir:
        return Path(ocr_output_dir).stem
    raise ValueError("Cannot infer document_id. Pass --document-id or --source-file.")


def resolve_default_paths(
    project_root: Path,
    config_path: Optional[Path],
    document_id: Optional[str],
    ocr_output_dir: Optional[Path],
    source_file: Optional[Path],
) -> dict:
    rag_config = RagConfig.load(str(config_path) if config_path else None)
    indexing_settings = rag_config.values.get("indexing", {})
    if not isinstance(indexing_settings, dict):
        indexing_settings = {}
    configured_source = source_file or rag_config.get_path("input_file")
    doc_id = document_id or infer_document_id(configured_source, None)
    output_dir = ocr_output_dir or rag_config.get_path("paddleocr_output_dir") or project_root / "data" / "output"
    index_root_value = indexing_settings.get("index_root") or "indexes"
    index_root = Path(index_root_value)
    if not index_root.is_absolute():
        index_root = project_root / index_root
    return {
        "document_id": doc_id,
        "source_file": make_metadata_path(configured_source, project_root) if configured_source else f"{doc_id}.pdf",
        "text_chunks_path": output_dir / f"{doc_id}_{DEFAULT_TEXT_CHUNK_SUFFIX}.json",
        "table_enrichment_path": output_dir / f"{doc_id}_{DEFAULT_TABLE_ENRICHMENT_SUFFIX}.json",
        "image_enrichment_path": output_dir / f"{doc_id}_{DEFAULT_IMAGE_ENRICHMENT_SUFFIX}.json",
        "table_asset_dir": output_dir / f"{doc_id}_tables",
        "image_asset_dir": output_dir / f"{doc_id}_images",
        "table_asset_manifest_path": output_dir / f"{doc_id}_tables" / "asset_manifest.jsonl",
        "image_asset_manifest_path": output_dir / f"{doc_id}_images" / "asset_manifest.jsonl",
        "index_root": index_root,
        "embedding_model": rag_config.get("embedding_model", "BAAI/bge-m3"),
        "embed_batch_size": int(indexing_settings.get("embed_batch_size") or DEFAULT_EMBED_BATCH_SIZE),
        "device": indexing_settings.get("device") or None,
        "metadata": {},
    }


def require_existing_paths(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required input paths: " + ", ".join(str(path) for path in missing))
