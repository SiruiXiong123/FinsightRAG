import html
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

try:
    from .paddleocr_blocks import (
        get_block_content,
        get_block_label,
        get_parsing_res_list,
        load_ocr_json,
        normalize_label,
    )
except ImportError:
    from paddleocr_blocks import (
        get_block_content,
        get_block_label,
        get_parsing_res_list,
        load_ocr_json,
        normalize_label,
    )


DEFAULT_DROP_LABELS = {
    "table",
    "chart",
    "image",
    "seal",
    "formula",
}
DEFAULT_DROP_LABEL_FRAGMENTS = ("image",)
TITLE_PREFIXES = {
    "doc_title": "#",
    "paragraph_title": "##",
}


@dataclass(frozen=True)
class OcrDocumentPair:
    json_path: Path
    md_path: Optional[Path]


@dataclass(frozen=True)
class TextMarkdownResult:
    json_path: Path
    md_path: Optional[Path]
    output_path: Path
    kept_blocks: int
    dropped_blocks: int
    kept_labels: Counter
    dropped_labels: Counter


def discover_document_pairs(
    ocr_output_dir: Path,
    recursive: bool = False,
    require_md: bool = True,
) -> list[OcrDocumentPair]:
    """
    Find PaddleOCR-VL JSON/Markdown document pairs.

    By default this only scans the top-level output directory so page_0001.json
    files inside per-document folders are not treated as whole documents.
    """
    pattern = "**/*.json" if recursive else "*.json"
    pairs: list[OcrDocumentPair] = []
    for json_path in sorted(ocr_output_dir.glob(pattern)):
        if json_path.stem.endswith("_text"):
            continue
        md_path = json_path.with_suffix(".md")
        if not md_path.exists():
            if require_md:
                continue
            md_path = None
        pairs.append(OcrDocumentPair(json_path=json_path, md_path=md_path))
    return pairs


def generate_text_markdown_for_dir(
    ocr_output_dir: Path,
    output_dir: Optional[Path] = None,
    recursive: bool = False,
    require_md: bool = True,
    keep_labels: Optional[Iterable[str]] = None,
    drop_labels: Optional[Iterable[str]] = None,
    format_titles: bool = True,
    overwrite: bool = True,
) -> list[TextMarkdownResult]:
    pairs = discover_document_pairs(
        ocr_output_dir=ocr_output_dir,
        recursive=recursive,
        require_md=require_md,
    )
    return [
        generate_text_markdown(
            json_path=pair.json_path,
            md_path=pair.md_path,
            output_dir=output_dir,
            keep_labels=keep_labels,
            drop_labels=drop_labels,
            format_titles=format_titles,
            overwrite=overwrite,
        )
        for pair in pairs
    ]


def generate_text_markdown(
    json_path: Path,
    md_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    keep_labels: Optional[Iterable[str]] = None,
    drop_labels: Optional[Iterable[str]] = None,
    format_titles: bool = True,
    overwrite: bool = True,
) -> TextMarkdownResult:
    json_path = json_path.resolve()
    if md_path is not None:
        md_path = md_path.resolve()
        if not md_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {md_path}")

    output_base_dir = output_dir.resolve() if output_dir else json_path.parent
    output_base_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_base_dir / f"{json_path.stem}_text.md"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    data = load_ocr_json(json_path)
    blocks = get_parsing_res_list(data)
    keep_label_set = normalize_label_set(keep_labels)
    drop_label_set = set(DEFAULT_DROP_LABELS)
    if drop_labels is not None:
        drop_label_set.update(normalize_label_set(drop_labels) or set())

    kept_chunks: list[str] = []
    kept_labels: Counter = Counter()
    dropped_labels: Counter = Counter()

    for block in blocks:
        label = get_block_label(block)
        content = normalize_content(get_block_content(block))
        if should_keep_block(
            label=label,
            content=content,
            keep_labels=keep_label_set,
            drop_labels=drop_label_set,
        ):
            kept_labels[label or "<missing>"] += 1
            kept_chunks.append(format_block(label, content, format_titles=format_titles))
        else:
            dropped_labels[label or "<missing>"] += 1

    text = join_markdown_chunks(kept_chunks)
    output_path.write_text(text, encoding="utf-8")

    return TextMarkdownResult(
        json_path=json_path,
        md_path=md_path,
        output_path=output_path,
        kept_blocks=sum(kept_labels.values()),
        dropped_blocks=sum(dropped_labels.values()),
        kept_labels=kept_labels,
        dropped_labels=dropped_labels,
    )


def should_keep_block(
    label: str,
    content: str,
    keep_labels: Optional[set[str]],
    drop_labels: set[str],
) -> bool:
    if not content:
        return False
    if keep_labels is not None:
        return label in keep_labels
    if label in drop_labels:
        return False
    return not any(fragment in label for fragment in DEFAULT_DROP_LABEL_FRAGMENTS)


def normalize_label_set(labels: Optional[Iterable[str]]) -> Optional[set[str]]:
    if labels is None:
        return None
    return {normalize_label(label) for label in labels if normalize_label(label)}


def normalize_label(label) -> str:
    return str(label or "").strip().lower()


def normalize_content(content) -> str:
    if content is None:
        return ""
    text = html.unescape(str(content))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def format_block(label: str, content: str, format_titles: bool = True) -> str:
    if format_titles and label in TITLE_PREFIXES:
        title = collapse_inline_text(content)
        if title:
            return f"{TITLE_PREFIXES[label]} {title}"
    return content


def collapse_inline_text(text: str) -> str:
    return " ".join(part.strip() for part in text.split() if part.strip())


def join_markdown_chunks(chunks: Sequence[str]) -> str:
    cleaned = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    if not cleaned:
        return ""
    return "\n\n".join(cleaned) + "\n"
