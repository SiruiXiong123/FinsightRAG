import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

try:
    from .paddleocr_blocks import (
        BBox,
        block_order_key,
        clip_bbox,
        expand_bbox,
        filter_blocks_by_label,
        get_block_bbox,
        get_block_content,
        get_block_label,
        get_parsing_res_list,
        horizontal_overlap_ratio,
        load_ocr_json,
        merge_bboxes,
        normalize_label,
    )
except ImportError:
    from paddleocr_blocks import (
        BBox,
        block_order_key,
        clip_bbox,
        expand_bbox,
        filter_blocks_by_label,
        get_block_bbox,
        get_block_content,
        get_block_label,
        get_parsing_res_list,
        horizontal_overlap_ratio,
        load_ocr_json,
        merge_bboxes,
        normalize_label,
    )


DEFAULT_TITLE = "defaulttitle"


@dataclass(frozen=True)
class PageBlocks:
    page_index: int
    width: float
    height: float
    blocks: list[dict]


@dataclass(frozen=True)
class BlockAssetCrop:
    asset_kind: str
    asset_label: str
    asset_index: int
    page_index: int
    title: str
    title_label: Optional[str]
    asset_bbox: BBox
    title_bbox: Optional[BBox]
    crop_bbox: BBox
    output_path: Path


def write_asset_manifest(crops: list[BlockAssetCrop], manifest_path: Path, asset_dir: Optional[Path] = None) -> None:
    asset_dir = asset_dir.resolve() if asset_dir else manifest_path.parent.resolve()
    records = [asset_crop_to_manifest_record(crop, asset_dir) for crop in crops]
    manifest_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def asset_crop_to_manifest_record(crop: BlockAssetCrop, asset_dir: Path) -> dict:
    try:
        asset_path = str(crop.output_path.resolve().relative_to(asset_dir)).replace("\\", "/")
    except ValueError:
        asset_path = str(crop.output_path.resolve())
    return {
        "asset_index": crop.asset_index,
        "asset_kind": crop.asset_kind,
        "asset_label": crop.asset_label,
        "filename": crop.output_path.name,
        "page": crop.page_index + 1,
        "page_index": crop.page_index,
        "bbox": bbox_to_list(crop.asset_bbox),
        "asset_bbox": bbox_to_list(crop.asset_bbox),
        "title": crop.title,
        "title_label": crop.title_label,
        "title_bbox": bbox_to_list(crop.title_bbox) if crop.title_bbox else None,
        "crop_bbox": bbox_to_list(crop.crop_bbox),
        "asset_path": asset_path,
    }


def bbox_to_list(bbox: BBox) -> list[float]:
    return [float(value) for value in bbox]


def extract_block_assets(
    json_path: Path,
    pdf_path: Path,
    target_labels: Iterable[str],
    asset_kind: str,
    output_suffix: str,
    file_prefix: str,
    output_root: Optional[Path] = None,
    page_json_dir: Optional[Path] = None,
    title_labels: Optional[Iterable[str]] = None,
    title_max_distance: Optional[float] = 160.0,
    min_overlap_ratio: float = 0.15,
    padding: float = 8.0,
    dpi: int = 200,
    overwrite: bool = True,
) -> list[BlockAssetCrop]:
    json_path = Path(json_path).resolve()
    pdf_path = Path(pdf_path).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"OCR JSON file not found: {json_path}")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    output_base = Path(output_root).resolve() if output_root else json_path.parent
    asset_dir = output_base / f"{json_path.stem}_{output_suffix}"
    asset_dir.mkdir(parents=True, exist_ok=True)

    pages = load_page_blocks(json_path=json_path, page_json_dir=page_json_dir)
    if not pages:
        return []

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for PDF cropping. Install it in llm_env with: "
            "python -m pip install PyMuPDF"
        ) from exc

    title_label_set = {
        normalize_label(label)
        for label in (title_labels if title_labels is not None else [])
    }
    crops: list[BlockAssetCrop] = []
    asset_index = 0
    with fitz.open(pdf_path) as pdf_doc:
        for page in pages:
            if page.page_index < 0 or page.page_index >= pdf_doc.page_count:
                continue

            asset_blocks = filter_blocks_by_label(page.blocks, target_labels)
            asset_blocks = sorted(asset_blocks, key=block_order_key)
            for asset_block in asset_blocks:
                asset_bbox = get_block_bbox(asset_block)
                if asset_bbox is None:
                    continue

                asset_index += 1
                title_block = find_title_block(
                    asset_block=asset_block,
                    candidate_blocks=page.blocks,
                    title_labels=title_label_set,
                    title_max_distance=title_max_distance,
                    min_overlap_ratio=min_overlap_ratio,
                )
                title_bbox = get_block_bbox(title_block) if title_block else None
                merged_bbox = merge_bboxes([asset_bbox, title_bbox]) or asset_bbox
                crop_bbox = clip_bbox(expand_bbox(merged_bbox, padding), page.width, page.height)
                title = clean_title(get_block_content(title_block)) if title_block else DEFAULT_TITLE
                if not title:
                    title = DEFAULT_TITLE

                output_path = asset_dir / build_asset_filename(
                    file_prefix=file_prefix,
                    asset_index=asset_index,
                    page_index=page.page_index,
                    title=title,
                )
                if output_path.exists() and not overwrite:
                    raise FileExistsError(f"Output file already exists: {output_path}")

                crop_pdf_region_to_png(
                    pdf_doc=pdf_doc,
                    page_index=page.page_index,
                    ocr_page_width=page.width,
                    ocr_page_height=page.height,
                    crop_bbox=crop_bbox,
                    output_path=output_path,
                    dpi=dpi,
                )
                crops.append(
                    BlockAssetCrop(
                        asset_kind=asset_kind,
                        asset_label=get_block_label(asset_block),
                        asset_index=asset_index,
                        page_index=page.page_index,
                        title=title,
                        title_label=get_block_label(title_block) if title_block else None,
                        asset_bbox=asset_bbox,
                        title_bbox=title_bbox,
                        crop_bbox=crop_bbox,
                        output_path=output_path,
                    )
                )
    write_asset_manifest(crops, asset_dir / "asset_manifest.jsonl", asset_dir=asset_dir)
    return crops


def load_page_blocks(json_path: Path, page_json_dir: Optional[Path] = None) -> list[PageBlocks]:
    if page_json_dir is None:
        candidate_dir = json_path.with_suffix("")
        page_json_dir = candidate_dir if candidate_dir.exists() else None

    if page_json_dir is not None and Path(page_json_dir).exists():
        pages = []
        for page_json in sorted(Path(page_json_dir).glob("page_*.json")):
            page_data = load_ocr_json(page_json)
            page_index = get_page_index(page_data, page_json)
            width = float(page_data.get("width") or 0)
            height = float(page_data.get("height") or 0)
            blocks = [dict(block, _page_index=page_index) for block in get_parsing_res_list(page_data)]
            pages.append(PageBlocks(page_index=page_index, width=width, height=height, blocks=blocks))
        if pages:
            return pages

    data = load_ocr_json(json_path)
    page_index = data.get("page_index")
    if page_index is not None:
        width = float(data.get("width") or 0)
        height = float(data.get("height") or 0)
        blocks = [dict(block, _page_index=int(page_index)) for block in get_parsing_res_list(data)]
        return [PageBlocks(page_index=int(page_index), width=width, height=height, blocks=blocks)]

    return load_pages_from_layout(data)


def load_pages_from_layout(data) -> list[PageBlocks]:
    layout_pages = data.get("layout_det_res") if isinstance(data, dict) else None
    if not isinstance(layout_pages, list):
        return []

    width = float(data.get("width") or 0)
    height = float(data.get("height") or 0)
    pages = []
    for page_index, layout_page in enumerate(layout_pages):
        boxes = layout_page.get("boxes") if isinstance(layout_page, dict) else None
        if not isinstance(boxes, list):
            continue
        blocks = []
        for block_id, box in enumerate(boxes):
            if not isinstance(box, dict):
                continue
            block = {
                "block_label": box.get("label"),
                "block_bbox": box.get("coordinate"),
                "block_order": box.get("order"),
                "block_id": block_id,
                "_page_index": page_index,
            }
            blocks.append(block)
        pages.append(PageBlocks(page_index=page_index, width=width, height=height, blocks=blocks))
    return pages


def get_page_index(page_data: dict, page_json: Path) -> int:
    page_index = page_data.get("page_index")
    if isinstance(page_index, int):
        return page_index
    match = re.search(r"page_(\d+)", page_json.stem)
    if match:
        return max(0, int(match.group(1)) - 1)
    return 0


def find_title_block(
    asset_block: dict,
    candidate_blocks: Iterable[dict],
    title_labels: set[str],
    title_max_distance: Optional[float],
    min_overlap_ratio: float,
) -> Optional[dict]:
    asset_bbox = get_block_bbox(asset_block)
    if asset_bbox is None:
        return None

    above = []
    below = []
    asset_center_y = (asset_bbox[1] + asset_bbox[3]) / 2
    for block in candidate_blocks:
        if block is asset_block or not is_title_block(block, title_labels):
            continue
        title_bbox = get_block_bbox(block)
        if title_bbox is None:
            continue
        if horizontal_overlap_ratio(asset_bbox, title_bbox) < min_overlap_ratio:
            continue

        title_center_y = (title_bbox[1] + title_bbox[3]) / 2
        if title_center_y < asset_center_y:
            distance = max(0.0, asset_bbox[1] - title_bbox[3])
            if title_max_distance is None or distance <= title_max_distance:
                above.append((distance, -title_bbox[3], block))
        elif title_center_y > asset_center_y:
            distance = max(0.0, title_bbox[1] - asset_bbox[3])
            if title_max_distance is None or distance <= title_max_distance:
                below.append((distance, title_bbox[1], block))

    if above:
        return sorted(above, key=lambda item: (item[0], item[1]))[0][2]
    if below:
        return sorted(below, key=lambda item: (item[0], item[1]))[0][2]
    return None


def is_title_block(block: dict, title_labels: set[str]) -> bool:
    label = get_block_label(block)
    if label.endswith("_title"):
        return True
    if label in title_labels:
        return True
    return any(str(key).lower().endswith("_title") and bool(value) for key, value in block.items())


def crop_pdf_region_to_png(
    pdf_doc,
    page_index: int,
    ocr_page_width: float,
    ocr_page_height: float,
    crop_bbox: BBox,
    output_path: Path,
    dpi: int,
) -> None:
    import fitz

    page = pdf_doc.load_page(page_index)
    pdf_width = page.rect.width
    pdf_height = page.rect.height
    x_scale = pdf_width / ocr_page_width
    y_scale = pdf_height / ocr_page_height
    clip = fitz.Rect(
        crop_bbox[0] * x_scale,
        crop_bbox[1] * y_scale,
        crop_bbox[2] * x_scale,
        crop_bbox[3] * y_scale,
    )
    zoom = dpi / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip, alpha=False)
    pixmap.save(output_path)


def clean_title(title: str) -> str:
    return " ".join(str(title or "").split()).strip()


def build_asset_filename(
    file_prefix: str,
    asset_index: int,
    page_index: int,
    title: str,
) -> str:
    slug = sanitize_filename(title) or DEFAULT_TITLE
    return f"{file_prefix}_{asset_index:04d}_page_{page_index + 1:04d}_{slug}.png"


def sanitize_filename(value: str, max_length: int = 80) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", value.strip(), flags=re.UNICODE)
    slug = re.sub(r"_+", "_", slug).strip("_.")
    return (slug or DEFAULT_TITLE)[:max_length]
