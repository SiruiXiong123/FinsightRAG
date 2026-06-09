import json
from pathlib import Path
from typing import Iterable, Optional, Sequence


BBox = tuple[float, float, float, float]


def load_ocr_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_parsing_res_list(data) -> list[dict]:
    parsing_res_list = data.get("parsing_res_list") if isinstance(data, dict) else data
    if not isinstance(parsing_res_list, list):
        return []
    return [item for item in parsing_res_list if isinstance(item, dict)]


def filter_blocks_by_label(blocks: Iterable[dict], labels: Iterable[str]) -> list[dict]:
    wanted = {normalize_label(label) for label in labels}
    return [block for block in blocks if get_block_label(block) in wanted]


def get_block_label(block: dict) -> str:
    return normalize_label(block.get("block_label", block.get("label")))


def get_block_content(block: dict) -> str:
    content = block.get("block_content", block.get("content", ""))
    return "" if content is None else str(content)


def get_block_bbox(block: dict) -> Optional[BBox]:
    raw_bbox = (
        block.get("block_bbox")
        or block.get("bbox")
        or block.get("coordinate")
        or block.get("box")
    )
    if not isinstance(raw_bbox, Sequence) or len(raw_bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(raw_bbox[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    return normalize_bbox((x1, y1, x2, y2))


def merge_bboxes(bboxes: Iterable[Optional[BBox]]) -> Optional[BBox]:
    valid = [bbox for bbox in bboxes if bbox is not None]
    if not valid:
        return None
    return (
        min(bbox[0] for bbox in valid),
        min(bbox[1] for bbox in valid),
        max(bbox[2] for bbox in valid),
        max(bbox[3] for bbox in valid),
    )


def expand_bbox(bbox: BBox, padding: float) -> BBox:
    return (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)


def clip_bbox(bbox: BBox, width: float, height: float) -> BBox:
    return (
        max(0.0, min(width, bbox[0])),
        max(0.0, min(height, bbox[1])),
        max(0.0, min(width, bbox[2])),
        max(0.0, min(height, bbox[3])),
    )


def normalize_bbox(bbox: BBox) -> BBox:
    x1, y1, x2, y2 = bbox
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def normalize_label(label) -> str:
    return str(label or "").strip().lower()


def block_order_key(block: dict) -> tuple:
    bbox = get_block_bbox(block) or (0.0, 0.0, 0.0, 0.0)
    order = block.get("block_order", block.get("order"))
    global_block_id = block.get("global_block_id")
    block_id = block.get("block_id")
    sortable_order = order if isinstance(order, int) else 1_000_000
    sortable_global = global_block_id if isinstance(global_block_id, int) else 1_000_000
    sortable_block = block_id if isinstance(block_id, int) else 1_000_000
    return (sortable_order, sortable_global, bbox[1], bbox[0], sortable_block)


def horizontal_overlap_ratio(first: BBox, second: BBox) -> float:
    overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    min_width = max(1.0, min(first[2] - first[0], second[2] - second[0]))
    return overlap / min_width
