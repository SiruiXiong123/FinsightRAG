from pathlib import Path
from typing import Iterable, Optional

try:
    from .block_assets import BlockAssetCrop, extract_block_assets
except ImportError:
    from block_assets import BlockAssetCrop, extract_block_assets


DEFAULT_IMAGE_LABELS = {"image", "chart"}


def extract_image_assets(
    json_path: Path,
    pdf_path: Path,
    output_root: Optional[Path] = None,
    page_json_dir: Optional[Path] = None,
    title_labels: Optional[Iterable[str]] = None,
    title_max_distance: Optional[float] = 160.0,
    min_overlap_ratio: float = 0.15,
    padding: float = 8.0,
    dpi: int = 200,
    overwrite: bool = True,
    target_labels: Optional[Iterable[str]] = None,
) -> list[BlockAssetCrop]:
    return extract_block_assets(
        json_path=json_path,
        pdf_path=pdf_path,
        target_labels=target_labels or DEFAULT_IMAGE_LABELS,
        asset_kind="image",
        output_suffix="images",
        file_prefix="image",
        output_root=output_root,
        page_json_dir=page_json_dir,
        title_labels=title_labels,
        title_max_distance=title_max_distance,
        min_overlap_ratio=min_overlap_ratio,
        padding=padding,
        dpi=dpi,
        overwrite=overwrite,
    )
