import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from .paths import default_project_root
    from .rag_config import RagConfig
except ImportError:
    from paths import default_project_root
    from rag_config import RagConfig


SUPPORTED_MONTAGE_KINDS = ("table", "image")
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 1280 * 28 * 28


@dataclass(frozen=True)
class MontageConfig:
    enabled: bool = True
    layout: str = "vertical"
    columns: int = 1
    target_width: Optional[int] = None
    cell_width: Optional[int] = None
    max_pixels: int = DEFAULT_MAX_PIXELS
    min_pixels: int = DEFAULT_MIN_PIXELS
    padding: int = 24
    label_height: int = 48
    background: str = "white"
    output_format: str = "jpg"
    jpeg_quality: int = 95


@dataclass(frozen=True)
class EvidenceBuilderConfig:
    project_root: Path
    output_dir: Path
    table_montage: MontageConfig
    image_montage: MontageConfig


class EvidenceBuilder:
    def __init__(self, config: EvidenceBuilderConfig):
        self.config = config

    @classmethod
    def from_config_path(
        cls,
        config_path: Optional[Path | str] = None,
        project_root: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
    ) -> "EvidenceBuilder":
        root = Path(project_root or default_project_root()).resolve()
        rag_config = RagConfig.load(str(config_path) if config_path else None)
        cfg = build_evidence_builder_config(
            rag_config=rag_config,
            project_root=root,
            output_dir=Path(output_dir).resolve() if output_dir else None,
        )
        return cls(cfg)

    def build(self, retriever_result: dict, output_dir: Optional[Path | str] = None) -> dict:
        selected_output_dir = Path(output_dir).resolve() if output_dir else self.config.output_dir
        selected_output_dir.mkdir(parents=True, exist_ok=True)

        contexts = list(retriever_result.get("contexts") or [])
        text_contexts = assign_evidence_ids(
            normalize_contexts(split_contexts(contexts, "text")),
            prefix="E",
        )
        table_contexts = assign_evidence_ids(
            normalize_contexts(split_contexts(contexts, "table")),
            prefix="T",
        )
        image_contexts = assign_evidence_ids(
            normalize_contexts(split_contexts(contexts, "image")),
            prefix="I",
        )

        document_id = str(retriever_result.get("document_id") or "document")
        query = str(retriever_result.get("query") or "")
        query_key = stable_query_key(query)
        table_montage = self.build_montage(
            evidence_items=table_contexts,
            kind="table",
            document_id=document_id,
            query_key=query_key,
            output_dir=selected_output_dir,
        )
        image_montage = self.build_montage(
            evidence_items=image_contexts,
            kind="image",
            document_id=document_id,
            query_key=query_key,
            output_dir=selected_output_dir,
        )

        return {
            "query": query,
            "document_id": document_id,
            "retrieval_mode": retriever_result.get("retrieval_mode"),
            "fallback_triggered": bool(retriever_result.get("fallback_triggered")),
            "retriever_result": retriever_result,
            "contexts_by_modality": {
                "text": text_contexts,
                "table": table_contexts,
                "image": image_contexts,
            },
            "montages": {
                "table": table_montage,
                "image": image_montage,
            },
            "vlm_inputs": {
                "text_contexts": text_contexts,
                "table_contexts": table_contexts,
                "image_contexts": image_contexts,
                "image_paths": montage_paths(table_montage, image_montage),
            },
            "trace": {
                "num_text": len(text_contexts),
                "num_table": len(table_contexts),
                "num_image": len(image_contexts),
                "table_montage_created": table_montage is not None,
                "image_montage_created": image_montage is not None,
            },
        }

    def build_montage(
        self,
        evidence_items: list[dict],
        kind: str,
        document_id: str,
        query_key: str,
        output_dir: Path,
    ) -> Optional[dict]:
        kind = normalize_kind(kind)
        config = self.config.table_montage if kind == "table" else self.config.image_montage
        if not config.enabled:
            return None

        image_items = [
            (item, resolve_crop_path(item, self.config.project_root))
            for item in evidence_items
        ]
        image_items = [(item, path) for item, path in image_items if path and path.exists()]
        if not image_items:
            return None

        output_path = output_dir / montage_filename(document_id, kind, query_key, config.output_format)
        result = write_montage(image_items=image_items, config=config, output_path=output_path)
        return {
            "path": project_relative_path(output_path, self.config.project_root),
            "absolute_path": str(output_path.resolve()),
            "kind": kind,
            "layout": config.layout,
            "evidence_ids": [item["evidence_id"] for item, _ in image_items],
            "width": result["width"],
            "height": result["height"],
            "pixel_count": result["pixel_count"],
            "source_count": len(image_items),
        }


def split_contexts(contexts: list[dict], modality: str) -> list[dict]:
    return [
        context
        for context in contexts
        if str(context.get("modality") or "").strip().lower() == modality
    ]


def normalize_kind(kind: str) -> str:
    kind = str(kind or "").strip().lower()
    if kind not in SUPPORTED_MONTAGE_KINDS:
        raise ValueError(f"Unsupported evidence montage kind: {kind}")
    return kind


def normalize_contexts(contexts: list[dict]) -> list[dict]:
    return [normalize_context(context) for context in contexts]


def normalize_context(context: dict) -> dict:
    crop_path = context.get("crop_path") or context.get("asset_path") or context.get("image_path")
    return {
        "id": context.get("id"),
        "document_id": context.get("document_id"),
        "modality": context.get("modality"),
        "score": context.get("score"),
        "page": context.get("page"),
        "page_span": context.get("page_span"),
        "bbox": context.get("bbox"),
        "title": context.get("title"),
        "content": context.get("content"),
        "summary": context.get("summary") or context.get("content"),
        "json": context.get("json"),
        "crop_path": crop_path,
        "asset_path": context.get("asset_path") or crop_path,
    }


def assign_evidence_ids(contexts: list[dict], prefix: str) -> list[dict]:
    return [
        {
            **context,
            "evidence_id": f"{prefix}{index}",
        }
        for index, context in enumerate(contexts, start=1)
    ]


def montage_paths(*montages: Optional[dict]) -> list[str]:
    return [
        montage["path"]
        for montage in montages
        if montage is not None and montage.get("path")
    ]


def build_evidence_builder_config(
    rag_config: RagConfig,
    project_root: Path,
    output_dir: Optional[Path] = None,
) -> EvidenceBuilderConfig:
    values = rag_config.values or {}
    section = values.get("evidence", {})
    if not isinstance(section, dict):
        section = {}
    configured_output = output_dir or resolve_project_path(
        section.get("output_dir") or values.get("paddleocr_output_dir") or "data/output",
        project_root,
    )
    return EvidenceBuilderConfig(
        project_root=project_root.resolve(),
        output_dir=configured_output.resolve(),
        table_montage=montage_config_from_dict(
            section.get("table_montage", {}),
            defaults=MontageConfig(
                layout="vertical",
                target_width=1400,
                columns=1,
            ),
        ),
        image_montage=montage_config_from_dict(
            section.get("image_montage", {}),
            defaults=MontageConfig(
                layout="grid",
                columns=2,
                cell_width=900,
            ),
        ),
    )


def montage_config_from_dict(raw: object, defaults: MontageConfig) -> MontageConfig:
    settings = raw if isinstance(raw, dict) else {}
    return MontageConfig(
        enabled=to_bool(settings.get("enabled", defaults.enabled)),
        layout=str(settings.get("layout", defaults.layout)),
        columns=max(1, to_int(settings.get("columns", defaults.columns))),
        target_width=optional_int(settings.get("target_width", defaults.target_width)),
        cell_width=optional_int(settings.get("cell_width", defaults.cell_width)),
        max_pixels=max(1, to_int(settings.get("max_pixels", defaults.max_pixels))),
        min_pixels=max(1, to_int(settings.get("min_pixels", defaults.min_pixels))),
        padding=max(0, to_int(settings.get("padding", defaults.padding))),
        label_height=max(0, to_int(settings.get("label_height", defaults.label_height))),
        background=str(settings.get("background", defaults.background)),
        output_format=str(settings.get("output_format", defaults.output_format)).lower(),
        jpeg_quality=max(1, min(100, to_int(settings.get("jpeg_quality", defaults.jpeg_quality)))),
    )


def write_montage(
    image_items: list[tuple[dict, Path]],
    config: MontageConfig,
    output_path: Path,
) -> dict:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for evidence montage generation. "
            "Install it with: python -m pip install Pillow"
        ) from exc

    source_images = [
        (item, Image.open(path).convert("RGB"))
        for item, path in image_items
    ]
    if config.layout == "vertical":
        montage = build_vertical_montage(source_images, config, Image, ImageDraw, ImageFont)
    elif config.layout == "grid":
        montage = build_grid_montage(source_images, config, Image, ImageDraw, ImageFont)
    else:
        raise ValueError(f"Unsupported montage layout: {config.layout}")

    montage = fit_pixel_budget(montage, config.max_pixels, config.min_pixels, Image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(montage, output_path, config)
    return {
        "width": montage.width,
        "height": montage.height,
        "pixel_count": montage.width * montage.height,
    }


def build_vertical_montage(source_images, config, image_module, draw_module, font_module):
    target_width = int(config.target_width or config.cell_width or max(image.width for _, image in source_images))
    cells = [
        render_labeled_cell(
            item=item,
            image=resize_to_width(image, target_width, image_module),
            width=target_width,
            label_height=config.label_height,
            background=config.background,
            image_module=image_module,
            draw_module=draw_module,
            font_module=font_module,
        )
        for item, image in source_images
    ]
    width = target_width + config.padding * 2
    height = config.padding + sum(cell.height for cell in cells) + config.padding * len(cells)
    montage = image_module.new("RGB", (width, height), config.background)
    y = config.padding
    for cell in cells:
        montage.paste(cell, (config.padding, y))
        y += cell.height + config.padding
    return montage


def build_grid_montage(source_images, config, image_module, draw_module, font_module):
    columns = max(1, config.columns)
    cell_width = int(config.cell_width or config.target_width or max(image.width for _, image in source_images))
    cells = [
        render_labeled_cell(
            item=item,
            image=resize_to_width(image, cell_width, image_module),
            width=cell_width,
            label_height=config.label_height,
            background=config.background,
            image_module=image_module,
            draw_module=draw_module,
            font_module=font_module,
        )
        for item, image in source_images
    ]
    rows = [cells[index : index + columns] for index in range(0, len(cells), columns)]
    row_heights = [max(cell.height for cell in row) for row in rows]
    width = config.padding + columns * cell_width + config.padding * columns
    height = config.padding + sum(row_heights) + config.padding * len(rows)
    montage = image_module.new("RGB", (width, height), config.background)
    y = config.padding
    for row, row_height in zip(rows, row_heights):
        x = config.padding
        for cell in row:
            montage.paste(cell, (x, y))
            x += cell_width + config.padding
        y += row_height + config.padding
    return montage


def render_labeled_cell(
    item: dict,
    image,
    width: int,
    label_height: int,
    background: str,
    image_module,
    draw_module,
    font_module,
):
    height = image.height + label_height
    cell = image_module.new("RGB", (width, height), background)
    draw = draw_module.Draw(cell)
    label = evidence_label(item)
    font = load_default_font(font_module)
    draw.text((12, max(4, (label_height - 12) // 2)), label, fill="black", font=font)
    cell.paste(image, (0, label_height))
    return cell


def evidence_label(item: dict) -> str:
    parts = [str(item.get("evidence_id") or "")]
    page = item.get("page")
    if page:
        parts.append(f"page {page}")
    title = str(item.get("title") or item.get("id") or "").strip()
    if title:
        parts.append(title[:90])
    return " | ".join(part for part in parts if part)


def resize_to_width(image, width: int, image_module):
    if image.width == width:
        return image
    height = max(1, round(image.height * (width / image.width)))
    return image.resize((width, height), image_module.Resampling.LANCZOS)


def fit_pixel_budget(image, max_pixels: int, min_pixels: int, image_module):
    pixels = image.width * image.height
    scale = 1.0
    if pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
    elif pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
    if abs(scale - 1.0) < 0.01:
        return image
    width = max(1, math.floor(image.width * scale))
    height = max(1, math.floor(image.height * scale))
    resized = image.resize((width, height), image_module.Resampling.LANCZOS)
    while resized.width * resized.height > max_pixels:
        width = max(1, resized.width - 1)
        height = max(1, math.floor(max_pixels / width))
        resized = resized.resize((width, height), image_module.Resampling.LANCZOS)
    return resized


def save_image(image, output_path: Path, config: MontageConfig) -> None:
    fmt = normalized_image_format(config.output_format)
    if fmt == "JPEG":
        image.save(output_path, format=fmt, quality=config.jpeg_quality, optimize=True)
    else:
        image.save(output_path, format=fmt)


def normalized_image_format(output_format: str) -> str:
    output_format = output_format.lower().lstrip(".")
    if output_format in {"jpg", "jpeg"}:
        return "JPEG"
    if output_format == "png":
        return "PNG"
    raise ValueError(f"Unsupported output_format: {output_format}")


def montage_filename(document_id: str, kind: str, query_key: str, output_format: str) -> str:
    extension = "jpg" if output_format.lower() in {"jpg", "jpeg"} else output_format.lower()
    return f"{safe_filename(document_id)}_{kind}_montage_{query_key}.{extension}"


def stable_query_key(query: str) -> str:
    text = str(query or "").strip()
    if not text:
        return "noquery"
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def safe_filename(value: str) -> str:
    safe = []
    for char in str(value or "document"):
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "document"


def resolve_crop_path(item: dict, project_root: Path) -> Optional[Path]:
    value = item.get("crop_path") or item.get("asset_path") or item.get("image_path")
    if value is None or str(value).strip() == "":
        return None
    return resolve_project_path(value, project_root)


def resolve_project_path(value: object, project_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def project_relative_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def load_default_font(font_module):
    try:
        return font_module.load_default(size=20)
    except TypeError:
        return font_module.load_default()


def write_evidence_package(package: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(package, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def optional_int(value: object) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    return to_int(value)


def to_int(value: object) -> int:
    return int(value)
