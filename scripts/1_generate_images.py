import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.image_assets import extract_image_assets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract image crops from PaddleOCR-VL JSON and the source PDF. "
            "Each crop includes a nearby title block when one is found."
        )
    )
    add_common_args(parser)
    parser.add_argument(
        "--image-label",
        action="append",
        dest="target_labels",
        help="Image block label to extract. Can be repeated. Defaults to image and chart.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    title_max_distance = None if args.title_max_distance < 0 else args.title_max_distance
    work_items = build_work_items(args)
    if not work_items:
        print("No PaddleOCR-VL document JSON files found.")
        return 1

    total = 0
    for json_file, pdf_file, output_root, page_json_dir in work_items:
        crops = extract_image_assets(
            json_path=json_file,
            pdf_path=pdf_file,
            output_root=output_root,
            page_json_dir=page_json_dir,
            title_labels=args.title_labels,
            title_max_distance=title_max_distance,
            min_overlap_ratio=args.min_overlap_ratio,
            padding=args.padding,
            dpi=args.dpi,
            overwrite=not args.no_overwrite,
            target_labels=args.target_labels,
        )
        total += len(crops)
        target_dir = (output_root or json_file.parent) / f"{json_file.stem}_images"
        print(f"{json_file.name}: wrote {len(crops)} image PNG files to {target_dir}")
        for crop in crops:
            print(
                f"  page {crop.page_index + 1}, image {crop.asset_index}: "
                f"{crop.title} -> {crop.output_path.name}"
            )

    print(f"Done. Total image PNG files: {total}")
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ocr-output-dir",
        type=Path,
        default=None,
        help="Directory containing top-level PaddleOCR-VL <document>.json files.",
    )
    parser.add_argument(
        "--json-file",
        type=Path,
        action="append",
        dest="json_files",
        help="Specific PaddleOCR-VL JSON file. Can be provided more than once.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Directory containing source PDFs named like <document>.pdf.",
    )
    parser.add_argument(
        "--pdf-file",
        type=Path,
        action="append",
        dest="pdf_files",
        help="Specific source PDF paired with --json-file. Use the same order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root for generated asset folders.",
    )
    parser.add_argument(
        "--page-json-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing page_*.json files for a single --json-file. "
            "Defaults to the sibling folder named like the JSON stem."
        ),
    )
    parser.add_argument(
        "--title-label",
        action="append",
        dest="title_labels",
        help=(
            "Additional block label treated as a title. Can be repeated. "
            "By default, any block_label ending with _title is treated as a title."
        ),
    )
    parser.add_argument(
        "--title-max-distance",
        type=float,
        default=160.0,
        help="Maximum vertical pixel distance between title and asset. Use -1 to disable.",
    )
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.15,
        help="Minimum horizontal overlap ratio between title and asset bboxes.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=8.0,
        help="OCR-pixel padding added around the merged title/asset bbox.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG render DPI.")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if a target PNG already exists.",
    )


def validate_args(args) -> None:
    if not args.json_files and args.ocr_output_dir is None:
        raise SystemExit("Please provide --ocr-output-dir or at least one --json-file.")
    if args.pdf_files and not args.json_files:
        raise SystemExit("--pdf-file can only be used together with --json-file.")
    if args.json_files and args.pdf_files and len(args.json_files) != len(args.pdf_files):
        raise SystemExit("--json-file and --pdf-file counts must match.")
    if args.json_files and not args.pdf_files and args.pdf_dir is None:
        raise SystemExit("Please provide --pdf-file or --pdf-dir for --json-file inputs.")
    if not args.json_files and args.pdf_dir is None:
        raise SystemExit("Please provide --pdf-dir when using --ocr-output-dir.")
    if args.page_json_dir is not None and (not args.json_files or len(args.json_files) != 1):
        raise SystemExit("--page-json-dir is only supported with a single --json-file.")


def build_work_items(args) -> list[tuple[Path, Path, Path | None, Path | None]]:
    if args.json_files:
        pdf_files = args.pdf_files or [
            resolve_pdf_for_json(json_file=json_file, pdf_dir=args.pdf_dir)
            for json_file in args.json_files
        ]
        return [
            (
                json_file.resolve(),
                pdf_file.resolve(),
                args.output_dir.resolve() if args.output_dir else None,
                args.page_json_dir.resolve() if args.page_json_dir else None,
            )
            for json_file, pdf_file in zip(args.json_files, pdf_files)
        ]

    json_files = sorted(
        path
        for path in args.ocr_output_dir.glob("*.json")
        if not path.stem.endswith("_text")
    )
    output_root = args.output_dir.resolve() if args.output_dir else args.ocr_output_dir.resolve()
    return [
        (
            json_file.resolve(),
            resolve_pdf_for_json(json_file=json_file, pdf_dir=args.pdf_dir).resolve(),
            output_root,
            None,
        )
        for json_file in json_files
    ]


def resolve_pdf_for_json(json_file: Path, pdf_dir: Path) -> Path:
    pdf_path = pdf_dir / f"{json_file.stem}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found for {json_file.name}: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    raise SystemExit(main())
