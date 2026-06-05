import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_md_filter import generate_text_markdown, generate_text_markdown_for_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate <document>_text.md from PaddleOCR-VL JSON/Markdown outputs, "
            "excluding non-text blocks such as tables, charts, images, and seals."
        )
    )
    parser.add_argument(
        "--ocr-output-dir",
        type=Path,
        default=None,
        help="Directory containing PaddleOCR-VL <stem>.json and <stem>.md outputs.",
    )
    parser.add_argument(
        "--json-file",
        type=Path,
        action="append",
        dest="json_files",
        help="Process a specific JSON file. Can be provided more than once.",
    )
    parser.add_argument(
        "--md-file",
        type=Path,
        action="append",
        dest="md_files",
        help=(
            "Markdown file paired with --json-file. Use the same order when "
            "passing multiple files. Defaults to changing the JSON suffix to .md."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated *_text.md files. Defaults to each JSON file directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan JSON/Markdown pairs recursively instead of only the top-level output dir.",
    )
    parser.add_argument(
        "--allow-missing-md",
        action="store_true",
        help="Allow generating from JSON even when the sibling Markdown file is missing.",
    )
    parser.add_argument(
        "--keep-label",
        action="append",
        dest="keep_labels",
        help="Only keep this block label. Can be provided more than once.",
    )
    parser.add_argument(
        "--drop-label",
        action="append",
        dest="drop_labels",
        help=(
            "Drop this block label. Can be provided more than once. "
            "Defaults to table/chart/image/seal/formula."
        ),
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Do not convert doc_title/paragraph_title blocks to Markdown headings.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if the target *_text.md file already exists.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not args.json_files and args.ocr_output_dir is None:
        raise SystemExit("Please provide --ocr-output-dir or at least one --json-file.")
    if args.md_files and not args.json_files:
        raise SystemExit("--md-file can only be used together with --json-file.")
    if args.json_files and args.md_files and len(args.json_files) != len(args.md_files):
        raise SystemExit("--json-file and --md-file counts must match.")

    if args.json_files:
        results = []
        md_files = args.md_files or [None] * len(args.json_files)
        for json_file, md_file in zip(args.json_files, md_files):
            if md_file is None:
                default_md = json_file.with_suffix(".md")
                md_file = default_md if default_md.exists() else None
                if md_file is None and not args.allow_missing_md:
                    raise SystemExit(f"Markdown file not found for {json_file}: {default_md}")
            results.append(
                generate_text_markdown(
                    json_path=json_file,
                    md_path=md_file,
                    output_dir=args.output_dir,
                    keep_labels=args.keep_labels,
                    drop_labels=args.drop_labels,
                    format_titles=not args.plain,
                    overwrite=not args.no_overwrite,
                )
            )
    else:
        results = generate_text_markdown_for_dir(
            ocr_output_dir=args.ocr_output_dir,
            output_dir=args.output_dir,
            recursive=args.recursive,
            require_md=not args.allow_missing_md,
            keep_labels=args.keep_labels,
            drop_labels=args.drop_labels,
            format_titles=not args.plain,
            overwrite=not args.no_overwrite,
        )

    if not results:
        print("No PaddleOCR-VL JSON/Markdown pairs found.")
        return 1

    for result in results:
        kept = ", ".join(f"{label}:{count}" for label, count in result.kept_labels.items())
        dropped = ", ".join(
            f"{label}:{count}" for label, count in result.dropped_labels.items()
        )
        print(f"Wrote {result.output_path}")
        print(f"  kept blocks: {result.kept_blocks} ({kept})")
        print(f"  dropped blocks: {result.dropped_blocks} ({dropped})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
