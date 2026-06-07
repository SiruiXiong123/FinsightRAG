import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.asset_enricher import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_OUTPUT_SUFFIX,
    DEFAULT_IMAGE_PROMPT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_SINGLE_FALLBACK_RETRIES,
    DEFAULT_TABLE_OUTPUT_SUFFIX,
    DEFAULT_TABLE_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    EnrichmentConfig,
    enrich_asset_directory,
)
from src.rag_config import RagConfig


ASSET_KINDS = ("table", "image")
PROMPT_DEFAULTS = {
    "table": DEFAULT_TABLE_PROMPT,
    "image": DEFAULT_IMAGE_PROMPT,
}
OUTPUT_SUFFIXES = {
    "table": DEFAULT_TABLE_OUTPUT_SUFFIX,
    "image": DEFAULT_IMAGE_OUTPUT_SUFFIX,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-enrich cropped table/image assets with a vision model."
    )
    parser.add_argument(
        "--asset-kind",
        choices=[*ASSET_KINDS, "all"],
        default="all",
        help="Asset type to enrich. Defaults to all.",
    )
    parser.add_argument("--config-path", type=Path, default=None, help="Optional runtime config.yaml path.")
    parser.add_argument("--pdf-name", default=None, help="Document name used in output filenames.")
    parser.add_argument(
        "--ocr-output-dir",
        type=Path,
        default=None,
        help="Directory containing <document>_tables and <document>_images folders.",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=None,
        help="Asset directory for a single --asset-kind table/image run.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Compatibility alias for --asset-dir.",
    )
    parser.add_argument("--table-dir", type=Path, default=None, help="Directory containing cropped table images.")
    parser.add_argument(
        "--image-asset-dir",
        "--figure-dir",
        dest="image_asset_dir",
        type=Path,
        default=None,
        help="Directory containing cropped image/chart assets.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for enrichment JSON outputs.")
    parser.add_argument("--prompt-path", default=None, help="Prompt path for a single asset kind run.")
    parser.add_argument("--table-prompt", default=None, help="Prompt path for table enrichment.")
    parser.add_argument("--image-prompt", default=None, help="Prompt path for image enrichment.")
    parser.add_argument("--prompt-variable", default=None, help="Prompt string variable for .py prompt files.")
    parser.add_argument("--batch-size", type=int, default=None, help="Initial batch size.")
    parser.add_argument("--single-fallback-retries", type=int, default=None, help="Single-image retry count.")
    parser.add_argument("--model", default=None, help="Vision model name.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=None, help="OpenAI-compatible API key.")
    parser.add_argument("--temperature", type=float, default=None, help="Vision model temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Vision response max tokens.")
    parser.add_argument("--timeout", type=int, default=None, help="Vision request timeout seconds.")
    parser.add_argument("--include-raw-responses", action="store_true", help="Store raw model text in output JSON.")
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if output JSON already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call the model; write deterministic dummy output.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    rag_config = RagConfig.load(str(args.config_path) if args.config_path else None)
    settings = load_enrichment_settings(rag_config)
    work_items = build_work_items(args)
    if not work_items:
        print("No asset directories selected.")
        return 1

    failed = False
    for asset_kind, asset_dir in work_items:
        config = build_enrichment_config(
            args=args,
            rag_config=rag_config,
            settings=settings,
            asset_kind=asset_kind,
            asset_dir=asset_dir,
        )
        payload = enrich_asset_directory(config)
        print(
            f"Wrote {payload['success_count']}/{payload['total_files']} "
            f"{asset_kind} enrichments to {payload['output_path']}"
        )
        if payload["failed_filenames"]:
            failed = True
            print(f"Failed filenames: {', '.join(payload['failed_filenames'])}")

    return 1 if failed else 0


def selected_asset_kinds(asset_kind: str) -> tuple[str, ...]:
    if asset_kind == "all":
        return ASSET_KINDS
    return (asset_kind,)


def build_work_items(args) -> list[tuple[str, Path]]:
    kinds = selected_asset_kinds(args.asset_kind)
    single_asset_dir = args.asset_dir or args.image_dir
    if single_asset_dir and args.asset_kind == "all":
        raise SystemExit("--asset-dir/--image-dir can only be used with --asset-kind table or image.")
    if args.image_dir and args.asset_dir:
        raise SystemExit("Use only one of --asset-dir or --image-dir.")

    items = []
    for asset_kind in kinds:
        asset_dir = resolve_asset_dir(args, asset_kind)
        if asset_dir is None:
            raise SystemExit(
                f"Cannot infer {asset_kind} asset directory. Provide --ocr-output-dir "
                f"with --pdf-name, or pass --{asset_kind if asset_kind == 'table' else 'image-asset'}-dir."
            )
        items.append((asset_kind, asset_dir.resolve()))
    return items


def resolve_asset_dir(args, asset_kind: str) -> Path | None:
    if asset_kind == "table" and args.table_dir:
        return args.table_dir
    if asset_kind == "image" and args.image_asset_dir:
        return args.image_asset_dir
    asset_dir = args.asset_dir or args.image_dir
    if asset_dir and args.asset_kind in ASSET_KINDS:
        return asset_dir
    if args.ocr_output_dir and args.pdf_name:
        suffix = "tables" if asset_kind == "table" else "images"
        return args.ocr_output_dir / f"{args.pdf_name}_{suffix}"
    return None


def build_enrichment_config(
    args,
    rag_config: RagConfig,
    settings: dict,
    asset_kind: str,
    asset_dir: Path,
) -> EnrichmentConfig:
    prompt_path = resolve_project_path(
        choose_prompt_path(args, settings, asset_kind),
        PROMPT_DEFAULTS[asset_kind],
    )
    return EnrichmentConfig(
        asset_dir=asset_dir,
        output_dir=args.output_dir,
        prompt_path=prompt_path,
        prompt_variable=get_setting(args, settings, "prompt_variable", None),
        config_path=args.config_path,
        pdf_name=args.pdf_name,
        asset_kind=asset_kind,
        output_suffix=OUTPUT_SUFFIXES[asset_kind],
        batch_size=int(get_setting(args, settings, "batch_size", DEFAULT_BATCH_SIZE)),
        single_fallback_retries=int(
            get_setting(args, settings, "single_fallback_retries", DEFAULT_SINGLE_FALLBACK_RETRIES)
        ),
        model=get_setting(args, settings, "model", None) or rag_config.vision_model,
        base_url=get_setting(args, settings, "base_url", None) or rag_config.vision_base_url,
        api_key=get_setting(args, settings, "api_key", None) or rag_config.vision_api_key,
        temperature=float(get_setting(args, settings, "temperature", DEFAULT_TEMPERATURE)),
        max_tokens=int(get_setting(args, settings, "max_tokens", DEFAULT_MAX_TOKENS)),
        timeout=int(get_setting(args, settings, "timeout", DEFAULT_TIMEOUT)),
        include_raw_responses=bool(args.include_raw_responses),
        overwrite=not args.no_overwrite,
        dry_run=bool(args.dry_run),
    )


def load_enrichment_settings(rag_config: RagConfig) -> dict:
    section = dict(rag_config.values or {}).get("enrichment", {})
    return section if isinstance(section, dict) else {}


def choose_prompt_path(args, settings: dict, asset_kind: str):
    if args.prompt_path and args.asset_kind in ASSET_KINDS:
        return args.prompt_path
    cli_value = args.table_prompt if asset_kind == "table" else args.image_prompt
    if cli_value:
        return cli_value
    asset_prompt_key = f"{asset_kind}_prompt"
    if asset_prompt_key in settings:
        return settings[asset_prompt_key]
    if "prompt_path" in settings:
        return settings["prompt_path"]
    return None


def get_setting(args, settings: dict, name: str, default):
    cli_value = getattr(args, name, None)
    if cli_value is not None:
        return cli_value
    if name in settings:
        return settings[name]
    return default


def resolve_project_path(raw_path, default_relative: str) -> Path:
    if raw_path:
        path = Path(raw_path).expanduser()
    else:
        path = PROJECT_ROOT / default_relative
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
