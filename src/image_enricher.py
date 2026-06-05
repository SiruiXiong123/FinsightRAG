try:
    from .table_enricher import (
        EnrichmentConfig,
        build_base_parser,
        build_config_from_args,
        enrich_image_directory,
    )
except ImportError:
    from table_enricher import (
        EnrichmentConfig,
        build_base_parser,
        build_config_from_args,
        enrich_image_directory,
    )


DEFAULT_IMAGE_PROMPT = "prompts/image_enrichment_prompt.py"
DEFAULT_IMAGE_OUTPUT_SUFFIX = "image_enrichment"


def enrich_images(config: EnrichmentConfig) -> dict:
    return enrich_image_directory(config)


def main(argv=None) -> int:
    parser = build_base_parser(
        description="Batch-enrich cropped image/chart images with a vision model.",
        default_prompt=DEFAULT_IMAGE_PROMPT,
    )
    args = parser.parse_args(argv)
    config = build_config_from_args(
        args=args,
        asset_kind="image",
        output_suffix=DEFAULT_IMAGE_OUTPUT_SUFFIX,
        default_prompt=DEFAULT_IMAGE_PROMPT,
    )
    payload = enrich_images(config)
    print(
        f"Wrote {payload['success_count']}/{payload['total_files']} image enrichments "
        f"to {payload['output_path']}"
    )
    if payload["failed_filenames"]:
        print(f"Failed filenames: {', '.join(payload['failed_filenames'])}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
