import argparse
from pathlib import Path


from finsightrag.paths import default_project_root


PROJECT_ROOT = default_project_root()

from finsightrag.rag_config import RagConfig
from finsightrag.text_chunking import (
    DEFAULT_BREAKPOINT_PERCENTILE,
    DEFAULT_CHUNK_OUTPUT_SUFFIX,
    DEFAULT_DUPLICATE_OVERLAP_THRESHOLD,
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_HIGH_OVERLAP_THRESHOLD,
    DEFAULT_INITIAL_CHUNK_OUTPUT_SUFFIX,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_SENTENCE_GAP,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MERGE_SAFETY_RATIO,
    DEFAULT_MERGE_SIMILARITY_THRESHOLD,
    DEFAULT_OUTPUT_SUFFIX,
    DEFAULT_OVERLAP,
    DEFAULT_SMALL_CHUNK_REPAIR,
    DEFAULT_SMALL_CHUNK_SENTENCE_COUNT,
    DEFAULT_SHORT_PREVIOUS_MAX_TOKENS,
    DEFAULT_SHORT_PREVIOUS_MERGE,
    DEFAULT_WINDOW,
    SemanticChunkingConfig,
    build_sentence_items_from_page_json_dir,
    split_text_md_to_sentences,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split text-only Markdown files into sentences and semantic chunks."
        )
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Optional runtime config.yaml path. Defaults to the project config.yaml.",
    )
    parser.add_argument(
        "--text-md-file",
        type=Path,
        action="append",
        dest="text_md_files",
        help="Specific *_text.md file. Can be provided more than once.",
    )
    parser.add_argument(
        "--text-md-dir",
        type=Path,
        default=None,
        help="Directory containing *_text.md files.",
    )
    parser.add_argument(
        "--page-json-dir",
        type=Path,
        default=None,
        help="Optional page_*.json directory for a single --text-md-file; writes page metadata into sentences/chunks.",
    )
    parser.add_argument(
        "--sentence-split-file",
        type=Path,
        default=None,
        help="Optional existing sentence split JSON to reuse sentence/page metadata for a single --text-md-file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Compatibility alias for --sentence-output-dir.",
    )
    parser.add_argument(
        "--sentence-output-dir",
        type=Path,
        default=None,
        help="Directory for *_sentence_split.json files. Defaults to each input file directory.",
    )
    parser.add_argument(
        "--chunk-output-dir",
        type=Path,
        default=None,
        help="Directory for *_initial_chunks.json and *_chunk_merge.json files. Defaults to sentence output dir.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="NLTK punkt language name. Defaults to english.",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Output filename suffix. Defaults to sentence_split.",
    )
    parser.add_argument(
        "--chunk-output-suffix",
        default=None,
        help="Chunk output filename suffix. Defaults to chunk_merge.",
    )
    parser.add_argument(
        "--initial-chunk-output-suffix",
        default=None,
        help="Initial chunk output filename suffix. Defaults to initial_chunks.",
    )
    parser.add_argument(
        "--sentences-only",
        action="store_true",
        help="Only write sentence split JSON; skip semantic chunking.",
    )
    parser.add_argument("--window", type=int, default=None, help="Sentence window size.")
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Number of overlapping sentences between adjacent windows.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Window step. Defaults to window - overlap.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--breakpoint-percentile",
        type=float,
        default=None,
        help="Percentile threshold used for within-block distance breakpoints.",
    )
    parser.add_argument(
        "--merge-similarity-threshold",
        type=float,
        default=None,
        help="Minimum chunk-pair cosine similarity for iterative merging.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum token count after merging two chunks.",
    )
    parser.add_argument(
        "--max-embedding-tokens",
        type=int,
        default=None,
        help="Maximum token count for embedding-level merged chunks. Defaults to --max-tokens.",
    )
    parser.add_argument(
        "--max-sentence-gap",
        type=int,
        default=None,
        help="Maximum allowed sentence_id gap for position-protected merging.",
    )
    parser.add_argument(
        "--merge-safety-ratio",
        type=float,
        default=None,
        help="Stop merging once merge_count reaches initial_chunk_count * this ratio.",
    )
    parser.add_argument(
        "--high-overlap-threshold",
        type=float,
        default=None,
        help="Merge chunk pairs whose sentence overlap ratio reaches this threshold.",
    )
    parser.add_argument(
        "--disable-small-chunk-repair",
        action="store_true",
        help="Disable local repair for very short chunks.",
    )
    parser.add_argument(
        "--disable-short-previous-merge",
        action="store_true",
        help="Disable the pre-cleanup rule that merges short answer-like chunks into the previous chunk.",
    )
    parser.add_argument(
        "--short-previous-max-tokens",
        type=int,
        default=None,
        help="Maximum token count for short answer-like chunks that should prefer the previous chunk.",
    )
    parser.add_argument(
        "--small-chunk-sentence-count",
        type=int,
        default=None,
        help="Chunks with this many sentence ids or fewer are eligible for local repair.",
    )
    parser.add_argument(
        "--duplicate-overlap-threshold",
        type=float,
        default=None,
        help="Final validation threshold for duplicate sentence overlap pairs.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=None,
        help="Batch size for SentenceTransformer encoding.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional SentenceTransformer device, for example cpu or cuda.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if the target JSON already exists.",
    )
    parser.add_argument(
        "--write-initial-chunks",
        action="store_true",
        help="Also write a debug *_initial_chunks.json file.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config_values = load_text_chunking_config(args.config_path)
    language = get_setting(args, config_values, "language", DEFAULT_LANGUAGE)
    output_suffix = get_setting(args, config_values, "output_suffix", DEFAULT_OUTPUT_SUFFIX)
    chunk_output_suffix = get_setting(
        args,
        config_values,
        "chunk_output_suffix",
        DEFAULT_CHUNK_OUTPUT_SUFFIX,
    )
    initial_chunk_output_suffix = get_setting(
        args,
        config_values,
        "initial_chunk_output_suffix",
        DEFAULT_INITIAL_CHUNK_OUTPUT_SUFFIX,
    )
    semantic_config = build_semantic_config(args, config_values)
    sentence_output_dir = args.sentence_output_dir or args.output_dir
    chunk_output_dir = args.chunk_output_dir
    files = collect_text_md_files(args)
    if not files:
        print("No text-only Markdown files found.")
        return 1

    results = []
    for text_md_file in files:
        sentence_items = load_sentence_items_for_file(args, text_md_file, language)
        results.append(
            split_text_md_to_sentences(
                text_md_path=text_md_file,
                output_dir=sentence_output_dir,
                chunk_output_dir=chunk_output_dir,
                language=language,
                output_suffix=output_suffix,
                initial_chunk_output_suffix=initial_chunk_output_suffix,
                chunk_output_suffix=chunk_output_suffix,
                semantic_config=semantic_config,
                semantic_chunking=not args.sentences_only,
                write_initial_chunks=args.write_initial_chunks,
                sentence_items=sentence_items,
                overwrite=not args.no_overwrite,
            )
        )

    for result in results:
        print(
            f"{result.source_path.name}: wrote {result.sentence_count} sentences "
            f"to {result.sentence_output_path}"
        )
        if not args.sentences_only:
            if result.initial_chunk_output_path:
                print(
                    f"  wrote {result.initial_chunk_count} initial chunks to "
                    f"{result.initial_chunk_output_path}"
                )
            print(
                f"  wrote {result.final_chunk_count} merged chunks to "
                f"{result.chunk_output_path}"
            )
            print(
                f"  initial chunks: {result.initial_chunk_count}, merges: {result.merge_count}"
            )
            print(
                f"  validation: passed={result.validation_passed}, "
                f"contained_pairs={result.containment_issue_count}, "
                f"duplicate_overlap_pairs={result.duplicate_overlap_issue_count}"
            )
    print(f"Done. Total files: {len(results)}")
    return 0


def collect_text_md_files(args) -> list[Path]:
    if args.text_md_files and args.text_md_dir:
        raise SystemExit("Use either --text-md-file or --text-md-dir, not both.")
    if args.text_md_files:
        return [path.resolve() for path in args.text_md_files]
    if args.text_md_dir is None:
        raise SystemExit("Please provide --text-md-file or --text-md-dir.")
    return sorted(path.resolve() for path in args.text_md_dir.glob("*_text.md"))


def load_sentence_items_for_file(args, text_md_file: Path, language: str):
    if args.page_json_dir and args.sentence_split_file:
        raise SystemExit("Use either --page-json-dir or --sentence-split-file, not both.")
    if args.page_json_dir and (not args.text_md_files or len(args.text_md_files) != 1):
        raise SystemExit("--page-json-dir is only supported with a single --text-md-file.")
    if args.sentence_split_file and (not args.text_md_files or len(args.text_md_files) != 1):
        raise SystemExit("--sentence-split-file is only supported with a single --text-md-file.")
    if args.page_json_dir:
        return build_sentence_items_from_page_json_dir(args.page_json_dir, language=language)
    if args.sentence_split_file:
        import json

        data = json.loads(args.sentence_split_file.read_text(encoding="utf-8"))
        return data.get("sentences", [])
    return None


def load_text_chunking_config(config_path: Path | None) -> dict:
    rag_config = RagConfig.load(str(config_path) if config_path else None)
    data = rag_config.values or {}
    if not isinstance(data, dict):
        return {}
    section = data.get("text_chunking", {})
    return section if isinstance(section, dict) else {}


def get_setting(args, config_values: dict, name: str, default):
    cli_value = getattr(args, name, None)
    if cli_value is not None:
        return cli_value
    flat_name = f"text_chunking_{name}"
    if name in config_values:
        return config_values[name]
    if flat_name in config_values:
        return config_values[flat_name]
    return default


def build_semantic_config(args, config_values: dict) -> SemanticChunkingConfig:
    max_tokens = int(get_setting(args, config_values, "max_tokens", DEFAULT_MAX_TOKENS))
    max_embedding_tokens = optional_int(
        get_setting(args, config_values, "max_embedding_tokens", None)
    )
    if max_embedding_tokens is None:
        max_embedding_tokens = max_tokens
    small_chunk_repair = optional_bool(
        get_setting(
            args,
            config_values,
            "small_chunk_repair",
            DEFAULT_SMALL_CHUNK_REPAIR,
        )
    )
    if args.disable_small_chunk_repair:
        small_chunk_repair = False
    short_previous_merge = optional_bool(
        get_setting(
            args,
            config_values,
            "short_previous_merge",
            DEFAULT_SHORT_PREVIOUS_MERGE,
        )
    )
    if args.disable_short_previous_merge:
        short_previous_merge = False
    return SemanticChunkingConfig(
        window=int(get_setting(args, config_values, "window", DEFAULT_WINDOW)),
        overlap=int(get_setting(args, config_values, "overlap", DEFAULT_OVERLAP)),
        step=optional_int(get_setting(args, config_values, "step", None)),
        embedding_model=str(
            get_setting(args, config_values, "embedding_model", DEFAULT_EMBEDDING_MODEL)
        ),
        breakpoint_percentile=float(
            get_setting(
                args,
                config_values,
                "breakpoint_percentile",
                DEFAULT_BREAKPOINT_PERCENTILE,
            )
        ),
        merge_similarity_threshold=float(
            get_setting(
                args,
                config_values,
                "merge_similarity_threshold",
                DEFAULT_MERGE_SIMILARITY_THRESHOLD,
            )
        ),
        max_tokens=max_tokens,
        max_embedding_tokens=max_embedding_tokens,
        max_sentence_gap=int(
            get_setting(args, config_values, "max_sentence_gap", DEFAULT_MAX_SENTENCE_GAP)
        ),
        merge_safety_ratio=float(
            get_setting(args, config_values, "merge_safety_ratio", DEFAULT_MERGE_SAFETY_RATIO)
        ),
        embed_batch_size=int(
            get_setting(args, config_values, "embed_batch_size", DEFAULT_EMBED_BATCH_SIZE)
        ),
        high_overlap_threshold=float(
            get_setting(
                args,
                config_values,
                "high_overlap_threshold",
                DEFAULT_HIGH_OVERLAP_THRESHOLD,
            )
        ),
        small_chunk_repair=small_chunk_repair,
        small_chunk_sentence_count=int(
            get_setting(
                args,
                config_values,
                "small_chunk_sentence_count",
                DEFAULT_SMALL_CHUNK_SENTENCE_COUNT,
            )
        ),
        short_previous_merge=short_previous_merge,
        short_previous_max_tokens=int(
            get_setting(
                args,
                config_values,
                "short_previous_max_tokens",
                DEFAULT_SHORT_PREVIOUS_MAX_TOKENS,
            )
        ),
        duplicate_overlap_threshold=float(
            get_setting(
                args,
                config_values,
                "duplicate_overlap_threshold",
                DEFAULT_DUPLICATE_OVERLAP_THRESHOLD,
            )
        ),
        device=optional_str(get_setting(args, config_values, "device", None)),
    )


def optional_int(value):
    if value in (None, ""):
        return None
    return int(value)


def optional_str(value):
    if value in (None, ""):
        return None
    return str(value)


def optional_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    raise SystemExit(main())

