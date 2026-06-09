import argparse
import json
from pathlib import Path


from finsightrag.paths import default_project_root


PROJECT_ROOT = default_project_root()

from finsightrag.generator_vlm import build_config_from_rag_config, generate_from_evidence_package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a final VLM answer from an evidence package."
    )
    parser.add_argument("--config-path", type=Path, default=None, help="Optional runtime config.yaml path.")
    parser.add_argument("--evidence-package", type=Path, required=True, help="Evidence package JSON path.")
    parser.add_argument("--output-path", type=Path, default=None, help="Optional answer JSON output path.")
    parser.add_argument("--prompt-path", type=Path, default=None, help="Prompt file path.")
    parser.add_argument("--prompt-variable", default=None, help="Prompt string variable for .py prompt files.")
    parser.add_argument("--model", default=None, help="Vision model name.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=None, help="OpenAI-compatible API key.")
    parser.add_argument("--temperature", type=float, default=None, help="Vision model temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Vision response max tokens.")
    parser.add_argument("--timeout", type=int, default=None, help="Vision request timeout seconds.")
    parser.add_argument("--max-context-chars", type=int, default=None, help="Max chars per evidence item.")
    parser.add_argument("--include-raw-response", action="store_true", help="Store raw model text and prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Render prompt but do not call the model.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = build_config_from_rag_config(
        config_path=args.config_path,
        evidence_package_path=args.evidence_package,
        output_path=args.output_path,
        prompt_path=args.prompt_path,
        prompt_variable=args.prompt_variable,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_context_chars=args.max_context_chars,
        include_raw_response=args.include_raw_response,
        dry_run=args.dry_run,
    )
    payload = generate_from_evidence_package(config)
    if args.output_path:
        print(f"Wrote answer payload: {Path(args.output_path).resolve()}")
    print(json.dumps({"answer": payload["answer"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

