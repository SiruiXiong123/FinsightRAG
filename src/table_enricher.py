import argparse
import base64
import importlib.util
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from .parse_enrichment_output import (
        EnrichmentItem,
        enrichment_item_to_payload,
        parse_enrichment_output,
    )
    from .rag_config import RagConfig
except ImportError:
    from parse_enrichment_output import (
        EnrichmentItem,
        enrichment_item_to_payload,
        parse_enrichment_output,
    )
    from rag_config import RagConfig


DEFAULT_BATCH_SIZE = 5
DEFAULT_SINGLE_FALLBACK_RETRIES = 3
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 240
DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_TABLE_PROMPT = "prompts/table_enrichment_prompt.py"
DEFAULT_TABLE_OUTPUT_SUFFIX = "table_enrichment"


@dataclass(frozen=True)
class EnrichmentConfig:
    image_dir: Path
    output_dir: Optional[Path]
    prompt_path: Path
    prompt_variable: Optional[str]
    config_path: Optional[Path]
    pdf_name: Optional[str]
    asset_kind: str
    output_suffix: str
    batch_size: int
    single_fallback_retries: int
    model: Optional[str]
    base_url: Optional[str]
    api_key: Optional[str]
    temperature: float
    max_tokens: int
    timeout: int
    include_raw_responses: bool = False
    overwrite: bool = True
    dry_run: bool = False

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if self.single_fallback_retries <= 0:
            raise ValueError("single_fallback_retries must be greater than 0.")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0.")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than 0.")
        if not self.dry_run:
            if not self.model:
                raise ValueError("vision model is required. Set vision_model or pass --model.")
            if not self.base_url:
                raise ValueError("vision base URL is required. Set llm_binding_host or pass --base-url.")
            if not self.api_key:
                raise ValueError("vision API key is required. Set llm_binding_api_key or pass --api-key.")


@dataclass(frozen=True)
class BatchAttempt:
    phase: str
    batch_size: int
    filenames: list[str]
    parsed_filenames: list[str]
    missing_filenames: list[str]
    unexpected_filenames: list[str]
    duplicate_filenames: list[str]
    success: bool
    error: Optional[str] = None
    raw_response: Optional[str] = None


def enrich_table_images(config: EnrichmentConfig) -> dict:
    return enrich_image_directory(config)


def enrich_image_directory(config: EnrichmentConfig) -> dict:
    config.validate()
    image_paths = collect_image_paths(config.image_dir)
    prompt_template = load_prompt_template(config.prompt_path, config.prompt_variable)
    document_name = config.pdf_name or infer_document_name(config.image_dir, config.asset_kind)
    output_dir = (config.output_dir or config.image_dir.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{document_name}_{config.output_suffix}.json"
    if output_path.exists() and not config.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    filename_to_path = {path.name: path for path in image_paths}
    pending = [path.name for path in image_paths]
    results: dict[str, EnrichmentItem] = {}
    attempts: list[BatchAttempt] = []

    batch_schedule = make_batch_schedule(config.batch_size)
    for round_index, batch_size in enumerate(batch_schedule, start=1):
        if not pending:
            break
        phase = f"batch_round_{round_index}"
        for filenames in chunked(pending, batch_size):
            batch_result = run_batch(
                filenames=filenames,
                filename_to_path=filename_to_path,
                prompt_template=prompt_template,
                config=config,
                phase=phase,
                batch_size=batch_size,
            )
            attempts.append(batch_result["attempt"])
            for item in batch_result["items"]:
                results[item.filename] = item
        pending = [
            filename
            for filename in filename_to_path
            if filename not in results
        ]

    if pending:
        for filename in list(pending):
            for retry_index in range(1, config.single_fallback_retries + 1):
                batch_result = run_batch(
                    filenames=[filename],
                    filename_to_path=filename_to_path,
                    prompt_template=prompt_template,
                    config=config,
                    phase=f"single_fallback_{retry_index}",
                    batch_size=1,
                )
                attempts.append(batch_result["attempt"])
                for item in batch_result["items"]:
                    results[item.filename] = item
                if filename in results:
                    break
        pending = [
            filename
            for filename in filename_to_path
            if filename not in results
        ]

    payload = build_output_payload(
        document_name=document_name,
        config=config,
        image_paths=image_paths,
        output_path=output_path,
        attempts=attempts,
        results=results,
        failed_filenames=pending,
    )
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def run_batch(
    filenames: list[str],
    filename_to_path: dict[str, Path],
    prompt_template: str,
    config: EnrichmentConfig,
    phase: str,
    batch_size: int,
) -> dict:
    try:
        prompt = build_batch_prompt(prompt_template, filenames)
        if config.dry_run:
            raw_response = build_dry_run_response(filenames)
        else:
            raw_response = call_vision_model(
                prompt=prompt,
                image_paths=[filename_to_path[filename] for filename in filenames],
                config=config,
            )
        parse_result = parse_enrichment_output(raw_response, expected_filenames=filenames)
        attempt = BatchAttempt(
            phase=phase,
            batch_size=batch_size,
            filenames=filenames,
            parsed_filenames=[item.filename for item in parse_result.items],
            missing_filenames=parse_result.missing_filenames,
            unexpected_filenames=parse_result.unexpected_filenames,
            duplicate_filenames=parse_result.duplicate_filenames,
            success=not parse_result.missing_filenames,
            raw_response=raw_response if config.include_raw_responses else None,
        )
        return {
            "items": parse_result.items,
            "attempt": attempt,
        }
    except Exception as exc:
        attempt = BatchAttempt(
            phase=phase,
            batch_size=batch_size,
            filenames=filenames,
            parsed_filenames=[],
            missing_filenames=filenames,
            unexpected_filenames=[],
            duplicate_filenames=[],
            success=False,
            error=str(exc),
        )
        return {
            "items": [],
            "attempt": attempt,
        }


def call_vision_model(prompt: str, image_paths: list[Path], config: EnrichmentConfig) -> str:
    content = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        content.append({"type": "text", "text": f"Filename: {image_path.name}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_data_url(image_path),
                },
            }
        )

    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    request = urllib.request.Request(
        url=chat_completions_url(config.base_url or ""),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Vision API HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Vision API request failed: {exc}") from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected vision API response: {data}") from exc


def build_output_payload(
    document_name: str,
    config: EnrichmentConfig,
    image_paths: list[Path],
    output_path: Path,
    attempts: list[BatchAttempt],
    results: dict[str, EnrichmentItem],
    failed_filenames: list[str],
) -> dict:
    ordered_results = [
        enrichment_item_to_payload(results[path.name])
        for path in image_paths
        if path.name in results
    ]
    return {
        "document_name": document_name,
        "asset_kind": config.asset_kind,
        "source_image_dir": str(config.image_dir.resolve()),
        "output_path": str(output_path.resolve()),
        "prompt_path": str(config.prompt_path.resolve()),
        "model": config.model,
        "batch_size": config.batch_size,
        "batch_schedule": make_batch_schedule(config.batch_size),
        "single_fallback_retries": config.single_fallback_retries,
        "total_files": len(image_paths),
        "success_count": len(ordered_results),
        "failed_count": len(failed_filenames),
        "failed_filenames": failed_filenames,
        "results": ordered_results,
        "attempts": [asdict(attempt) for attempt in attempts],
        "created_at_unix": int(time.time()),
    }


def build_batch_prompt(prompt_template: str, filenames: list[str]) -> str:
    filenames_block = "\n".join(f"- {filename}" for filename in filenames)
    return prompt_template.format(filenames=filenames_block)


def build_dry_run_response(filenames: list[str]) -> str:
    return "\n\n".join(
        "\n".join(
            [
                f"[F] {filename}",
                "[T] DRY RUN",
                "[M] Dry run metadata.",
                "[C] Dry run content.",
            ]
        )
        for filename in filenames
    )


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def chat_completions_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    if clean.endswith("/chat/completions"):
        return clean
    return f"{clean}/chat/completions"


def collect_image_paths(image_dir: Path) -> list[Path]:
    image_dir = Path(image_dir).resolve()
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS
    )


def make_batch_schedule(initial_batch_size: int) -> list[int]:
    second = max(1, initial_batch_size // 2)
    third = max(1, second // 2)
    schedule = []
    for value in (initial_batch_size, second, third):
        if value not in schedule:
            schedule.append(value)
    return schedule


def chunked(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def infer_document_name(image_dir: Path, asset_kind: str) -> str:
    name = image_dir.name
    for suffix in (
        f"_{asset_kind}s",
        f"_{asset_kind}",
        "_tables",
        "_table",
        "_images",
        "_image",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_prompt_template(prompt_path: Path, variable_name: Optional[str]) -> str:
    prompt_path = Path(prompt_path).resolve()
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    if prompt_path.suffix.lower() != ".py":
        return prompt_path.read_text(encoding="utf-8")

    module_name = f"_prompt_{prompt_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, prompt_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import prompt file: {prompt_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if variable_name:
        value = getattr(module, variable_name, None)
        if isinstance(value, str):
            return value
        raise ValueError(f"Prompt variable is not a string: {variable_name}")

    prompt_values = [
        value
        for name, value in vars(module).items()
        if isinstance(value, str) and "prompt" in name.lower()
    ]
    if len(prompt_values) == 1:
        return prompt_values[0]
    raise ValueError(
        f"Prompt file must contain one string variable with 'prompt' in its name, "
        f"or pass --prompt-variable: {prompt_path}"
    )


def resolve_project_path(raw_path: Optional[str], default_relative: str) -> Path:
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / default_relative).resolve()


def load_enrichment_settings(config_path: Optional[Path]) -> dict:
    rag_config = RagConfig.load(str(config_path) if config_path else None)
    values = dict(rag_config.values or {})
    section = values.get("enrichment", {})
    return section if isinstance(section, dict) else {}


def get_setting(args, settings: dict, name: str, default):
    cli_value = getattr(args, name, None)
    if cli_value is not None:
        return cli_value
    if name in settings:
        return settings[name]
    return default


def build_base_parser(description: str, default_prompt: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config-path", type=Path, default=None, help="Optional YAML config path.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing cropped images.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for enrichment JSON.")
    parser.add_argument("--pdf-name", default=None, help="Document name used in the output filename.")
    parser.add_argument("--prompt-path", default=None, help=f"Prompt path. Defaults to {default_prompt}.")
    parser.add_argument("--prompt-variable", default=None, help="Prompt string variable for .py prompt files.")
    parser.add_argument("--batch-size", type=int, default=None, help="Initial batch size. Defaults to config or 5.")
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


def build_config_from_args(
    args,
    asset_kind: str,
    output_suffix: str,
    default_prompt: str,
) -> EnrichmentConfig:
    rag_config = RagConfig.load(str(args.config_path) if args.config_path else None)
    settings = load_enrichment_settings(args.config_path)
    prompt_path = resolve_project_path(
        choose_prompt_path(args, settings, asset_kind),
        default_prompt,
    )
    return EnrichmentConfig(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        prompt_path=prompt_path,
        prompt_variable=get_setting(args, settings, "prompt_variable", None),
        config_path=args.config_path,
        pdf_name=args.pdf_name,
        asset_kind=asset_kind,
        output_suffix=output_suffix,
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


def choose_prompt_path(args, settings: dict, asset_kind: str):
    if args.prompt_path:
        return args.prompt_path
    asset_prompt_key = f"{asset_kind}_prompt"
    if asset_prompt_key in settings:
        return settings[asset_prompt_key]
    if "prompt_path" in settings:
        return settings["prompt_path"]
    return None


def main(argv=None) -> int:
    parser = build_base_parser(
        description="Batch-enrich cropped table images with a vision model.",
        default_prompt=DEFAULT_TABLE_PROMPT,
    )
    args = parser.parse_args(argv)
    config = build_config_from_args(
        args=args,
        asset_kind="table",
        output_suffix=DEFAULT_TABLE_OUTPUT_SUFFIX,
        default_prompt=DEFAULT_TABLE_PROMPT,
    )
    payload = enrich_table_images(config)
    print(
        f"Wrote {payload['success_count']}/{payload['total_files']} table enrichments "
        f"to {payload['output_path']}"
    )
    if payload["failed_filenames"]:
        print(f"Failed filenames: {', '.join(payload['failed_filenames'])}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
