import base64
import importlib.util
import json
import mimetypes
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

try:
    from .parse_enrichment_output import (
        EnrichmentItem,
        enrichment_item_to_payload,
        parse_enrichment_output,
    )
except ImportError:
    from parse_enrichment_output import (
        EnrichmentItem,
        enrichment_item_to_payload,
        parse_enrichment_output,
    )


DEFAULT_BATCH_SIZE = 5
DEFAULT_SINGLE_FALLBACK_RETRIES = 3
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 240
DEFAULT_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_TABLE_PROMPT = "prompts/table_enrichment_prompt.py"
DEFAULT_IMAGE_PROMPT = "prompts/image_enrichment_prompt.py"
DEFAULT_TABLE_OUTPUT_SUFFIX = "table_enrichment"
DEFAULT_IMAGE_OUTPUT_SUFFIX = "image_enrichment"


@dataclass(frozen=True)
class EnrichmentConfig:
    asset_dir: Path
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
                raise ValueError("vision base URL is required. Set vision_binding_host or pass --base-url.")
            if not self.api_key:
                raise ValueError("vision API key is required. Set vision_binding_api_key or pass --api-key.")


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


def enrich_asset_directory(config: EnrichmentConfig) -> dict:
    config.validate()
    image_paths = collect_image_paths(config.asset_dir)
    prompt_template = load_prompt_template(config.prompt_path, config.prompt_variable)
    document_name = config.pdf_name or infer_document_name(config.asset_dir, config.asset_kind)
    output_dir = (config.output_dir or config.asset_dir.parent).resolve()
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
        "source_asset_dir": str(config.asset_dir.resolve()),
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


def collect_image_paths(asset_dir: Path) -> list[Path]:
    asset_dir = Path(asset_dir).resolve()
    if not asset_dir.exists() or not asset_dir.is_dir():
        raise FileNotFoundError(f"Asset directory not found: {asset_dir}")
    return sorted(
        path
        for path in asset_dir.iterdir()
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


def infer_document_name(asset_dir: Path, asset_kind: str) -> str:
    name = asset_dir.name
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
