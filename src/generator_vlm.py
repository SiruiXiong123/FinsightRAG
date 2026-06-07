import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

try:
    from .asset_enricher import chat_completions_url, image_to_data_url, load_prompt_template
    from .rag_config import RagConfig
except ImportError:
    from asset_enricher import chat_completions_url, image_to_data_url, load_prompt_template
    from rag_config import RagConfig


DEFAULT_PROMPT = "prompts/search_prompt.py"
DEFAULT_PROMPT_VARIABLE = "VLM_QA_PROMPT"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512
DEFAULT_TIMEOUT = 240
DEFAULT_CONTEXT_CHARS = 1800


@dataclass(frozen=True)
class VlmGeneratorConfig:
    project_root: Path
    evidence_package_path: Path
    output_path: Optional[Path]
    prompt_path: Path
    prompt_variable: Optional[str]
    model: Optional[str]
    base_url: Optional[str]
    api_key: Optional[str]
    temperature: float
    max_tokens: int
    timeout: int
    max_context_chars: int
    include_raw_response: bool = False
    dry_run: bool = False

    def validate(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0.")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than 0.")
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than 0.")
        if not self.evidence_package_path.exists():
            raise FileNotFoundError(f"Evidence package not found: {self.evidence_package_path}")
        if not self.dry_run:
            if not self.model:
                raise ValueError("vision model is required. Set vision_model or pass --model.")
            if not self.base_url:
                raise ValueError("vision base URL is required. Set vision_binding_host or pass --base-url.")
            if not self.api_key:
                raise ValueError("vision API key is required. Set vision_binding_api_key or pass --api-key.")


def generate_from_evidence_package(config: VlmGeneratorConfig) -> dict:
    config.validate()
    package = load_evidence_package(config.evidence_package_path)
    prompt_template = load_prompt_template(config.prompt_path, config.prompt_variable)
    prompt = build_vlm_prompt(
        prompt_template=prompt_template,
        package=package,
        max_context_chars=config.max_context_chars,
    )
    image_paths = resolve_vlm_image_paths(package, config.project_root)

    if config.dry_run:
        raw_response = "DRY RUN: prompt rendered; model was not called."
        answer = raw_response
    else:
        raw_response = call_vlm_answer_model(
            prompt=prompt,
            image_paths=image_paths,
            config=config,
        )
        answer = normalize_answer(raw_response)

    payload = build_answer_payload(
        answer=answer,
        raw_response=raw_response,
        prompt=prompt,
        image_paths=image_paths,
        package=package,
        config=config,
    )
    if config.output_path:
        write_answer_payload(payload, config.output_path)
    return payload


def build_config_from_rag_config(
    evidence_package_path: Path | str,
    config_path: Optional[Path | str] = None,
    project_root: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    prompt_path: Optional[Path | str] = None,
    prompt_variable: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    max_context_chars: Optional[int] = None,
    include_raw_response: bool = False,
    dry_run: bool = False,
) -> VlmGeneratorConfig:
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    rag_config = RagConfig.load(str(config_path) if config_path else None)
    settings = load_generation_settings(rag_config)
    selected_prompt_path = resolve_project_path(
        prompt_path or settings.get("prompt") or settings.get("prompt_path") or DEFAULT_PROMPT,
        root,
    )
    return VlmGeneratorConfig(
        project_root=root,
        evidence_package_path=resolve_project_path(evidence_package_path, root),
        output_path=resolve_project_path(output_path, root) if output_path else None,
        prompt_path=selected_prompt_path,
        prompt_variable=prompt_variable
        if prompt_variable is not None
        else settings.get("prompt_variable", DEFAULT_PROMPT_VARIABLE),
        model=model or settings.get("model") or rag_config.vision_model,
        base_url=base_url or settings.get("base_url") or rag_config.vision_base_url,
        api_key=api_key or settings.get("api_key") or rag_config.vision_api_key,
        temperature=float(value_or_default(temperature, settings.get("temperature"), DEFAULT_TEMPERATURE)),
        max_tokens=int(value_or_default(max_tokens, settings.get("max_tokens"), DEFAULT_MAX_TOKENS)),
        timeout=int(value_or_default(timeout, settings.get("timeout"), DEFAULT_TIMEOUT)),
        max_context_chars=int(
            value_or_default(max_context_chars, settings.get("max_context_chars"), DEFAULT_CONTEXT_CHARS)
        ),
        include_raw_response=include_raw_response,
        dry_run=dry_run,
    )


def load_generation_settings(rag_config: RagConfig) -> dict:
    section = dict(rag_config.values or {}).get("generation", {})
    return section if isinstance(section, dict) else {}


def value_or_default(primary, secondary, default):
    if primary is not None:
        return primary
    if secondary is not None:
        return secondary
    return default


def load_evidence_package(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Evidence package must be a JSON object: {path}")
    return data


def build_vlm_prompt(prompt_template: str, package: dict, max_context_chars: int) -> str:
    contexts = package.get("contexts_by_modality") or {}
    return prompt_template.format(
        query=str(package.get("query") or ""),
        text_evidence=format_contexts(contexts.get("text") or [], "E", max_context_chars),
        table_evidence=format_contexts(contexts.get("table") or [], "T", max_context_chars),
        image_evidence=format_contexts(contexts.get("image") or [], "I", max_context_chars),
    )


def format_contexts(contexts: list[dict], default_prefix: str, max_chars: int) -> str:
    if not contexts:
        return "None."
    return "\n\n".join(
        format_context(context, index, default_prefix, max_chars)
        for index, context in enumerate(contexts, start=1)
    )


def format_context(context: dict, index: int, default_prefix: str, max_chars: int) -> str:
    evidence_id = str(context.get("evidence_id") or f"{default_prefix}{index}")
    lines = [f"[{evidence_id}]"]
    location = format_location(context)
    if location:
        lines.append(f"Location: {location}")
    score = context.get("score")
    if score is not None:
        lines.append(f"Score: {score}")
    title = compact_text(context.get("title"), 240)
    if title:
        lines.append(f"Title: {title}")
    summary = compact_text(context.get("summary"), max_chars)
    content = compact_text(context.get("content"), max_chars)
    if summary and summary != content:
        lines.append(f"Summary: {summary}")
    if content:
        lines.append(f"Content: {content}")
    if len(lines) == 1:
        lines.append("Content: None.")
    return "\n".join(lines)


def format_location(context: dict) -> str:
    parts = []
    page = context.get("page")
    if page:
        parts.append(f"page {page}")
    page_span = context.get("page_span")
    if page_span and page_span != page:
        parts.append(f"page_span {page_span}")
    bbox = context.get("bbox")
    if bbox:
        parts.append(f"bbox {bbox}")
    return ", ".join(parts)


def compact_text(value: object, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def resolve_vlm_image_paths(package: dict, project_root: Path) -> list[Path]:
    raw_paths = ((package.get("vlm_inputs") or {}).get("image_paths") or [])
    paths = []
    for raw_path in raw_paths:
        path = resolve_project_path(raw_path, project_root)
        if path.exists():
            paths.append(path)
    return paths


def call_vlm_answer_model(prompt: str, image_paths: list[Path], config: VlmGeneratorConfig) -> str:
    content = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        content.append({"type": "text", "text": f"Visual input: {image_path.name}"})
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


def normalize_answer(raw_response: str) -> str:
    answer = str(raw_response or "").strip()
    if not answer:
        raise RuntimeError("Vision model returned an empty final answer.")
    return answer


def build_answer_payload(
    answer: str,
    raw_response: str,
    prompt: str,
    image_paths: list[Path],
    package: dict,
    config: VlmGeneratorConfig,
) -> dict:
    payload = {
        "query": package.get("query"),
        "document_id": package.get("document_id"),
        "answer": answer,
        "evidence_package_path": str(config.evidence_package_path.resolve()),
        "prompt_path": str(config.prompt_path.resolve()),
        "model": config.model,
        "image_paths": [str(path.resolve()) for path in image_paths],
        "generation_config": {
            **asdict(config),
            "project_root": str(config.project_root),
            "evidence_package_path": str(config.evidence_package_path),
            "output_path": str(config.output_path) if config.output_path else None,
            "prompt_path": str(config.prompt_path),
            "api_key": "***" if config.api_key else None,
        },
        "created_at_unix": int(time.time()),
    }
    if config.include_raw_response:
        payload["raw_response"] = raw_response
        payload["prompt"] = prompt
    return payload


def write_answer_payload(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resolve_project_path(value: object, project_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()
