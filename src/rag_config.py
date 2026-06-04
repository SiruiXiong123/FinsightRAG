import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - fallback keeps config loading lightweight.
    yaml = None


MISSING_VALUES = {"", "...", "<your api key>", "<your_api_key>", "none", "null"}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING_VALUES
    return False


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_simple_yaml(path: Path) -> Dict[str, Any]:
    """Tiny key/value YAML fallback for simple flat config files."""
    values: Dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key.strip()] = value.strip("\"'")
    return values


class RagConfig:
    """Loads the project config without forcing one fixed local path."""

    def __init__(self, values: Optional[Dict[str, Any]] = None, path: Optional[Path] = None):
        self.values = values or {}
        self.path = path

    @classmethod
    def load(cls, explicit_path: Optional[str] = None) -> "RagConfig":
        for path in cls.candidate_paths(explicit_path):
            if path.exists() and path.is_file():
                if yaml is not None:
                    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                else:
                    loaded = _read_simple_yaml(path)
                if not isinstance(loaded, dict):
                    raise ValueError(f"Config file must contain a mapping: {path}")
                return cls(values=loaded, path=path)
        return cls()

    @staticmethod
    def candidate_paths(explicit_path: Optional[str] = None) -> Iterable[Path]:
        seen = set()
        raw_paths = []
        if explicit_path:
            raw_paths.append(explicit_path)
        raw_paths.extend(
            value
            for value in (
                os.getenv("RAG_CONFIG_PATH"),
                os.getenv("MULTIFINRAG_CONFIG"),
            )
            if value
        )

        root = _repo_root()
        raw_paths.extend(
            [
                str(Path.cwd() / "config.yaml"),
                str(root / "config.yaml"),
                str(root.parent / "config.yaml"),
            ]
        )

        for raw_path in raw_paths:
            path = _expand_path(raw_path).resolve()
            if path not in seen:
                seen.add(path)
                yield path

    def get(self, key: str, default: Any = None, env_names: Optional[Iterable[str]] = None) -> Any:
        value = self.values.get(key)
        if not _is_missing(value):
            return value
        for env_name in env_names or ():
            env_value = os.getenv(env_name)
            if not _is_missing(env_value):
                return env_value
        return default

    def get_int(self, key: str, default: int) -> int:
        value = self.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def get_float(self, key: str, default: float) -> float:
        value = self.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def get_path(self, key: str, default: Optional[str] = None) -> Optional[Path]:
        value = self.get(key, default)
        if _is_missing(value):
            return None
        return _expand_path(str(value)).resolve()

    @property
    def vision_model(self) -> Optional[str]:
        return self.get("vision_model", self.get("llm_model"))

    @property
    def vision_base_url(self) -> Optional[str]:
        return self.get(
            "vision_binding_host",
            self.get("llm_binding_host"),
            env_names=("VISION_BINDING_HOST", "OPENAI_BASE_URL", "SILICONFLOW_BASE_URL"),
        )

    @property
    def vision_api_key(self) -> Optional[str]:
        return self.get(
            "vision_binding_api_key",
            self.get("llm_binding_api_key"),
            env_names=("VISION_API_KEY", "OPENAI_API_KEY", "SILICONFLOW_API_KEY"),
        )

    @property
    def paddleocr_command(self) -> Optional[str]:
        return self.get("paddleocr_command", env_names=("PADDLEOCR_COMMAND",))

    def paddleocr_output_dirs(self) -> Iterable[Path]:
        keys = (
            "paddleocr_output_dir",
            "paddleocr_vl_output_dir",
            "ocr_output_dir",
            "output_dir",
        )
        for key in keys:
            path = self.get_path(key)
            if path is not None:
                yield path

