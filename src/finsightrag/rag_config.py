import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .paths import default_project_root

try:
    import yaml
except ImportError:  # pragma: no cover - fallback keeps config loading lightweight.
    yaml = None


MISSING_VALUES = {"", "...", "<your api key>", "<your_api_key>", "none", "null"}
CONFIG_FILENAME = "config.yaml"
CONFIG_TEMPLATE_FILENAME = "config.example.yaml"
CONFIG_ENV_NAMES = ("RAG_CONFIG_PATH", "MULTIFINRAG_CONFIG")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING_VALUES
    return False


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


def _repo_root() -> Path:
    return default_project_root()


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
    """Loads one runtime YAML config file for the project."""

    def __init__(self, values: Optional[Dict[str, Any]] = None, path: Optional[Path] = None):
        self.values = values or {}
        self.path = path

    @classmethod
    def load(cls, explicit_path: Optional[str] = None) -> "RagConfig":
        path, required = cls.resolve_config_path(explicit_path)
        if path.exists() and path.is_file():
            if path.name == CONFIG_TEMPLATE_FILENAME:
                raise ValueError(
                    f"{CONFIG_TEMPLATE_FILENAME} is a template. Copy it to {CONFIG_FILENAME} "
                    "or pass the runtime config.yaml path."
                )
            if yaml is not None:
                loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            else:
                loaded = _read_simple_yaml(path)
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file must contain a mapping: {path}")
            return cls(values=loaded, path=path)
        if required:
            raise FileNotFoundError(f"Config file not found: {path}")
        return cls()

    @staticmethod
    def candidate_paths(explicit_path: Optional[str] = None) -> Iterable[Path]:
        path, _ = RagConfig.resolve_config_path(explicit_path)
        yield path

    @staticmethod
    def resolve_config_path(explicit_path: Optional[str] = None) -> tuple[Path, bool]:
        if explicit_path:
            return _expand_path(explicit_path).resolve(), True
        for env_name in CONFIG_ENV_NAMES:
            env_value = os.getenv(env_name)
            if env_value:
                return _expand_path(env_value).resolve(), True
        return (_repo_root() / CONFIG_FILENAME).resolve(), False

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
        return self.get("vision_model")

    @property
    def vision_base_url(self) -> Optional[str]:
        return self.get(
            "vision_binding_host",
            env_names=("VISION_BINDING_HOST", "OPENAI_BASE_URL", "SILICONFLOW_BASE_URL"),
        )

    @property
    def vision_api_key(self) -> Optional[str]:
        return self.get(
            "vision_binding_api_key",
            env_names=("VISION_API_KEY", "OPENAI_API_KEY", "SILICONFLOW_API_KEY"),
        )
