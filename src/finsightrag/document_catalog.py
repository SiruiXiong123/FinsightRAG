import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


CATALOG_FILENAME = "catalog.json"
CATALOG_SCHEMA_VERSION = 1
MODALITY_PATH_KEYS = {"faiss_path", "metadata_path"}
CATALOG_PATH_KEYS = {"source_file", "manifest_path", "index_dir"}
PARSED_AT_KEYS = ("parsed_at", "ocr_parsed_at", "ocr_completed_at")
OCR_DURATION_KEYS = ("ocr_elapsed_seconds", "ocr_duration_seconds", "parse_elapsed_seconds")


@dataclass
class DocumentCatalog:
    catalog_path: Path
    project_root: Path
    documents: list[dict] = field(default_factory=list)
    schema_version: int = CATALOG_SCHEMA_VERSION

    @classmethod
    def load(
        cls,
        catalog_path: Path,
        project_root: Optional[Path] = None,
    ) -> "DocumentCatalog":
        root = Path(project_root or Path.cwd()).resolve()
        catalog_path = resolve_project_path(catalog_path, root)
        schema_version = CATALOG_SCHEMA_VERSION
        documents = []
        if catalog_path.exists():
            data = json.loads(catalog_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                documents = data
            elif isinstance(data, dict):
                schema_version = int(data.get("schema_version") or CATALOG_SCHEMA_VERSION)
                documents = data.get("documents", [])
            else:
                raise ValueError(f"Catalog must be a JSON object or list: {catalog_path}")
            if not isinstance(documents, list):
                raise ValueError(f"Catalog documents must be a list: {catalog_path}")

        return cls(
            catalog_path=catalog_path.resolve(),
            project_root=root,
            documents=[normalize_entry(entry, root) for entry in documents],
            schema_version=schema_version,
        )

    def save(self) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": self.schema_version,
            "updated_at": utc_now(),
            "documents": self.list_documents(),
        }
        self.catalog_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def upsert_from_manifest(
        self,
        manifest: dict,
        manifest_path: Optional[Path] = None,
    ) -> dict:
        entry = entry_from_manifest(manifest, manifest_path, self.project_root)
        self.documents = [
            existing
            for existing in self.documents
            if existing.get("document_id") != entry["document_id"]
        ]
        self.documents.append(entry)
        return entry

    def scan_index_root(self, index_root: Path) -> list[dict]:
        index_root = resolve_project_path(index_root, self.project_root)
        entries = []
        for manifest_path in sorted(index_root.glob("*/manifest.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entries.append(self.upsert_from_manifest(manifest, manifest_path))
        return entries

    def list_documents(self) -> list[dict]:
        return sorted(
            [dict(entry) for entry in self.documents],
            key=lambda entry: str(entry.get("document_id") or ""),
        )

    def get(self, document_id: str) -> Optional[dict]:
        for entry in self.documents:
            if entry.get("document_id") == document_id:
                return dict(entry)
        return None

    def validate_paths(self) -> list[str]:
        missing = set()
        for entry in self.documents:
            manifest_path = entry.get("manifest_path")
            if not manifest_path:
                missing.add(f"<missing manifest_path for {entry.get('document_id')}>")
                continue

            resolved_manifest_path = resolve_project_path(manifest_path, self.project_root)
            if not resolved_manifest_path.exists():
                missing.add(str(resolved_manifest_path))
                continue

            manifest = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
            for path in manifest_paths(manifest):
                resolved = resolve_project_path(path, self.project_root)
                if not resolved.exists():
                    missing.add(str(resolved))
        return sorted(missing)


def catalog_path_for_index_root(index_root: Path, project_root: Path) -> Path:
    return resolve_project_path(index_root, project_root) / CATALOG_FILENAME


def entry_from_manifest(
    manifest: dict,
    manifest_path: Optional[Path],
    project_root: Path,
) -> dict:
    document_id = manifest.get("document_id")
    if not document_id:
        raise ValueError("Manifest is missing document_id.")

    index_dir = manifest.get("index_dir")
    if manifest_path is None:
        if not index_dir:
            raise ValueError("manifest_path is required when manifest has no index_dir.")
        manifest_path = resolve_project_path(index_dir, project_root) / "manifest.json"

    metadata = normalize_metadata(manifest.get("metadata"))
    entry = {
        "document_id": str(document_id),
        "source_file": optional_catalog_path(manifest.get("source_file"), project_root),
        "manifest_path": relative_path(manifest_path, project_root),
        "index_dir": optional_catalog_path(index_dir or Path(manifest_path).parent, project_root),
        "embedding_model": optional_string(manifest.get("embedding_model")),
        "indexed_at": optional_string(manifest.get("indexed_at")),
        "parsed_at": first_present(manifest, metadata, PARSED_AT_KEYS),
        "ocr_elapsed_seconds": first_float(manifest, metadata, OCR_DURATION_KEYS),
        "metadata": metadata,
        "modalities": normalize_modalities(
            manifest.get("modalities") or manifest.get("outputs") or {},
            project_root,
        ),
    }
    return drop_empty(entry)


def manifest_paths(manifest: dict) -> Iterable[str]:
    for key in ("source_file", "index_dir"):
        value = manifest.get(key)
        if value:
            yield str(value)

    for section_name in ("modalities", "outputs"):
        section = manifest.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for value in section.values():
            if not isinstance(value, dict):
                continue
            for key in MODALITY_PATH_KEYS:
                if value.get(key):
                    yield str(value[key])

    inputs = manifest.get("inputs", {})
    if isinstance(inputs, dict):
        for value in inputs.values():
            if isinstance(value, str) and value:
                yield value


def normalize_entry(entry: dict, project_root: Path) -> dict:
    if not isinstance(entry, dict):
        raise ValueError(f"Invalid catalog entry: {entry}")
    document_id = entry.get("document_id")
    manifest_path = entry.get("manifest_path")
    if not document_id or not manifest_path:
        raise ValueError(f"Invalid catalog entry: {entry}")

    normalized = dict(entry)
    normalized["document_id"] = str(document_id)
    normalized["metadata"] = normalize_metadata(normalized.get("metadata"))
    if "modalities" in normalized:
        normalized["modalities"] = normalize_modalities(normalized.get("modalities"), project_root)

    for key in CATALOG_PATH_KEYS:
        if normalized.get(key):
            normalized[key] = optional_catalog_path(normalized[key], project_root)

    if normalized.get("ocr_elapsed_seconds") is not None:
        normalized["ocr_elapsed_seconds"] = to_float(normalized["ocr_elapsed_seconds"])
    return drop_empty(normalized)


def normalize_modalities(modalities: object, project_root: Path) -> dict:
    if not isinstance(modalities, dict):
        return {}

    normalized = {}
    for modality, data in modalities.items():
        if not isinstance(data, dict):
            continue
        item = {}
        for key, value in data.items():
            if key in MODALITY_PATH_KEYS and value:
                item[key] = optional_catalog_path(value, project_root)
            elif key in {"count", "dimension"}:
                item[key] = to_int(value)
            else:
                item[key] = value
        normalized[str(modality)] = drop_empty(item)
    return normalized


def normalize_metadata(metadata: object) -> dict[str, str]:
    if not isinstance(metadata, dict):
        return {}
    return {str(key): str(value) for key, value in metadata.items() if value is not None}


def resolve_project_path(path: str | Path, project_root: Path) -> Path:
    path = Path(str(path))
    if path.is_absolute():
        return path
    return project_root / path


def relative_path(path: str | Path, project_root: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def optional_catalog_path(value: object, project_root: Path) -> str:
    if value is None or str(value).strip() == "":
        return ""
    return relative_path(resolve_project_path(str(value), project_root), project_root)


def first_present(manifest: dict, metadata: dict, keys: Iterable[str]) -> str:
    for key in keys:
        value = manifest.get(key)
        if value is None:
            value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def first_float(manifest: dict, metadata: dict, keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        value = manifest.get(key)
        if value is None:
            value = metadata.get(key)
        if value is not None and str(value).strip():
            return to_float(value)
    return None


def optional_string(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def to_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def drop_empty(data: dict) -> dict:
    return {
        key: value
        for key, value in data.items()
        if value not in ("", None, {}, [])
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
