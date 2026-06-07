import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from .document_catalog import CATALOG_FILENAME, resolve_project_path
    from .rag_config import RagConfig
except ImportError:
    from document_catalog import CATALOG_FILENAME, resolve_project_path
    from rag_config import RagConfig


SUPPORTED_MODALITIES = ("text", "table", "image")


@dataclass(frozen=True)
class VectorStoreConfig:
    project_root: Path
    index_root: Path
    config_path: Optional[Path]
    embedding_model: Optional[str]
    device: Optional[str]
    thresholds: dict[str, float]
    candidate_k: dict[str, int]


class MultiModalVectorStore:
    def __init__(
        self,
        config_path: Optional[Path | str] = None,
        project_root: Optional[Path | str] = None,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
        self.rag_config = RagConfig.load(str(config_path) if config_path else None)
        self.config = build_vector_store_config(
            rag_config=self.rag_config,
            project_root=self.project_root,
        )
        self._models = {}
        self._indexes = {}
        self._metadata = {}
        self._manifests = {}
        self._catalog = None

    def search(self, document_id: str, query: str, modality: str) -> list[dict]:
        modality = normalize_modality(modality)
        query = str(query or "").strip()
        if not query:
            raise ValueError("query cannot be empty.")

        manifest = self.load_manifest(document_id)
        modality_outputs = get_manifest_modalities(manifest)
        if modality not in modality_outputs:
            raise ValueError(f"Document {document_id} has no {modality} index.")

        output = modality_outputs[modality]
        faiss_path = self.resolve_path(output.get("faiss_path"))
        metadata_path = self.resolve_path(output.get("metadata_path"))
        index = self.load_faiss_index(faiss_path)
        metadata_by_id = self.load_metadata(metadata_path)
        if index.ntotal <= 0:
            return []

        embedding_model = self.embedding_model_name(manifest)
        model = self.load_embedding_model(embedding_model)
        query_embedding = encode_query(model, query)
        validate_query_embedding_dimension(
            query_embedding=query_embedding,
            index=index,
            document_id=document_id,
            modality=modality,
            embedding_model=embedding_model,
        )
        scores, faiss_ids = index.search(query_embedding, self.config.candidate_k[modality])
        threshold = self.config.thresholds[modality]

        results = []
        for rank, (score, faiss_id) in enumerate(zip(scores[0], faiss_ids[0]), start=1):
            if int(faiss_id) < 0:
                continue
            score = float(score)
            if score < threshold:
                continue
            chunk = metadata_by_id.get(int(faiss_id))
            if chunk is None:
                continue
            results.append(
                {
                    "document_id": document_id,
                    "modality": modality,
                    "rank": rank,
                    "score": round(score, 6),
                    "threshold": threshold,
                    **chunk,
                }
            )
        return results

    def search_text(self, document_id: str, query: str) -> list[dict]:
        return self.search(document_id=document_id, query=query, modality="text")

    def search_table(self, document_id: str, query: str) -> list[dict]:
        return self.search(document_id=document_id, query=query, modality="table")

    def search_image(self, document_id: str, query: str) -> list[dict]:
        return self.search(document_id=document_id, query=query, modality="image")

    def search_all(
        self,
        document_id: str,
        query: str,
        modalities: Optional[list[str]] = None,
    ) -> dict[str, list[dict]]:
        selected_modalities = modalities or list(SUPPORTED_MODALITIES)
        return {
            modality: self.search(document_id, query, normalize_modality(modality))
            for modality in selected_modalities
        }

    def load_manifest(self, document_id: str) -> dict:
        document_id = normalize_document_id(document_id)
        if document_id not in self._manifests:
            manifest_path = self.resolve_manifest_path(document_id)
            self._manifests[document_id] = json.loads(manifest_path.read_text(encoding="utf-8"))
        return self._manifests[document_id]

    def resolve_manifest_path(self, document_id: str) -> Path:
        document_id = normalize_document_id(document_id)
        for entry in self.load_catalog().get("documents", []):
            if entry.get("document_id") == document_id and entry.get("manifest_path"):
                return self.resolve_path(entry["manifest_path"])

        manifest_path = self.config.index_root / document_id / "manifest.json"
        if manifest_path.exists():
            return manifest_path
        raise FileNotFoundError(f"Manifest not found for document_id={document_id}: {manifest_path}")

    def load_catalog(self) -> dict:
        if self._catalog is None:
            catalog_path = self.config.index_root / CATALOG_FILENAME
            if not catalog_path.exists():
                self._catalog = {"documents": []}
            else:
                data = json.loads(catalog_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    documents = data
                elif isinstance(data, dict):
                    documents = data.get("documents", [])
                else:
                    raise ValueError(f"Catalog must be a JSON object or list: {catalog_path}")
                if not isinstance(documents, list):
                    raise ValueError(f"Catalog documents must be a list: {catalog_path}")
                self._catalog = {"documents": documents}
        return self._catalog

    def resolve_path(self, value: object) -> Path:
        if value is None or str(value).strip() == "":
            raise ValueError("Path value cannot be empty.")
        return resolve_project_path(str(value), self.project_root).resolve()

    def load_faiss_index(self, faiss_path: Path):
        faiss_path = faiss_path.resolve()
        if faiss_path not in self._indexes:
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError("faiss is required for vector search. Install faiss-cpu in llm_env.") from exc
            if not faiss_path.exists():
                raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
            self._indexes[faiss_path] = faiss.read_index(str(faiss_path))
        return self._indexes[faiss_path]

    def load_metadata(self, metadata_path: Path) -> dict[int, dict]:
        metadata_path = metadata_path.resolve()
        if metadata_path not in self._metadata:
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata JSONL not found: {metadata_path}")
            records = {}
            for line_number, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                faiss_id = item.get("faiss_id")
                if faiss_id is None:
                    raise ValueError(f"Missing faiss_id in {metadata_path} line {line_number}.")
                records[int(faiss_id)] = item
            self._metadata[metadata_path] = records
        return self._metadata[metadata_path]

    def embedding_model_name(self, manifest: dict) -> str:
        model_name = manifest.get("embedding_model") or self.config.embedding_model
        if model_name is None:
            model_name = self.rag_config.get("embedding_model")
        if model_name is None:
            raise ValueError(f"Missing embedding_model in {config_label(self.rag_config)} or document manifest.")
        return str(model_name)

    def load_embedding_model(self, embedding_model: str):
        cache_key = (embedding_model, self.config.device)
        if cache_key not in self._models:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for vector search. "
                    "Install it in llm_env with: python -m pip install sentence-transformers"
                ) from exc
            if self.config.device:
                self._models[cache_key] = SentenceTransformer(embedding_model, device=self.config.device)
            else:
                self._models[cache_key] = SentenceTransformer(embedding_model)
        return self._models[cache_key]


def build_vector_store_config(
    rag_config: RagConfig,
    project_root: Path,
) -> VectorStoreConfig:
    retrieval_settings = load_section(rag_config, "retrieval", required=True)
    indexing_settings = load_section(rag_config, "indexing", required=False)

    index_root = indexing_settings.get("index_root")
    if index_root is None:
        raise ValueError(f"Missing indexing.index_root in {config_label(rag_config)}.")

    return VectorStoreConfig(
        project_root=project_root,
        index_root=resolve_project_path(str(index_root), project_root).resolve(),
        config_path=rag_config.path,
        embedding_model=optional_string(retrieval_settings.get("embedding_model")),
        device=optional_string(retrieval_settings.get("device") or indexing_settings.get("device")),
        thresholds=load_retrieval_floats(
            settings=retrieval_settings,
            suffix="threshold",
            rag_config=rag_config,
        ),
        candidate_k=load_retrieval_ints(
            settings=retrieval_settings,
            suffix="candidate_k",
            rag_config=rag_config,
        ),
    )


def load_section(rag_config: RagConfig, section_name: str, required: bool) -> dict:
    section = rag_config.values.get(section_name, {})
    if isinstance(section, dict):
        return section
    if required:
        raise ValueError(f"Missing or invalid {section_name} section in {config_label(rag_config)}.")
    return {}


def load_retrieval_floats(settings: dict, suffix: str, rag_config: RagConfig) -> dict[str, float]:
    values = {}
    for modality in SUPPORTED_MODALITIES:
        key = f"{modality}_{suffix}"
        values[modality] = to_float(require_setting(settings, key, rag_config), f"retrieval.{key}")
    return values


def load_retrieval_ints(settings: dict, suffix: str, rag_config: RagConfig) -> dict[str, int]:
    values = {}
    for modality in SUPPORTED_MODALITIES:
        key = f"{modality}_{suffix}"
        values[modality] = max(1, to_int(require_setting(settings, key, rag_config), f"retrieval.{key}"))
    return values


def require_setting(settings: dict, key: str, rag_config: RagConfig) -> object:
    if settings.get(key) is None:
        raise ValueError(f"Missing retrieval.{key} in {config_label(rag_config)}.")
    return settings[key]


def get_manifest_modalities(manifest: dict) -> dict:
    outputs = manifest.get("modalities") or manifest.get("outputs") or {}
    if not isinstance(outputs, dict):
        return {}
    return outputs


def normalize_modality(modality: str) -> str:
    modality = str(modality or "").strip().lower()
    if modality not in SUPPORTED_MODALITIES:
        raise ValueError(f"Unsupported modality: {modality}. Use one of: {', '.join(SUPPORTED_MODALITIES)}")
    return modality


def normalize_document_id(document_id: str) -> str:
    document_id = str(document_id or "").strip()
    if not document_id:
        raise ValueError("document_id cannot be empty.")
    return document_id


def encode_query(model, query: str):
    return model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")


def validate_query_embedding_dimension(
    query_embedding,
    index,
    document_id: str,
    modality: str,
    embedding_model: str,
) -> None:
    query_dim = int(query_embedding.shape[1]) if len(query_embedding.shape) == 2 else 0
    index_dim = getattr(index, "d", None)
    if index_dim is None:
        return
    index_dim = int(index_dim)
    if query_dim != index_dim:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"document_id={document_id}, modality={modality}, "
            f"embedding_model={embedding_model}, query_dim={query_dim}, index_dim={index_dim}"
        )


def optional_string(value: object) -> Optional[str]:
    if value is None or str(value).strip() == "":
        return None
    return str(value)


def to_float(value: object, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number, got {value!r}.") from exc


def to_int(value: object, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}.") from exc


def config_label(rag_config: RagConfig) -> str:
    return str(rag_config.path or "config.yaml")
