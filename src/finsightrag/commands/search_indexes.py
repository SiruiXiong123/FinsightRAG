import argparse
import json
from pathlib import Path


from finsightrag.paths import default_project_root


PROJECT_ROOT = default_project_root()

from finsightrag.vector_store import MultiModalVectorStore, SUPPORTED_MODALITIES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search per-document text/table/image FAISS indexes."
    )
    parser.add_argument("--config-path", type=Path, default=None, help="Optional runtime config.yaml path.")
    parser.add_argument("--document-id", required=True, help="Document ID, usually the source PDF filename stem.")
    parser.add_argument("--query", required=True, help="Search query.")
    parser.add_argument(
        "--modality",
        choices=[*SUPPORTED_MODALITIES, "all"],
        default="all",
        help="Modality to search.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    store = MultiModalVectorStore(config_path=args.config_path)
    if args.modality == "all":
        output = store.search_all(document_id=args.document_id, query=args.query)
    else:
        output = store.search(document_id=args.document_id, query=args.query, modality=args.modality)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

