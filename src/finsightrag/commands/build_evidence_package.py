import argparse
import json
from pathlib import Path


from finsightrag.paths import default_project_root


PROJECT_ROOT = default_project_root()

from finsightrag.evidence_builder import EvidenceBuilder, safe_filename, write_evidence_package
from finsightrag.rag_config import RagConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a VLM evidence package and table/image montages."
    )
    parser.add_argument("--config-path", type=Path, default=None, help="Optional runtime config.yaml path.")
    parser.add_argument("--retriever-result", type=Path, default=None, help="JSON output from retriever.retrieve().")
    parser.add_argument("--document-id", default=None, help="Document ID for preview mode.")
    parser.add_argument("--query", default="evidence preview", help="Query used in preview mode.")
    parser.add_argument("--ocr-output-dir", type=Path, default=None, help="Directory containing OCR/index intermediates.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for evidence package and montage files.")
    parser.add_argument("--table-limit", type=int, default=4, help="Preview mode table evidence count.")
    parser.add_argument("--image-limit", type=int, default=3, help="Preview mode image evidence count.")
    parser.add_argument("--package-output", type=Path, default=None, help="Evidence package JSON output path.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    builder = EvidenceBuilder.from_config_path(
        config_path=args.config_path,
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
    )
    retriever_result = load_or_build_retriever_result(args)
    package = builder.build(retriever_result, output_dir=args.output_dir)
    package_output = args.package_output or default_package_path(
        output_dir=Path(args.output_dir).resolve() if args.output_dir else builder.config.output_dir,
        document_id=package["document_id"],
    )
    write_evidence_package(package, package_output)

    print(f"Wrote evidence package: {package_output}")
    for kind, montage in package["montages"].items():
        if montage:
            print(f"Wrote {kind} montage: {montage['absolute_path']}")
        else:
            print(f"No {kind} montage created.")
    return 0


def load_or_build_retriever_result(args) -> dict:
    if args.retriever_result:
        return json.loads(args.retriever_result.read_text(encoding="utf-8"))
    return build_preview_retriever_result(args)


def build_preview_retriever_result(args) -> dict:
    rag_config = RagConfig.load(str(args.config_path) if args.config_path else None)
    document_id = args.document_id or infer_document_id(rag_config)
    output_dir = (
        args.ocr_output_dir
        or rag_config.get_path("paddleocr_output_dir")
        or PROJECT_ROOT / "data" / "output"
    )
    output_dir = Path(output_dir).resolve()
    table_contexts = load_asset_contexts(
        document_id=document_id,
        modality="table",
        asset_dir=output_dir / f"{document_id}_tables",
        enrichment_path=output_dir / f"{document_id}_table_enrichment.json",
        limit=max(0, args.table_limit),
    )
    image_contexts = load_asset_contexts(
        document_id=document_id,
        modality="image",
        asset_dir=output_dir / f"{document_id}_images",
        enrichment_path=output_dir / f"{document_id}_image_enrichment.json",
        limit=max(0, args.image_limit),
    )
    return {
        "query": args.query,
        "document_id": document_id,
        "retrieval_mode": "preview",
        "fallback_triggered": bool(table_contexts or image_contexts),
        "contexts": table_contexts + image_contexts,
        "trace": {
            "num_text": 0,
            "num_table": len(table_contexts),
            "num_image": len(image_contexts),
            "total": len(table_contexts) + len(image_contexts),
        },
    }


def load_asset_contexts(
    document_id: str,
    modality: str,
    asset_dir: Path,
    enrichment_path: Path,
    limit: int,
) -> list[dict]:
    if limit <= 0:
        return []
    manifest_path = asset_dir / "asset_manifest.jsonl"
    if not manifest_path.exists():
        return []
    enrichment_by_filename = load_enrichment_by_filename(enrichment_path)
    contexts = []
    for item in load_jsonl(manifest_path):
        filename = str(item.get("filename") or "").strip()
        if not filename:
            continue
        enrichment = enrichment_by_filename.get(filename, {})
        contexts.append(
            {
                "id": f"{modality}_{int(item.get('asset_index') or len(contexts) + 1):06d}",
                "document_id": document_id,
                "modality": modality,
                "page": item.get("page"),
                "bbox": item.get("bbox") or item.get("asset_bbox"),
                "title": enrichment.get("title") or item.get("title"),
                "content": enrichment.get("content") or item.get("title"),
                "summary": enrichment.get("metadata"),
                "crop_path": str(asset_dir / (item.get("asset_path") or filename)),
                "asset_path": str(asset_dir / (item.get("asset_path") or filename)),
                "score": None,
            }
        )
        if len(contexts) >= limit:
            break
    return contexts


def load_enrichment_by_filename(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item.get("filename") or ""): item
        for item in data.get("results", [])
        if item.get("filename")
    }


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def infer_document_id(rag_config: RagConfig) -> str:
    input_file = rag_config.get_path("input_file")
    if input_file:
        return input_file.stem
    raise SystemExit("Cannot infer document ID. Pass --document-id.")


def default_package_path(output_dir: Path, document_id: str) -> Path:
    return output_dir / f"{safe_filename(document_id)}_evidence_package.json"


if __name__ == "__main__":
    raise SystemExit(main())

