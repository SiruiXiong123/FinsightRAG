import argparse
import json
from pathlib import Path


from finsightrag.paths import default_project_root


PROJECT_ROOT = default_project_root()

from finsightrag.document_indexer import (
    DocumentIndexConfig,
    build_document_indexes,
    require_existing_paths,
    resolve_default_paths,
)
from finsightrag.document_catalog import DocumentCatalog, catalog_path_for_index_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build per-document FAISS indexes for text, table, and image records."
    )
    parser.add_argument("--config-path", type=Path, default=None, help="Optional runtime config.yaml path.")
    parser.add_argument(
        "--document-id",
        dest="document_id",
        default=None,
        help="Document ID. Defaults to the source file stem, for example MorganStanleyQ10.",
    )
    parser.add_argument("--source-file", type=Path, default=None, help="Source PDF path or filename.")
    parser.add_argument("--ocr-output-dir", type=Path, default=None, help="Directory containing OCR outputs.")
    parser.add_argument("--text-chunks", type=Path, default=None, help="Text chunk_merge JSON.")
    parser.add_argument("--table-enrichment", type=Path, default=None, help="Table enrichment JSON.")
    parser.add_argument("--image-enrichment", type=Path, default=None, help="Image enrichment JSON.")
    parser.add_argument("--table-asset-dir", type=Path, default=None, help="Directory containing cropped table PNGs.")
    parser.add_argument("--image-asset-dir", type=Path, default=None, help="Directory containing cropped image PNGs.")
    parser.add_argument("--table-asset-manifest", type=Path, default=None, help="Table asset_manifest.jsonl.")
    parser.add_argument("--image-asset-manifest", type=Path, default=None, help="Image asset_manifest.jsonl.")
    parser.add_argument("--index-root", type=Path, default=None, help="Root directory for indexes.")
    parser.add_argument("--embedding-model", default=None, help="SentenceTransformer model for FAISS vectors.")
    parser.add_argument("--embed-batch-size", type=int, default=None, help="Embedding batch size.")
    parser.add_argument("--device", default=None, help="Optional SentenceTransformer device, for example cpu or cuda.")
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Optional document metadata as key=value. Can be repeated.",
    )
    parser.add_argument("--metadata-json", type=Path, default=None, help="JSON file with document metadata.")
    parser.add_argument(
        "--ocr-timing-report",
        type=Path,
        default=None,
        help="Optional OCR timing JSON for reporting metadata only.",
    )
    parser.add_argument(
        "--auto-ocr-timing",
        action="store_true",
        help="Infer OCR timing metadata from runs/ocr_timing. Disabled by default.",
    )
    parser.add_argument(
        "--no-auto-ocr-timing",
        action="store_true",
        help="Compatibility no-op. build-indexes no longer reads OCR timing unless explicitly requested.",
    )
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if index files already exist.")
    parser.add_argument("--catalog-path", type=Path, default=None, help="Catalog JSON path. Defaults to <index-root>/catalog.json.")
    parser.add_argument("--no-catalog", action="store_true", help="Do not update indexes/catalog.json.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    defaults = resolve_default_paths(
        project_root=PROJECT_ROOT,
        config_path=args.config_path,
        document_id=args.document_id,
        ocr_output_dir=args.ocr_output_dir,
        source_file=args.source_file,
    )
    source_file = str(args.source_file) if args.source_file else defaults["source_file"]
    document_id = args.document_id or defaults["document_id"]
    config = DocumentIndexConfig(
        document_id=document_id,
        source_file=source_file,
        text_chunks_path=args.text_chunks or defaults["text_chunks_path"],
        table_enrichment_path=args.table_enrichment or defaults["table_enrichment_path"],
        image_enrichment_path=args.image_enrichment or defaults["image_enrichment_path"],
        table_asset_dir=args.table_asset_dir or defaults["table_asset_dir"],
        image_asset_dir=args.image_asset_dir or defaults["image_asset_dir"],
        table_asset_manifest_path=args.table_asset_manifest or defaults["table_asset_manifest_path"],
        image_asset_manifest_path=args.image_asset_manifest or defaults["image_asset_manifest_path"],
        index_root=args.index_root or defaults["index_root"],
        project_root=PROJECT_ROOT,
        embedding_model=args.embedding_model or defaults["embedding_model"],
        embed_batch_size=args.embed_batch_size or defaults["embed_batch_size"],
        device=args.device if args.device is not None else defaults["device"],
        overwrite=not args.no_overwrite,
        metadata=build_metadata(args, defaults, document_id, source_file),
    )
    require_existing_paths(
        [
            config.text_chunks_path,
            config.table_enrichment_path,
            config.image_enrichment_path,
            config.table_asset_dir,
            config.image_asset_dir,
            config.table_asset_manifest_path,
            config.image_asset_manifest_path,
        ]
    )
    manifest = build_document_indexes(config)
    if not args.no_catalog:
        catalog_path = args.catalog_path or catalog_path_for_index_root(config.index_root, PROJECT_ROOT)
        catalog = DocumentCatalog.load(catalog_path, PROJECT_ROOT)
        entry = catalog.upsert_from_manifest(manifest)
        catalog.save()
        print(f"Updated catalog: {catalog.catalog_path}")
        print(f"  {entry['document_id']}: {entry['manifest_path']}")

    print(f"Built indexes for {manifest['document_id']} with {manifest['embedding_model']}")
    for modality, output in manifest["modalities"].items():
        print(
            f"  {modality}: {output['count']} vectors, dim={output['dimension']}, "
            f"faiss={output['faiss_path']}, metadata={output['metadata_path']}"
        )
    print(f"  manifest: {config.index_dir / 'manifest.json'}")
    return 0


def build_metadata(args, defaults: dict, document_id: str, source_file: str) -> dict[str, str]:
    metadata = dict(defaults.get("metadata") or {})
    timing_report = args.ocr_timing_report
    if timing_report is None and args.auto_ocr_timing and not args.no_auto_ocr_timing:
        timing_report = find_latest_ocr_timing_report(PROJECT_ROOT, document_id, source_file)
    if timing_report:
        metadata.update(load_ocr_timing_metadata(timing_report))
    if args.metadata_json:
        loaded = json.loads(args.metadata_json.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise SystemExit("--metadata-json must contain a JSON object.")
        metadata.update({str(key): str(value) for key, value in loaded.items() if value is not None})
    for item in args.metadata:
        if "=" not in item:
            raise SystemExit(f"--metadata must use key=value format: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--metadata key cannot be empty: {item}")
        metadata[key] = value.strip()
    return metadata


def find_latest_ocr_timing_report(project_root: Path, document_id: str, source_file: str) -> Path | None:
    report_dir = project_root / "runs" / "ocr_timing"
    if not report_dir.exists():
        return None

    reports = sorted(
        report_dir.glob("paddleocr_timing_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not reports:
        return None

    source_path = Path(str(source_file))
    tokens = {
        str(document_id),
        source_path.name,
        source_path.stem,
    }
    tokens = {token for token in tokens if token}
    for report in reports:
        text = report.read_text(encoding="utf-8", errors="ignore")
        if any(token in text for token in tokens):
            return report

    if len(reports) == 1:
        return reports[0]
    return None


def load_ocr_timing_metadata(report_path: Path) -> dict[str, str]:
    report_path = resolve_project_path(report_path)
    data = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = {
        "ocr_timing_report": project_relative_path(report_path),
    }
    if data.get("started_at"):
        metadata["ocr_started_at"] = str(data["started_at"])
    if data.get("finished_at"):
        metadata["ocr_completed_at"] = str(data["finished_at"])
    if data.get("elapsed_seconds") is not None:
        metadata["ocr_elapsed_seconds"] = str(data["elapsed_seconds"])
    if data.get("exit_code") is not None:
        metadata["ocr_exit_code"] = str(data["exit_code"])
    return metadata


def resolve_project_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())

