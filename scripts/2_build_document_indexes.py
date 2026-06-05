import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_indexer import (
    DocumentIndexConfig,
    build_document_indexes,
    require_existing_paths,
    resolve_default_paths,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build per-document FAISS indexes for text, table, and image records."
    )
    parser.add_argument("--config-path", type=Path, default=None, help="Optional YAML config path.")
    parser.add_argument("--document-name", default=None, help="Document stem, for example MorganStanleyQ10.")
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
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if index files already exist.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    defaults = resolve_default_paths(
        project_root=PROJECT_ROOT,
        config_path=args.config_path,
        document_name=args.document_name,
        ocr_output_dir=args.ocr_output_dir,
        source_file=args.source_file,
    )
    source_file = args.source_file.name if args.source_file else defaults["source_file"]
    config = DocumentIndexConfig(
        document_name=args.document_name or defaults["document_name"],
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
    print(f"Built indexes for {manifest['document_name']} with {manifest['embedding_model']}")
    for modality, output in manifest["outputs"].items():
        print(
            f"  {modality}: {output['count']} vectors, dim={output['dimension']}, "
            f"faiss={output['faiss_path']}, metadata={output['metadata_path']}"
        )
    print(f"  manifest: {config.index_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
