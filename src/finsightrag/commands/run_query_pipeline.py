import argparse
import json
from pathlib import Path


from finsightrag.paths import default_project_root


PROJECT_ROOT = default_project_root()

from finsightrag.evidence_builder import EvidenceBuilder, write_evidence_package
from finsightrag.generator_vlm import build_config_from_rag_config, generate_from_evidence_package
from finsightrag.rag_config import RagConfig
from finsightrag.retriever import MultiFinRetriever
from finsightrag.vector_store import MultiModalVectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run retrieval, evidence packaging, and final VLM answer generation."
    )
    parser.add_argument("--config-path", type=Path, default=None, help="Optional runtime config.yaml path.")
    parser.add_argument("--document-id", required=True, help="Document ID, usually the PDF filename stem.")
    parser.add_argument("--query", required=True, help="Question to answer.")
    parser.add_argument("--expected-answer", default=None, help="Optional reference answer stored in summary JSON.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for pipeline artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Build prompt but do not call the VLM.")
    parser.add_argument("--include-raw-response", action="store_true", help="Store raw model response and prompt.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config_path = str(args.config_path) if args.config_path else None
    rag_config = RagConfig.load(config_path)
    retrieval_cfg = load_retrieval_config(rag_config)
    output_dir = (args.output_dir or default_output_dir(args.document_id)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    store = MultiModalVectorStore(config_path=config_path, project_root=PROJECT_ROOT)
    retriever = MultiFinRetriever(store, retrieval_cfg)
    retriever_result = retriever.retrieve(query=args.query, document_id=args.document_id)
    retriever_output = output_dir / "retriever_result.json"
    retriever_output.write_text(
        json.dumps(retriever_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    evidence_builder = EvidenceBuilder.from_config_path(
        config_path=config_path,
        project_root=PROJECT_ROOT,
        output_dir=output_dir,
    )
    evidence_package = evidence_builder.build(retriever_result, output_dir=output_dir)
    evidence_output = output_dir / "evidence_package.json"
    write_evidence_package(evidence_package, evidence_output)

    answer_output = output_dir / "answer.json"
    generator_config = build_config_from_rag_config(
        config_path=config_path,
        project_root=PROJECT_ROOT,
        evidence_package_path=evidence_output,
        output_path=answer_output,
        include_raw_response=args.include_raw_response,
        dry_run=args.dry_run,
    )
    answer_payload = generate_from_evidence_package(generator_config)

    summary = {
        "query": args.query,
        "expected_answer": args.expected_answer,
        "generated_answer": answer_payload["answer"],
        "retrieval_mode": retriever_result.get("retrieval_mode"),
        "fallback_triggered": retriever_result.get("fallback_triggered"),
        "trace": retriever_result.get("trace"),
        "top_contexts": summarize_contexts(retriever_result.get("contexts", [])),
        "artifacts": {
            "retriever_result": str(retriever_output),
            "evidence_package": str(evidence_output),
            "answer": str(answer_output),
            "table_montage": artifact_path(evidence_package, "table"),
            "image_montage": artifact_path(evidence_package, "image"),
        },
    }
    summary_output = output_dir / "summary.json"
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_retrieval_config(rag_config: RagConfig) -> dict:
    section = dict(rag_config.values or {}).get("retrieval", {})
    if not isinstance(section, dict):
        raise ValueError("Missing or invalid retrieval section in config.")
    return section


def default_output_dir(document_id: str) -> Path:
    return PROJECT_ROOT / "runs" / "query_pipeline" / safe_name(document_id)


def summarize_contexts(contexts: list[dict], limit: int = 8) -> list[dict]:
    return [
        {
            "modality": context.get("modality"),
            "score": context.get("score"),
            "page": context.get("page"),
            "title": context.get("title"),
            "content_preview": compact(context.get("content"), 280),
        }
        for context in contexts[:limit]
    ]


def artifact_path(evidence_package: dict, kind: str):
    montage = (evidence_package.get("montages") or {}).get(kind)
    if not montage:
        return None
    return montage.get("absolute_path") or montage.get("path")


def compact(value: object, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def safe_name(value: str) -> str:
    chars = [char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value)]
    return "".join(chars).strip("_") or "document"


if __name__ == "__main__":
    raise SystemExit(main())

