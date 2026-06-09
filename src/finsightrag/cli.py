import argparse
import importlib
from collections.abc import Sequence

from . import __version__


COMMANDS = {
    "generate-text-md": "finsightrag.commands.generate_text_md",
    "chunk-text": "finsightrag.commands.chunk_text",
    "extract-assets": "finsightrag.commands.extract_assets",
    "enrich-assets": "finsightrag.commands.enrich_assets",
    "build-indexes": "finsightrag.commands.build_indexes",
    "search-indexes": "finsightrag.commands.search_indexes",
    "build-evidence-package": "finsightrag.commands.build_evidence_package",
    "generate-answer": "finsightrag.commands.generate_answer",
    "run-query-pipeline": "finsightrag.commands.run_query_pipeline",
}


ALIASES = {
    "text-md": "generate-text-md",
    "chunk": "chunk-text",
    "assets": "extract-assets",
    "enrich": "enrich-assets",
    "index": "build-indexes",
    "search": "search-indexes",
    "evidence": "build-evidence-package",
    "answer": "generate-answer",
    "query": "run-query-pipeline",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finsightrag",
        description="FinsightRAG document processing and multimodal QA CLI.",
    )
    parser.add_argument("--version", action="version", version=f"finsightrag {__version__}")
    subcommands = ", ".join(sorted(COMMANDS))
    parser.add_argument("command", nargs="?", help=f"Subcommand to run. Available: {subcommands}")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the selected subcommand.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(argv)
    if not parsed.command:
        parser.print_help()
        return 0

    command = ALIASES.get(parsed.command, parsed.command)
    module_name = COMMANDS.get(command)
    if module_name is None:
        parser.error(f"unknown command: {parsed.command}")

    module = importlib.import_module(module_name)
    return int(module.main(parsed.args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
