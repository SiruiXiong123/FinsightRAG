import re
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


FIELD_BLOCK_PATTERN = re.compile(
    r"(?ms)^\s*\[F\]\s*(?P<filename>.*?)\s*"
    r"^\s*\[T\]\s*(?P<title>.*?)\s*"
    r"^\s*\[M\]\s*(?P<metadata>.*?)\s*"
    r"^\s*\[C\]\s*(?P<content>.*?)(?=^\s*\[F\]|\Z)"
)


@dataclass(frozen=True)
class EnrichmentItem:
    filename: str
    title: str
    metadata: str
    content: str


@dataclass(frozen=True)
class EnrichmentParseResult:
    items: list[EnrichmentItem]
    missing_filenames: list[str]
    unexpected_filenames: list[str]
    duplicate_filenames: list[str]


def parse_enrichment_output(
    text: str,
    expected_filenames: Optional[list[str]] = None,
) -> EnrichmentParseResult:
    expected_filenames = expected_filenames or []
    expected_set = set(expected_filenames)
    seen = set()
    duplicate_filenames = []
    unexpected_filenames = []
    items = []

    for match in FIELD_BLOCK_PATTERN.finditer(text or ""):
        item = EnrichmentItem(
            filename=normalize_field(match.group("filename")),
            title=normalize_field(match.group("title")),
            metadata=normalize_field(match.group("metadata")),
            content=normalize_field(match.group("content")),
        )
        if not item.filename:
            continue
        if expected_set and item.filename not in expected_set:
            unexpected_filenames.append(item.filename)
            continue
        if item.filename in seen:
            duplicate_filenames.append(item.filename)
            continue
        seen.add(item.filename)
        items.append(item)

    missing_filenames = [
        filename
        for filename in expected_filenames
        if filename not in seen
    ]
    return EnrichmentParseResult(
        items=items,
        missing_filenames=missing_filenames,
        unexpected_filenames=sorted(set(unexpected_filenames)),
        duplicate_filenames=sorted(set(duplicate_filenames)),
    )


def normalize_field(value: str) -> str:
    return "\n".join(
        line.rstrip()
        for line in str(value or "").strip().splitlines()
    ).strip()


def enrichment_item_to_payload(item: EnrichmentItem) -> dict:
    return {
        "filename": item.filename,
        "title": item.title,
        "metadata": item.metadata,
        "content": item.content,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse model enrichment output in [F][T][M][C] format."
    )
    parser.add_argument("--input-file", type=Path, required=True, help="Raw model output text file.")
    parser.add_argument("--output-file", type=Path, default=None, help="Optional parsed JSON output path.")
    parser.add_argument(
        "--expected-filename",
        action="append",
        dest="expected_filenames",
        help="Expected filename. Can be provided more than once.",
    )
    parser.add_argument(
        "--expected-file",
        type=Path,
        default=None,
        help="Text file containing one expected filename per line.",
    )
    args = parser.parse_args(argv)
    expected_filenames = list(args.expected_filenames or [])
    if args.expected_file:
        expected_filenames.extend(
            line.strip()
            for line in args.expected_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    result = parse_enrichment_output(
        text=args.input_file.read_text(encoding="utf-8"),
        expected_filenames=expected_filenames,
    )
    payload = {
        "items": [enrichment_item_to_payload(item) for item in result.items],
        "missing_filenames": result.missing_filenames,
        "unexpected_filenames": result.unexpected_filenames,
        "duplicate_filenames": result.duplicate_filenames,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0 if not result.missing_filenames else 1


if __name__ == "__main__":
    raise SystemExit(main())
