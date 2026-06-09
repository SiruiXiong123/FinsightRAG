from finsightrag.parse_enrichment_output import parse_enrichment_output


def test_parse_enrichment_output_tracks_missing_and_duplicates():
    text = """
    [F] table_1.png
    [T] Revenue table
    [M] period: Q1
    [C] Revenue increased.

    [F] table_1.png
    [T] Duplicate
    [M] duplicate
    [C] ignored

    [F] other.png
    [T] Unexpected
    [M] unexpected
    [C] ignored
    """

    result = parse_enrichment_output(
        text,
        expected_filenames=["table_1.png", "missing.png"],
    )

    assert [item.filename for item in result.items] == ["table_1.png"]
    assert result.missing_filenames == ["missing.png"]
    assert result.unexpected_filenames == ["other.png"]
    assert result.duplicate_filenames == ["table_1.png"]


def test_parse_enrichment_output_normalizes_multiline_fields():
    text = """
    [F] chart.png
    [T] Chart title
    [M] line one
        line two
    [C] content line
        second line
    """

    result = parse_enrichment_output(text, expected_filenames=["chart.png"])

    assert len(result.items) == 1
    assert result.items[0].metadata == "line one\n        line two"
    assert result.items[0].content == "content line\n        second line"
