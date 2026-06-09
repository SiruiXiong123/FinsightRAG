from pathlib import Path

from finsightrag.evidence_builder import EvidenceBuilder, EvidenceBuilderConfig, MontageConfig


def test_evidence_builder_handles_empty_contexts(tmp_path):
    builder = EvidenceBuilder(
        EvidenceBuilderConfig(
            project_root=Path.cwd(),
            output_dir=tmp_path,
            table_montage=MontageConfig(enabled=True),
            image_montage=MontageConfig(enabled=True),
        )
    )

    package = builder.build(
        {
            "query": "What changed?",
            "document_id": "demo",
            "retrieval_mode": "preview",
            "fallback_triggered": False,
            "contexts": [],
        }
    )

    assert package["contexts_by_modality"] == {"text": [], "table": [], "image": []}
    assert package["montages"] == {"table": None, "image": None}
    assert package["vlm_inputs"]["image_paths"] == []
