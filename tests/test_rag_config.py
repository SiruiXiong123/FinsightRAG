from pathlib import Path

import pytest

from finsightrag.rag_config import RagConfig


def test_rag_config_loads_explicit_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
vision_model: demo-model
vision_binding_host: https://example.test/v1
indexing:
  index_root: indexes
""",
        encoding="utf-8",
    )

    config = RagConfig.load(str(config_path))

    assert config.path == config_path.resolve()
    assert config.vision_model == "demo-model"
    assert config.vision_base_url == "https://example.test/v1"
    assert config.values["indexing"]["index_root"] == "indexes"


def test_rag_config_rejects_template_file(tmp_path):
    template = tmp_path / "config.example.yaml"
    template.write_text("vision_model: demo\n", encoding="utf-8")

    with pytest.raises(ValueError, match="template"):
        RagConfig.load(str(template))


def test_rag_config_reads_env_fallback(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("vision_model: null\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")

    config = RagConfig.load(str(config_path))

    assert config.vision_api_key == "from-env"
    assert config.get_path("missing", str(Path("data") / "output")).name == "output"
