from types import SimpleNamespace

import pytest

from finsightrag import cli


def test_cli_dispatches_alias(monkeypatch):
    calls = []

    def fake_import_module(module_name):
        calls.append(module_name)
        return SimpleNamespace(main=lambda argv: len(argv))

    monkeypatch.setattr(cli.importlib, "import_module", fake_import_module)

    assert cli.main(["query", "--document-id", "demo"]) == 2
    assert calls == ["finsightrag.commands.run_query_pipeline"]


def test_cli_rejects_unknown_command():
    with pytest.raises(SystemExit):
        cli.main(["unknown-command"])
