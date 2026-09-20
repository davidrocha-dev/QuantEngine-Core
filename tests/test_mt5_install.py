from pathlib import Path

from app.broker import mt5_install
from app.broker.mt5_install import EXPERTS_SUBDIR, install_ea


def make_terminal(tmp_path: Path, name: str) -> Path:
    terminal = tmp_path / name
    (terminal / "MQL5").mkdir(parents=True)
    return terminal


def test_install_writes_the_ea_into_every_terminal(tmp_path, monkeypatch):
    terminals = [make_terminal(tmp_path, "ABC123"), make_terminal(tmp_path, "DEF456")]
    monkeypatch.setattr(mt5_install, "terminal_data_dirs", lambda: terminals)

    result = install_ea("QE_TEST.mq5", "// source", compile_after_install=False)

    assert result.installed
    assert len(result.installed_paths) == 2
    for terminal in terminals:
        installed = terminal / EXPERTS_SUBDIR / "QE_TEST.mq5"
        assert installed.read_text(encoding="utf-8") == "// source"


def test_install_reports_honestly_when_no_terminal_exists(monkeypatch):
    monkeypatch.setattr(mt5_install, "terminal_data_dirs", lambda: [])

    result = install_ea("QE_TEST.mq5", "// source", compile_after_install=False)

    assert not result.installed
    assert "No MetaTrader 5 terminal" in result.detail


def test_install_without_a_compiler_says_the_file_still_needs_compiling(tmp_path, monkeypatch):
    monkeypatch.setattr(mt5_install, "terminal_data_dirs", lambda: [make_terminal(tmp_path, "ABC123")])
    monkeypatch.setattr(mt5_install, "find_metaeditor", lambda: None)

    result = install_ea("QE_TEST.mq5", "// source")

    assert result.installed
    assert result.compiled_paths == []
    assert "MetaEditor" in result.detail


def test_install_overwrites_a_previous_export_of_the_same_strategy(tmp_path, monkeypatch):
    terminal = make_terminal(tmp_path, "ABC123")
    monkeypatch.setattr(mt5_install, "terminal_data_dirs", lambda: [terminal])

    install_ea("QE_TEST.mq5", "// first", compile_after_install=False)
    install_ea("QE_TEST.mq5", "// second", compile_after_install=False)

    assert (terminal / EXPERTS_SUBDIR / "QE_TEST.mq5").read_text(encoding="utf-8") == "// second"
