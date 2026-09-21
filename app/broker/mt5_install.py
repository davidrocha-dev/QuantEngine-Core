"""Installing a generated EA into the MetaTrader 5 terminal.

MT5 does not load Expert Advisors from arbitrary paths: they must live in
the terminal's own data directory, which is keyed by an install hash under
`%APPDATA%/MetaQuotes/Terminal/<hash>/MQL5/Experts`. This module finds
those directories, copies the `.mq5` in, and compiles it with MetaEditor
when MetaEditor can be found.

Compilation is best-effort on purpose: a machine may have the terminal but
not the editor (portable installs, restricted setups). The caller reports
exactly what happened — copied and compiled, or copied and awaiting a
manual compile — instead of claiming a deploy that did not occur.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

EXPERTS_SUBDIR = Path("MQL5") / "Experts" / "QuantEngine"

METAEDITOR_CANDIDATES = (
    Path(r"C:\Program Files\MetaTrader 5\metaeditor64.exe"),
    Path(r"C:\Program Files (x86)\MetaTrader 5\metaeditor64.exe"),
)
COMPILE_TIMEOUT_SECONDS = 120


@dataclass
class InstallResult:
    installed_paths: list[str] = field(default_factory=list)
    compiled_paths: list[str] = field(default_factory=list)
    compiler_log: str | None = None
    detail: str = ""

    @property
    def installed(self) -> bool:
        return bool(self.installed_paths)


def terminal_data_dirs() -> list[Path]:
    """Every MT5 terminal data directory on this machine."""
    if sys.platform != "win32":
        return []
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return []
    terminals = Path(appdata) / "MetaQuotes" / "Terminal"
    if not terminals.exists():
        return []
    # "Common" is shared storage, not a terminal; a real terminal has MQL5/.
    return sorted(path for path in terminals.iterdir() if path.is_dir() and (path / "MQL5").is_dir())


def find_metaeditor() -> Path | None:
    for candidate in METAEDITOR_CANDIDATES:
        if candidate.exists():
            return candidate
    override = os.environ.get("QUANTENGINE_METAEDITOR")
    if override and Path(override).exists():
        return Path(override)
    return None


def compile_ea(source_path: Path) -> tuple[bool, str]:
    """Compiles one .mq5 with MetaEditor. Returns (compiled, log)."""
    editor = find_metaeditor()
    if editor is None:
        return False, "MetaEditor was not found on this machine."

    log_path = source_path.with_suffix(".log")
    try:
        subprocess.run(
            [str(editor), f"/compile:{source_path}", f"/log:{log_path}"],
            timeout=COMPILE_TIMEOUT_SECONDS,
            check=False,
            capture_output=True,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, f"MetaEditor could not be run: {exc}"

    compiled = source_path.with_suffix(".ex5")
    log = ""
    if log_path.exists():
        # MetaEditor writes its log as UTF-16.
        log = log_path.read_text(encoding="utf-16", errors="replace")
    return compiled.exists(), log


def uninstall_ea(filename: str) -> list[str]:
    """Removes a generated EA from every terminal, returning what it took.

    Only the .mq5 and its compiled .ex5 are touched. A chart still running
    the EA keeps running until MT5 is told otherwise — removing the file
    stops it being loaded again, it does not reach into a live chart.
    """
    removed: list[str] = []
    for terminal in terminal_data_dirs():
        for path in (terminal / EXPERTS_SUBDIR / filename, (terminal / EXPERTS_SUBDIR / filename).with_suffix(".ex5")):
            if path.exists():
                path.unlink()
                removed.append(str(path))
    return removed


def install_ea(filename: str, source: str, compile_after_install: bool = True) -> InstallResult:
    """Writes the EA into every MT5 terminal found and compiles it."""
    targets = terminal_data_dirs()
    if not targets:
        return InstallResult(detail="No MetaTrader 5 terminal data directory was found on this machine.")

    result = InstallResult()
    for terminal in targets:
        experts_dir = terminal / EXPERTS_SUBDIR
        experts_dir.mkdir(parents=True, exist_ok=True)
        destination = experts_dir / filename
        destination.write_text(source, encoding="utf-8")
        result.installed_paths.append(str(destination))

        if compile_after_install:
            compiled, log = compile_ea(destination)
            result.compiler_log = log or result.compiler_log
            if compiled:
                result.compiled_paths.append(str(destination.with_suffix(".ex5")))

    if result.compiled_paths:
        result.detail = f"Installed and compiled into {len(result.compiled_paths)} terminal(s). Attach it from the Navigator."
    else:
        result.detail = (
            f"Installed into {len(result.installed_paths)} terminal(s). MetaEditor did not compile it "
            "(not found or reported errors) — open the file in MetaEditor and press F7."
        )
    return result


def install_compiled_ea(filename: str, content: bytes, expected_sha256: str | None = None) -> InstallResult:
    """Installs an already-compiled `.ex5` — REQ-MKT-1.3.

    A Marketplace subscription hands over the compiled EA and never the
    source, so there is nothing to compile here and MetaEditor is not
    involved. The checksum is verified before anything is written: this
    binary arrived over the network and is about to be given permission
    to trade, which is not a moment to skip a two-line check.
    """
    if not filename.lower().endswith(".ex5"):
        return InstallResult(detail="Only compiled .ex5 files can be installed this way.")
    if expected_sha256:
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected_sha256:
            return InstallResult(detail="Checksum mismatch - the download was corrupted or tampered with.")

    targets = terminal_data_dirs()
    if not targets:
        return InstallResult(detail="No MetaTrader 5 terminal data directory was found on this machine.")

    result = InstallResult()
    for terminal in targets:
        experts_dir = terminal / EXPERTS_SUBDIR
        experts_dir.mkdir(parents=True, exist_ok=True)
        destination = experts_dir / filename
        destination.write_bytes(content)
        result.installed_paths.append(str(destination))
        # It is already compiled: from MT5's point of view it is ready to
        # attach to a chart as soon as the terminal rescans Experts.
        result.compiled_paths.append(str(destination))

    result.detail = f"Installed into {len(result.installed_paths)} terminal(s)."
    return result
