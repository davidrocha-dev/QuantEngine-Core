"""Engine-wide runtime configuration.

``built_in_data_only`` is the debug toggle from REQ-DATA-1.6 / Settings →
System → Debug: when set, the data layer forces use of the pre-bundled
Dukascopy subset regardless of full cache or broker availability. It is a
runtime flag (not a code path deleted between debug/release) so QA can
flip it without a rebuild — but it must never be on by default.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_bundled_data_dir() -> Path:
    # In dev, the source-relative path is correct. In a packaged build the
    # engine runs as a PyInstaller-frozen exe, so `__file__` resolves inside
    # its _MEIPASS extraction dir — which doesn't contain this data, since
    # it ships via electron-builder's extraResources instead. Electron's
    # engineProcess.ts sets this env var to the real on-disk path in that
    # case (see REQ-DATA-1.4).
    env_override = os.environ.get("QUANTENGINE_BUNDLED_DIR")
    if env_override:
        return Path(env_override)
    return Path(__file__).resolve().parent / "data" / "bundled"


@dataclass
class EngineConfig:
    data_dir: Path = field(default_factory=lambda: Path.home() / ".quantengine" / "data")
    bundled_data_dir: Path = field(default_factory=_default_bundled_data_dir)
    built_in_data_only: bool = False
    # REQ-SET-5.3 — default ON: creates an automatic backup before any
    # schema migration. Read by app/migrations/runner.py via main.py's
    # startup hook.
    auto_backup_before_updates: bool = True

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def manifest_path(self) -> Path:
        return self.cache_dir / "sync_manifest.json"

    @property
    def vault_db_path(self) -> Path:
        return self.data_dir / "vault.db"

    @property
    def settings_path(self) -> Path:
        return self.data_dir / "settings.json"

    @property
    def gene_pool_path(self) -> Path:
        return self.data_dir / "gene_pool.json"


_config = EngineConfig()


def get_config() -> EngineConfig:
    return _config


def set_built_in_data_only(enabled: bool) -> None:
    _config.built_in_data_only = enabled


def set_auto_backup_before_updates(enabled: bool) -> None:
    _config.auto_backup_before_updates = enabled
