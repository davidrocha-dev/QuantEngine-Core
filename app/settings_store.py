"""Settings screen sections 6.1, 6.3, 6.5, 6.6, 6.9, 6.10 — REQ-SET-1, 4, 7.

Everything here is local-only config (design.md B.3: "Settings, gene pool
config, broker caps | Local (config file)") — a single JSON blob deep-merged
over these defaults on load, so adding a field later doesn't break an
existing saved file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_APP_SETTINGS: dict[str, Any] = {
    "target_markets": {
        "enabled_symbols": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"],
        "enabled_timeframes": ["M15", "H1"],
        "broker_suffix": "",
    },
    "risk": {
        "max_drawdown_pct": 20.0,
        "per_trade_risk_pct": 1.0,
        "max_leverage": 30.0,
    },
    "campaign": {
        "name": "",
        "power": "MEDIUM",  # ECO | MEDIUM | HIGH | EXTREME
        "cores": 0,  # 0 = auto (all detected cores)
        "smart_mode": False,
        "history_depth_years": 3,
        # REQ-BREED-2 — default OFF (2.2); AUTO derives the context TF from
        # the trading timeframe, MANUAL lets the user pick one explicitly.
        "higher_tf_enabled": False,
        "higher_tf_mode": "AUTO",  # AUTO | MANUAL
        "higher_tf_manual_timeframe": None,
    },
    "costs": {
        "spread_mode": "AUTO",  # AUTO | MANUAL
        "manual_spread_points": 10.0,
        "commission_per_lot": 7.0,
        "slippage_points": 2.0,
    },
    "broker_caps": {
        "max_lot_size": 10.0,
        "max_open_positions": 20,
        "margin_call_pct": 100.0,
    },
    "vault_maintenance": {
        "protect_portfolio": True,
        "purge_criteria": {
            "dd_gt_return": {"enabled": False, "threshold": None},
            "built_in_data_bred": {"enabled": False, "threshold": None},
            "min_trades": {"enabled": False, "threshold": 30},
            "min_pf": {"enabled": False, "threshold": 1.2},
            "min_ret_pct": {"enabled": False, "threshold": 10.0},
            "min_win_rate_pct": {"enabled": False, "threshold": 40.0},
            "min_sharpe": {"enabled": False, "threshold": 0.5},
            "min_persistence_pct": {"enabled": False, "threshold": 50.0},
            "min_score": {"enabled": False, "threshold": 57.0},
            "min_robustness_pct": {"enabled": False, "threshold": 60.0},
            "min_ret_per_trade_oos": {"enabled": False, "threshold": None},
            "min_ret_per_trade_is": {"enabled": False, "threshold": None},
            "max_consecutive_losses": {"enabled": False, "threshold": 8},
            "max_age_days": {"enabled": False, "threshold": 180},
            "min_icir": {"enabled": False, "threshold": None},
            "min_avg_hold_hours": {"enabled": False, "threshold": None},
        },
        "automatic_purge_enabled": False,
        "automatic_purge_interval_days": 7,
    },
    "notifications": {
        "os_enabled": True,
        "telegram_enabled": False,
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "events": {
            "campaign_complete": {"os": True, "telegram": False},
            "health_check_triggered": {"os": True, "telegram": False},
            "bot_degradation": {"os": True, "telegram": True},
        },
    },
    "smart_mode": {
        "smart_insights_enabled": True,
    },
    # AUTOMATION (Elite) — tasks.md 12.2/12.6. The scheduler in
    # app/main.py reads these; `last_*_at` are written by it, not by the UI.
    "automation": {
        "recurring_campaigns_enabled": False,
        "recurring_interval_hours": 24,
        "last_campaign_started_at": None,
        "last_purge_at": None,
    },
}


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return json.loads(json.dumps(DEFAULT_APP_SETTINGS))
    saved = json.loads(path.read_text())
    return _deep_merge(DEFAULT_APP_SETTINGS, saved)


def save_settings(path: Path, settings: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2))


def update_section(path: Path, section: str, patch: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings(path)
    if section not in settings:
        raise KeyError(f"Unknown settings section: {section}")
    if isinstance(settings[section], dict) and isinstance(patch, dict):
        settings[section] = _deep_merge(settings[section], patch)
    else:
        settings[section] = patch
    save_settings(path, settings)
    return settings[section]
