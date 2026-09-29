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
        # The account every backtest is run on. It was hard-wired to 10,000
        # and it is not a cosmetic number: fixed-fractional sizing asks for
        # a fraction of a lot, and whether that fraction clears the broker's
        # minimum depends entirely on how big the account is. 1% of 10,000
        # on XAUUSD is about 0.034 lots and loses a seventh of itself to
        # rounding; 1% of 100,000 is 0.34 and loses a seventieth.
        "account_balance": 10_000.0,
        "max_drawdown_pct": 20.0,
        # Prop firms enforce two limits, not one: a total drawdown and a
        # daily one, typically 10% and 5%. A strategy can pass the first
        # and lose the account on the second. 0 disables the check.
        "max_daily_drawdown_pct": 0.0,
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
        # Per-symbol overrides, because one figure cannot serve the whole
        # catalogue: a point is 1/point_divisor, and the divisors differ by
        # a factor of a hundred. Ten points is one pip on EURUSD (divisor
        # 100,000) and one cent on XAUUSD (divisor 1,000) - roughly a
        # seventeenth of gold's real spread. Symbols absent here fall back
        # to `manual_spread_points`.
        "manual_spread_by_symbol": {},
        "commission_per_lot": 7.0,
        "slippage_points": 2.0,
    },
    "broker_caps": {
        # The volume ladder. Fixed-fractional sizing asks for a fraction of
        # a lot and a broker fills a multiple of `lot_step`, never less than
        # `min_lot_size` - so on a small account the risk per trade is not
        # the risk setting, it is whatever the minimum happens to cost. A
        # 10,000 account on XAUUSD at 0.35% asks for around 0.012 lots: every
        # trade fills at 0.01, and a wide stop then risks twice the setting.
        "min_lot_size": 0.01,
        "lot_step": 0.01,
        "max_lot_size": 10.0,
        "max_open_positions": 20,
        "margin_call_pct": 100.0,
        # The broker's server clock. History is cut into days, H4/D1 bars
        # and hours on this clock, because that is what the EA sees in
        # MetaTrader. NY_CLOSE (UTC+2 winter / UTC+3 summer, switching
        # with US daylight saving) is what nearly every retail broker
        # runs; FIXED uses `server_gmt_offset_hours` all year.
        "server_time": "NY_CLOSE",
        "server_gmt_offset_hours": 2.0,
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
    # REQ-NFR-2 — logs are always written; sending anything anywhere is
    # opt-in and off by default.
    "system": {
        "crash_reporting_enabled": False,
    },
    # AUTOMATION (Elite) — tasks.md 12.2/12.6. The scheduler in
    # app/main.py reads these; `last_*_at` are written by it, not by the UI.
    "automation": {
        "recurring_campaigns_enabled": False,
        "recurring_interval_hours": 24,
        "last_campaign_started_at": None,
        "last_purge_at": None,
        # REQ-PORT-3.6 — AUTO off by default: a promotion raises real money
        # at risk, so it starts as something the user does deliberately.
        "auto_graduate_enabled": False,
        # REQ-AUTO-4 — replacing a degraded bot is the most consequential
        # unattended action here, so it needs both a toggle and an explicit
        # first-activation confirmation (4.2) before it will act.
        "auto_deploy_enabled": False,
        "auto_deploy_confirmed": False,
        "last_auto_deploy_at": None,
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
