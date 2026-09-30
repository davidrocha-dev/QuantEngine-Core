"""Settings -> CampaignConfig: everything the user configures reaches the engine through here."""


import pandas as pd

from app.api.costs import CONTRACT_SIZE_BY_CATEGORY, _spread_points, _swap_spec
from app.config import get_config
from app.data.conversion import quote_to_account_series
from app.data.provider import get_historical_bars
from app.data.symbols import TIMEFRAMES, get_instrument
from app.engine import smart_mode
from app.engine.campaign import (
    CampaignConfig,
)
from app.engine.news_filter import load_calendar
from app.engine.vault_writer import HIGHER_TF_OFF
from app.genome.gene_pool import GenePoolConfig

# POWER (Settings → Campaign) controls compute intensity — REQ-SET-3/design.md
# A.9. The full 4,000/20-generation run is EXTREME; lighter presets keep
# ECO/MEDIUM campaigns fast enough to actually watch complete.
POWER_PRESETS = {
    "ECO": {"population_size": 60, "generations_per_tribe": 3},
    "MEDIUM": {"population_size": 300, "generations_per_tribe": 5},
    "HIGH": {"population_size": 1000, "generations_per_tribe": 10},
    "EXTREME": {"population_size": 4000, "generations_per_tribe": 20},
}


# REQ-BREED-2.1 — the higher-timeframe context each trading timeframe reads.
# REQ-BREED-2 — the timeframe a HIGHER TF filter reads its context from.
# M1 and M5 were missing, so the toggle on the Breed screen read "no
# higher context available" for the two timeframes people reach for
# first. Roughly one order of magnitude up in each case: enough context
# to be worth consulting, close enough to still be about the same trade.
HIGHER_TF_BY_TIMEFRAME = {
    "M1": "M15",
    "M5": "H1",
    "M15": "H1",
    "M30": "H1",
    "H1": "H4",
    "H2": "D1",
    "H4": "D1",
}


def _gene_pool_for_campaign(config: CampaignConfig) -> GenePoolConfig:
    """The enabled genes, minus any the campaign cannot actually use.

    EVENT is the one case today. With no economic calendar on disk it
    vetoes nothing, so a genome that draws it backtests exactly like one
    that does not and scores the same - and then cannot be exported at
    all, because the EA has no calendar feed and the generator refuses to
    emit a strategy that would trade without a filter its backtest had.
    Measured: 7.8% of randomly drawn genomes take EVENT, so roughly one
    candidate in thirteen was being bred towards a Vault row that no
    amount of work could deploy.

    Disabling it changes no backtest result - an empty calendar already
    vetoes nothing - it only stops the search spending itself on
    strategies that cannot leave the building.
    """
    gene_pool = GenePoolConfig.load(get_config().gene_pool_path)
    if not config.news_events:
        gene_pool.set_enabled("filter", "EVENT", False)
    # The same reasoning, twice more. HTF with HIGHER TF off is neutral on
    # every bar; CONSEC has never been applied by the backtester at all.
    # Either one only spends the search on a gene that changes nothing.
    if config.htf_df is None:
        gene_pool.set_enabled("bias", "HTF", False)
    gene_pool.set_enabled("filter", "CONSEC", False)
    return gene_pool


def _strategy_higher_timeframe(settings: dict, strategy) -> str | None:
    """The HIGHER TF context a Vault strategy is traded and re-checked with.

    What it was bred with, when that was recorded; for older rows the
    current settings, which is what REASSESS has always used for them.
    """
    recorded = getattr(strategy, "higher_timeframe", None)
    if recorded == HIGHER_TF_OFF:
        return None
    if recorded:
        return recorded
    campaign_settings = settings["campaign"]
    if not campaign_settings["higher_tf_enabled"]:
        return None
    return _higher_timeframe(strategy.timeframe, campaign_settings)


def _config_for_strategy(settings: dict, strategy) -> CampaignConfig:
    """build_campaign_config, with the strategy's own HIGHER TF context."""
    config = build_campaign_config(settings, strategy.asset, strategy.timeframe)
    wanted = _strategy_higher_timeframe(settings, strategy)
    if wanted != config.higher_timeframe:
        config.higher_timeframe = wanted
        config.htf_df = (
            _load_bars(strategy.asset, wanted, settings["campaign"]["history_depth_years"]) if wanted else None
        )
        if config.htf_df is not None and config.htf_df.empty:
            config.htf_df = None
    return config


def _learned_gene_weights() -> dict[str, float]:
    """Smart Mode's weights, tempered by what keeps failing screening.

    Two signals, and they answer different questions. Smart Mode asks
    which genes appear in strategies that made it into the Vault; the
    failure counters ask which genes keep being drawn and keep not
    surviving. A gene can be absent from the Vault because nobody tried
    it or because it has been tried thousands of times and never works,
    and only the second signal can tell those apart.

    Both live under the Smart Mode toggle. Someone who turned learning
    off meant all of it.
    """
    from app.engine.failure_signal import load as load_gene_outcomes
    from app.engine.failure_signal import penalties

    weights = dict(smart_mode.load(get_config().data_dir).weights)
    for key, penalty in penalties(load_gene_outcomes(get_config().data_dir)).items():
        weights[key] = weights.get(key, 1.0) * penalty
    return weights


def _higher_timeframe(timeframe: str, campaign_settings: dict) -> str | None:
    """The timeframe a HIGHER TF filter reads its context from.

    A manual choice is only honoured when it is actually higher than the
    one being traded. The screen offers nothing else, but the value is
    stored in settings.json and survives a change of trading timeframe -
    breed on H1 with a manual H4 context, drop to M15, and the stored H4
    is still above it; breed on M15 with H1 and move up to H4 and it is
    not. Loading a *lower* timeframe as "higher context" would not fail,
    it would quietly resample the wrong series into every gene that
    consults it, so an out-of-order value falls back to the mapping.
    """
    manual = campaign_settings.get("higher_tf_manual_timeframe")
    automatic = HIGHER_TF_BY_TIMEFRAME.get(timeframe)
    if campaign_settings.get("higher_tf_mode") != "MANUAL" or not manual:
        return automatic
    if timeframe not in TIMEFRAMES or manual not in TIMEFRAMES:
        return automatic
    return manual if TIMEFRAMES.index(manual) > TIMEFRAMES.index(timeframe) else automatic


def build_campaign_config(settings: dict, symbol: str, timeframe: str) -> CampaignConfig:
    """Translates the Settings screen into engine parameters. Everything the
    user configures — risk, costs, cores, POWER, HIGHER TF — has to reach the
    engine here, or the Settings screen is decoration."""
    campaign_settings = settings["campaign"]
    costs = settings["costs"]
    caps = settings.get("broker_caps", {})
    preset = POWER_PRESETS.get(campaign_settings["power"], POWER_PRESETS["MEDIUM"])
    instrument = get_instrument(symbol)

    htf_df = None
    htf_timeframe = None
    if campaign_settings["higher_tf_enabled"]:
        htf_timeframe = _higher_timeframe(timeframe, campaign_settings)
        if htf_timeframe:
            htf_df = _load_bars(symbol, htf_timeframe, campaign_settings["history_depth_years"])
            if htf_df.empty:
                htf_df, htf_timeframe = None, None

    # REQ-ENGINE-2 — the EVENT gene needs a calendar; without one it would
    # silently veto nothing, so an absent calendar simply means no events.
    news_events = load_calendar(
        get_config().data_dir / "calendar.json", get_config().data_dir / "calendar.csv"
    )

    return CampaignConfig(
        population_size=preset["population_size"],
        generations_per_tribe=preset["generations_per_tribe"],
        timeframe=timeframe,
        initial_balance=float(settings["risk"].get("account_balance", 10_000.0)),
        # Settings → Risk (REQ-SET-1.2) is a percentage; the backtester takes a fraction.
        risk_pct=settings["risk"]["per_trade_risk_pct"] / 100,
        max_drawdown_pct=float(settings["risk"]["max_drawdown_pct"]),
        max_daily_drawdown_pct=float(settings["risk"].get("max_daily_drawdown_pct", 0.0)),
        # Costs are in the instrument's own points (0.00001 on EURUSD,
        # 0.001 on USDJPY/XAUUSD), not a fixed pip.
        point_size=1 / instrument.point_divisor,
        cost_points=_spread_points(costs, symbol, instrument) + float(costs["slippage_points"]),
        commission_per_lot=float(costs["commission_per_lot"]),
        contract_size=CONTRACT_SIZE_BY_CATEGORY.get(instrument.category, 100_000.0),
        # Settings → Broker caps. Without these the Vault reports a return
        # no small account can reach: fixed-fractional sizing asks for a
        # fraction of a lot and the broker fills whole steps of one.
        lot_step=float(caps.get("lot_step", 0.0)),
        min_lot=float(caps.get("min_lot_size", 0.0)),
        max_lot=float(caps.get("max_lot_size", 0.0)),
        quote_to_account=quote_to_account_series(instrument, get_historical_bars),
        swap=_swap_spec(costs, symbol, instrument),
        max_workers=campaign_settings["cores"] or None,  # 0 = auto
        data_source="bundled" if get_config().built_in_data_only else "dukascopy",
        # SMART MODE (Settings → Campaign) biases which genes get drawn,
        # using what the Vault has already shown to work.
        gene_weights=_learned_gene_weights() if campaign_settings["smart_mode"] else {},
        htf_df=htf_df,
        higher_timeframe=htf_timeframe,
        news_events=news_events,
    )


def _load_bars(symbol: str, timeframe: str, history_depth_years: int):
    """Historical bars trimmed to the configured HISTORY DEPTH — breeding on
    more history than the user asked for silently multiplies campaign time."""
    bars = get_historical_bars(symbol, timeframe)
    if bars.empty or not history_depth_years:
        return bars
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365 * history_depth_years)
    trimmed = bars[pd.DatetimeIndex(bars["timestamp"]) >= cutoff]
    return trimmed.reset_index(drop=True) if not trimmed.empty else bars
