# QuantEngine — Strategy Discovery Engine

Python backend that searches for forex trading strategies with a genetic algorithm, validates them out-of-sample and exports the survivors as MetaTrader 5 Expert Advisors.

> **Status: work in progress.** This repository is the open-source core (backtesting and search engine). The desktop app, licensing and marketplace live in a separate private project.

## What it does

- **Genetic search** over a catalog of 46 entry/exit genes (`app/genome`), evolving populations of rule-based strategies.
- **Vectorised backtester** (pandas, NumPy, Numba) with closed-bar evaluation, ATR-based stops/targets and risk-based position sizing (`app/engine/backtest.py`).
- **Multi-phase validation pipeline**: walk-forward analysis, Monte Carlo resampling, tick-level verification and trade-parity checks against MT5 reports to reduce overfitting (`app/engine`).
- **Crash-resumable campaigns** checkpointed to SQLite after every generation.
- **Market data pipeline**: downloads Dukascopy tick history, resamples to OHLC and caches it as Parquet (`app/data`).
- **Code generation**: turns a stored genome into an MQL5 Expert Advisor with Jinja2 templates (`app/codegen`).
- **Local REST/WebSocket API** with FastAPI for driving campaigns and streaming progress.

## Stack

Python · FastAPI · SQLAlchemy/SQLite · pandas · NumPy · Numba · scikit-learn · Jinja2 · pytest

## Running

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m app.data.build_bundled_subset   # downloads the sample market data
pytest
python run_engine.py
```

The MetaTrader 5 bridge (`app/broker`) only works on Windows with MT5 installed; everything else runs cross-platform.

## Disclaimer

For research and education only. Backtested results do not guarantee future performance. Not financial advice.
