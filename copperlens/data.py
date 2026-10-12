"""Market-data ingestion.

Primary source: Yahoo Finance public chart REST endpoint (no API key).
Fallback: a clearly-labelled Geometric Brownian Motion simulator so the
dashboard still runs offline or when the API rate-limits.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import requests

from .config import SIM_BASE, SIM_VOL

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
HEADERS = {"User-Agent": "Mozilla/5.0 (CopperLens research dashboard)"}
OHLCV = ["open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    """Raised when the REST API returns no usable data."""


def fetch_prices(symbol: str, range_: str = "1y", interval: str = "1d",
                 timeout: float = 10.0) -> pd.DataFrame:
    """Download daily OHLCV for ``symbol`` from the REST API."""
    resp = requests.get(
        CHART_URL.format(symbol=symbol),
        params={"range": range_, "interval": interval},
        headers=HEADERS,
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json().get("chart", {})
    if payload.get("error") or not payload.get("result"):
        raise DataError(f"No data returned for {symbol}: {payload.get('error')}")
    result = payload["result"][0]
    quote = result["indicators"]["quote"][0]
    idx = pd.to_datetime(result["timestamp"], unit="s").normalize()
    df = pd.DataFrame({c: quote.get(c) for c in OHLCV}, index=idx)
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"])
    df["volume"] = df["volume"].fillna(0)
    if len(df) < 40:
        raise DataError(f"Too few rows for {symbol}: {len(df)}")
    return df


def simulate_prices(symbol: str, days: int = 252, seed: int | None = None) -> pd.DataFrame:
    """Simulated OHLCV via GBM - offline fallback, NOT real market data."""
    rng = np.random.default_rng(seed if seed is not None else zlib.crc32(symbol.encode()))
    s0 = SIM_BASE.get(symbol, 100.0)
    sigma = SIM_VOL.get(symbol, 0.2)
    mu, dt = 0.05, 1 / 252
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), days)
    close = s0 * np.exp(np.cumsum(shocks))
    open_ = np.concatenate([[s0], close[:-1]])
    spread = np.abs(rng.normal(0, sigma * np.sqrt(dt) * 0.6, days)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.lognormal(mean=11.2, sigma=0.35, size=days).round()
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days, name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


_RANGE_DAYS = {"6mo": 126, "1y": 252, "2y": 504}


def get_prices(symbol: str, range_: str = "1y") -> tuple[pd.DataFrame, str]:
    """Return ``(dataframe, source)`` where source is 'live' or 'simulated'."""
    try:
        return fetch_prices(symbol, range_), "live"
    except Exception:  # network, rate-limit, schema change...
        return simulate_prices(symbol, _RANGE_DAYS.get(range_, 252)), "simulated"
