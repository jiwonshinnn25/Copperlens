"""Technical + cross-market analytics (pure pandas, no I/O)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MA_LONG, MA_SHORT, RSI_PERIOD


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0)


def add_indicators(df: pd.DataFrame, short: int = MA_SHORT, long: int = MA_LONG) -> pd.DataFrame:
    out = df.copy()
    out["ma_short"] = out["close"].rolling(short).mean()
    out["ma_long"] = out["close"].rolling(long).mean()
    out["ret"] = out["close"].pct_change()
    out["vol_30d"] = out["ret"].rolling(30).std() * np.sqrt(252)  # annualised
    out["rsi"] = rsi(out["close"])
    out["vol_ratio"] = out["volume"] / out["volume"].rolling(30).mean()
    return out


def trend_signal(df: pd.DataFrame, lookback: int = 5) -> dict:
    """Classify the MA regime and flag a fresh golden/death cross."""
    d = df.dropna(subset=["ma_short", "ma_long"])
    if len(d) < lookback + 1:
        return {"regime": "Insufficient data", "cross": None, "spread_pct": 0.0}
    spread = d["ma_short"] - d["ma_long"]
    regime = "Bullish" if spread.iloc[-1] > 0 else "Bearish"
    cross = None
    recent = np.sign(spread.iloc[-(lookback + 1):])
    if recent.iloc[0] < 0 < recent.iloc[-1]:
        cross = "Golden cross"
    elif recent.iloc[0] > 0 > recent.iloc[-1]:
        cross = "Death cross"
    return {"regime": regime, "cross": cross,
            "spread_pct": float(spread.iloc[-1] / d["ma_long"].iloc[-1] * 100)}


def rsi_state(value: float) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return "Overbought" if value >= 70 else "Oversold" if value <= 30 else "Neutral"


def copper_gold_ratio(copper: pd.Series, gold: pd.Series) -> pd.Series:
    """Copper (USD/lb) per 1,000 USD of gold (USD/oz) - a classic growth proxy.

    Rising ratio => market pricing stronger industrial demand / risk appetite.
    """
    both = pd.concat([copper, gold], axis=1, keys=["cu", "au"]).dropna()
    return (both["cu"] / both["au"] * 1000).rename("cu_au_ratio")


def _z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0


def supply_demand_pressure(cu: pd.DataFrame, ratio: pd.Series) -> pd.DataFrame:
    """Market-implied supply/demand pressure index in [-100, +100].

    Built from observable proxies (no physical-inventory feed):
      * 30d price momentum              (50%)
      * 30d change in Cu/Au ratio       (30%)
      * volume surge signed by 5d trend (20%)
    Positive => demand-pull / tightening supply; negative => surplus / weak demand.
    """
    df = pd.DataFrame(index=cu.index)
    df["momentum"] = cu["close"].pct_change(30)
    df["ratio_chg"] = ratio.reindex(cu.index).ffill().pct_change(30)
    direction = np.sign(cu["close"].pct_change(5))
    df["volume_conf"] = (cu["volume"] / cu["volume"].rolling(30).mean() - 1) * direction
    comp = 0.5 * _z(df["momentum"]) + 0.3 * _z(df["ratio_chg"]) + 0.2 * _z(df["volume_conf"])
    df["pressure"] = np.tanh(comp / 2) * 100
    return df.dropna(subset=["pressure"])


def pressure_label(x: float) -> str:
    if x >= 40:
        return "Strong demand pull"
    if x >= 10:
        return "Mild demand pull"
    if x > -10:
        return "Balanced"
    if x > -40:
        return "Mild surplus pressure"
    return "Surplus / weak demand"


def rebase(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Rebase several price series to 100 at the first common date."""
    joined = pd.concat(series_map, axis=1).dropna()
    return joined / joined.iloc[0] * 100
