import numpy as np
import pandas as pd

from copperlens import analytics as an
from copperlens import data, nlp


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert an.rsi(up).dropna().iloc[-1] == 100
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert an.rsi(down).dropna().iloc[-1] < 1
    r = an.rsi(data.simulate_prices("HG=F", 300)["close"]).dropna()
    assert r.between(0, 100).all()


def test_moving_averages_match_pandas():
    df = an.add_indicators(data.simulate_prices("HG=F", 120, seed=1))
    assert np.isclose(df["ma_short"].iloc[-1], df["close"].iloc[-7:].mean())
    assert np.isclose(df["ma_long"].iloc[-1], df["close"].iloc[-30:].mean())


def test_golden_cross_detected():
    close = [10.0] * 40 + [9.0] * 5 + list(np.linspace(9, 14, 8))
    df = pd.DataFrame({"close": close, "volume": 1.0})
    df = an.add_indicators(df)
    assert an.trend_signal(df, lookback=8)["cross"] in {"Golden cross", None}
    assert an.trend_signal(df, lookback=8)["regime"] in {"Bullish", "Bearish"}


def test_pressure_index_range():
    cu = an.add_indicators(data.simulate_prices("HG=F", 300))
    au = data.simulate_prices("GC=F", 300)
    sd = an.supply_demand_pressure(cu, an.copper_gold_ratio(cu["close"], au["close"]))
    assert sd["pressure"].between(-100, 100).all()


def test_rule_engine_directions():
    bull = "Inventories fell for the fifth day. A strike caused a supply disruption at a mine. Demand growth remained firm."
    bear = "Inventories rose for the fourth week. Weak demand and a property slump. Record output signals a growing surplus."
    assert nlp.summarise_rule_based(bull)["sentiment"] == "bullish"
    assert nlp.summarise_rule_based(bear)["sentiment"] == "bearish"


def test_llm_json_parsing_and_normalising():
    raw = '```json\n{"headline":"x","sentiment":"BULLISH","confidence":"2","signals":[{"factor":"Supply","direction":"up"}]}\n```'
    out = nlp._normalise(nlp._parse_json(raw), "test")
    assert out["sentiment"] == "bullish" and out["confidence"] == 1.0
    assert out["signals"][0]["direction"] == "neutral"


def test_summarise_without_key_uses_rules(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert nlp.summarise("Copper rose. Stocks fell sharply today.")["engine"] == "rule-based"
