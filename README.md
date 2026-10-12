# CopperLens — Commodity Market Analytics Dashboard

A Streamlit dashboard that ingests live metal prices over a REST API and computes technical and
cross-market indicators, including moving averages, RSI, volatility and a market-implied
supply-demand pressure index.

**Stack:** Python · Streamlit · Pandas · REST APIs · Plotly

**Live demo:** _https://copperlens-test.streamlit.app/_

## Features
| Tab | What it does |
|---|---|
| **Market** | Candlestick chart with **7/30-day moving averages**, RSI(14), volume, 30-day annualised volatility, trend regime and golden/death-cross alerts. CSV export. |
| **Supply–Demand** | Copper/Gold ratio (growth proxy), a composite **supply–demand pressure index** (−100…+100), and cross-metal performance rebased to 100. |

## Architecture
```
Yahoo Finance REST API ──► data.py ──► analytics.py ──► app.py (Plotly charts)
        (GBM fallback)
```
* **Graceful degradation:** if the REST API is unreachable, the app switches to a clearly labelled
  simulated (GBM) series and shows a warning.
* **Caching:** `st.cache_data` with a 5-minute TTL for prices.

## Data source
Yahoo Finance public chart REST endpoint, `GET /v8/finance/chart/{symbol}?range=1y&interval=1d`
(no API key), parsed with `requests` into a Pandas DataFrame. Prices are COMEX/NYMEX front-month
futures (e.g. `HG=F`). The endpoint is unofficial and delayed, so it is suitable for analysis demos,
not for trading.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                     # or requirements.txt for just the app
streamlit run app.py
pytest -q                                               # unit tests
```

## Methodology notes
* **Supply–demand pressure index** is market-implied: 50% 30-day momentum, 30% 30-day change in the
  Cu/Au ratio, 20% volume surge signed by the 5-day trend (z-scored, squashed with tanh). It does not
  use physical inventory or mine-output data.

## Project layout
```
app.py                    Streamlit UI
copperlens/data.py        REST ingestion + GBM fallback
copperlens/analytics.py   MA, RSI, volatility, Cu/Au ratio, pressure index
copperlens/nlp.py         Report-summarisation module (not wired into the deployed app)
tests/test_core.py        pytest suite
```
