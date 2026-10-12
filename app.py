"""CopperLens - commodity market analytics dashboard (Streamlit)."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from copperlens import analytics as an
from copperlens import data, nlp
from copperlens.config import METALS, RANGES

ROOT = Path(__file__).parent
COPPER = "#C87533"
GREEN, RED, GREY = "#2ECC71", "#E74C3C", "#8A93A6"

st.set_page_config(page_title="CopperLens", layout="wide")

# Streamlit Cloud: expose secrets (ANTHROPIC_API_KEY, CL_MODEL) as environment variables.
try:
    for _k in ("ANTHROPIC_API_KEY", "CL_MODEL"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets configured -> rule-based engine

st.markdown(
    "<style>.block-container{padding-top:1.4rem}"
    "[data-testid=stMetricValue]{font-size:1.5rem}</style>",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------- caching
@st.cache_data(ttl=300, show_spinner=False)
def load(symbol: str, rng: str) -> tuple[pd.DataFrame, str]:
    return data.get_prices(symbol, rng)


@st.cache_data(show_spinner=False)
def summarise_cached(key: str, text: str, use_llm: bool) -> dict:
    return nlp.summarise(text, use_llm=use_llm)


# -------------------------------------------------------------- charts
def price_figure(df: pd.DataFrame, name: str, unit: str, short: int, long: int) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"],
                                 close=df["close"], name=name,
                                 increasing_line_color=GREEN, decreasing_line_color=RED), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_short"], name=f"{short}-day MA",
                             line=dict(color=COPPER, width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_long"], name=f"{long}-day MA",
                             line=dict(color="#5DADE2", width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI(14)",
                             line=dict(color="#BB8FCE", width=1.5)), row=2, col=1)
    for lvl, col in ((70, RED), (30, GREEN)):
        fig.add_hline(y=lvl, line=dict(color=col, dash="dot", width=1), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume",
                         marker_color=GREY, opacity=0.6), row=3, col=1)
    fig.update_yaxes(title_text=unit, row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="Vol", row=3, col=1)
    fig.update_layout(template="plotly_dark", height=640, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.03),
                      hovermode="x unified")
    return fig


def badge(sentiment: str) -> str:
    colour = {"bullish": "green", "bearish": "red"}.get(sentiment, "gray")
    return f":{colour}[**{sentiment.upper()}**]"


# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("CopperLens")
    st.caption("Commodity market analytics")
    metal = st.selectbox("Metal", list(METALS), index=0)
    rng_label = st.radio("History", list(RANGES), index=1, horizontal=True)
    short = st.slider("Short MA (days)", 3, 20, 7)
    long = st.slider("Long MA (days)", 20, 100, 30)
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption(f"NLP engine: **{'LLM API' if nlp.llm_available() else 'rule-based (no API key)'}**")

symbol, unit = METALS[metal]
rng = RANGES[rng_label]
raw, source = load(symbol, rng)
df = an.add_indicators(raw, short, long)

st.title(f"{metal} — Market Analytics")
if source == "live":
    st.caption(f"Live data · {symbol} · REST API · last bar {df.index[-1]:%d %b %Y}")
else:
    st.warning("Live API unreachable - showing **simulated** (GBM) prices for demonstration only.")

tab_mkt, tab_sd, tab_rep = st.tabs(["Market", "Supply–Demand", "Report Desk"])

# ----------------------------------------------------------- market tab
with tab_mkt:
    last, prev = df["close"].iloc[-1], df["close"].iloc[-2]
    sig = an.trend_signal(df)
    rsi_v = df["rsi"].iloc[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Last price", f"{last:,.2f}", f"{(last / prev - 1) * 100:+.2f}%")
    c2.metric(f"{short}d MA", f"{df['ma_short'].iloc[-1]:,.2f}")
    c3.metric(f"{long}d MA", f"{df['ma_long'].iloc[-1]:,.2f}", f"{sig['spread_pct']:+.2f}% spread")
    c4.metric("RSI(14)", f"{rsi_v:.0f}", an.rsi_state(rsi_v), delta_color="off")
    c5.metric("30d volatility", f"{df['vol_30d'].iloc[-1] * 100:.1f}%")

    msg = f"**Trend regime:** {sig['regime']}"
    if sig["cross"]:
        msg += f" · **{sig['cross']}** in the last 5 sessions"
    st.info(msg)
    st.plotly_chart(price_figure(df, metal, unit, short, long), width='stretch')
    with st.expander("Raw data"):
        st.dataframe(df.round(4).tail(60).sort_index(ascending=False), width='stretch')
        st.download_button("Download CSV", df.to_csv().encode(), f"{symbol}_{rng}.csv", "text/csv")

# ------------------------------------------------------ supply-demand tab
with tab_sd:
    st.caption("Market-implied indicators built from price, volume and cross-metal ratios "
               "(no physical-inventory feed). Treat as a screening signal, not a forecast.")
    cu_raw, cu_src = load("HG=F", rng)
    au_raw, au_src = load("GC=F", rng)
    cu = an.add_indicators(cu_raw)
    ratio = an.copper_gold_ratio(cu["close"], au_raw["close"])
    sd = an.supply_demand_pressure(cu, ratio)
    now = float(sd["pressure"].iloc[-1])

    a, b, c = st.columns(3)
    a.metric("S/D pressure index", f"{now:+.0f}", an.pressure_label(now), delta_color="off")
    b.metric("Cu/Au ratio", f"{ratio.iloc[-1]:.3f}",
             f"{(ratio.iloc[-1] / ratio.iloc[-31] - 1) * 100:+.1f}% 30d")
    c.metric("Volume vs 30d avg", f"{cu['vol_ratio'].iloc[-1]:.2f}×")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Copper/Gold ratio (growth proxy)", "Supply–demand pressure index"))
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio, name="Cu/Au", line=dict(color=COPPER)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio.rolling(30).mean(), name="30d MA",
                             line=dict(color="#5DADE2", dash="dot")), row=1, col=1)
    fig.add_trace(go.Bar(x=sd.index, y=sd["pressure"], name="Pressure",
                         marker_color=[GREEN if v >= 0 else RED for v in sd["pressure"]]), row=2, col=1)
    fig.update_yaxes(range=[-100, 100], row=2, col=1)
    fig.update_layout(template="plotly_dark", height=520, margin=dict(l=10, r=10, t=40, b=10),
                      showlegend=False, hovermode="x unified")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Cross-metal performance (rebased to 100)")
    series = {}
    for label, (sym, _) in METALS.items():
        series[label] = load(sym, rng)[0]["close"]
    reb = an.rebase(series)
    fig2 = go.Figure([go.Scatter(x=reb.index, y=reb[m], name=m,
                                 line=dict(width=3 if m == "Copper" else 1.5,
                                           color=COPPER if m == "Copper" else None)) for m in reb.columns])
    fig2.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=10, b=10),
                       hovermode="x unified", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, width='stretch')
    if "simulated" in (cu_src, au_src):
        st.caption("Note: some series are simulated because the live API was unreachable.")

# ------------------------------------------------------------ report desk
with tab_rep:
    st.caption("Upload daily market-report PDFs. The NLP pipeline (PDF → text → LLM/rule engine) "
               "returns a structured brief for a mock trading desk.")
    use_llm = st.toggle("Use LLM summariser", value=nlp.llm_available(), disabled=not nlp.llm_available(),
                        help="Set ANTHROPIC_API_KEY to enable. Falls back to rule-based NLP automatically.")
    uploads = st.file_uploader("Market-report PDFs", type="pdf", accept_multiple_files=True)
    samples = sorted((ROOT / "sample_reports").glob("*.pdf"))
    chosen = st.multiselect("…or use bundled synthetic samples", [p.name for p in samples],
                            default=[] if uploads else [p.name for p in samples[:1]])

    docs: list[tuple[str, bytes]] = [(u.name, u.getvalue()) for u in uploads]
    docs += [(p.name, p.read_bytes()) for p in samples if p.name in chosen]

    # Guardrails for a public deployment: cap size/count and LLM calls per session.
    MAX_MB, MAX_DOCS, MAX_LLM = 5, 6, 3
    if len(docs) > MAX_DOCS:
        st.warning(f"Only the first {MAX_DOCS} documents are processed.")
    briefs = []
    for i, (name, blob) in enumerate(docs[:MAX_DOCS]):
        if len(blob) > MAX_MB * 1024 * 1024:
            st.warning(f"{name} skipped (over {MAX_MB} MB).")
            continue
        text = nlp.extract_text(blob)
        key = hashlib.sha1(blob).hexdigest()
        briefs.append((name, summarise_cached(key, text, use_llm and i < MAX_LLM)))

    if not briefs:
        st.info("Add a PDF to generate a brief.")
    else:
        tally = pd.Series([b["sentiment"] for _, b in briefs]).value_counts()
        st.subheader("Desk sentiment")
        cols = st.columns(3)
        for col, s in zip(cols, ("bullish", "neutral", "bearish")):
            col.metric(s.capitalize(), int(tally.get(s, 0)))
        for name, b in briefs:
            with st.container(border=True):
                st.markdown(f"#### {b['headline'] or name}")
                st.caption(f"{name} · engine: {b['engine']}")
                if b.get("note"):
                    st.warning(b["note"])
                left, right = st.columns([3, 2])
                with left:
                    for line in b["summary"]:
                        st.markdown(f"- {line}")
                with right:
                    st.markdown(f"**Sentiment:** {badge(b['sentiment'])}")
                    st.progress(b["confidence"], text=f"Confidence {b['confidence']:.0%}")
                    if b["price_levels"]:
                        st.markdown("**Levels to watch**")
                        for lv in b["price_levels"]:
                            st.markdown(f"- {lv}")
                if b["signals"]:
                    sig_df = pd.DataFrame(b["signals"]).rename(columns=str.title)
                    sig_df["Direction"] = sig_df["Direction"].map(
                        {"bullish": "Bullish", "bearish": "Bearish", "neutral": "Neutral"})
                    st.dataframe(sig_df, hide_index=True, width='stretch')
