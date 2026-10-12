"""CopperLens - commodity market analytics console (Streamlit)."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from copperlens import analytics as an
from copperlens import data
from copperlens.config import METALS, RANGES

try:
    from copperlens import nlp
except Exception:  # pypdf missing etc. -> tab shows a clear error
    nlp = None

COPPER, BLUE = "#C87533", "#6FA8DC"
GREEN, AMBER, RED, GREY = "#5FB380", "#D9A05B", "#D4666A", "#8D867A"
COL = {"ok": GREEN, "warn": AMBER, "bad": RED, "mut": GREY}

st.set_page_config(page_title="CopperLens", layout="wide")
st.markdown("""<style>
.stApp{background:#0f0e0c;color:#ece6d8}
.block-container{padding-top:1rem;max-width:1500px}
.lbl{font:600 .68rem ui-monospace,Menlo,monospace;letter-spacing:.16em;text-transform:uppercase;color:#8d867a}
.card{background:#171613;border:1px solid #2b2924;border-radius:14px;padding:14px 18px;height:100%}
.big{font-size:1.9rem;font-weight:600;margin:.25rem 0;color:#f3eddf}
.sub{font:.7rem ui-monospace,Menlo,monospace;color:#8d867a;letter-spacing:.08em;text-transform:uppercase}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block;float:right;margin-top:3px}
.pill{font:600 .68rem ui-monospace,Menlo,monospace;letter-spacing:.1em;border:1px solid;border-radius:999px;padding:2px 10px;white-space:nowrap}
.top{display:flex;justify-content:space-between;align-items:center;padding:6px 2px 14px}
.top .r{display:flex;gap:10px;align-items:center}
.ev{display:flex;justify-content:space-between;align-items:center;background:#171613;border:1px solid #2b2924;border-radius:12px;padding:9px 12px;margin-bottom:8px}
.ev b{font:600 .85rem ui-monospace,Menlo,monospace}
.ev small{display:block;color:#8d867a;font-size:.72rem;margin-top:2px}
.flow{display:flex;gap:8px;align-items:stretch;flex-wrap:wrap}
.stage{flex:1;min-width:150px;background:#171613;border:1px solid #3b6b52;border-radius:12px;padding:12px}
.stage.fb{border-color:#8a6a3a;border-style:dashed}
.stage h4{margin:.2rem 0 .4rem;font-size:.95rem;color:#f3eddf}
.stage p{margin:.4rem 0 0;font-size:.74rem;color:#a39c8e}
.arrow{align-self:center;color:#5a564d}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def pill(text: str, kind: str = "mut") -> str:
    c = COL[kind]
    return f'<span class="pill" style="color:{c};border-color:{c}66;background:{c}14">{text}</span>'


def card(label: str, value: str, sub: str = "", kind: str | None = None) -> str:
    dot = f'<span class="dot" style="background:{COL[kind]}"></span>' if kind else ""
    return (f'<div class="card"><div class="lbl">{label}{dot}</div>'
            f'<div class="big">{value}</div><div class="sub">{sub}</div></div>')


def spark(s: pd.Series, color: str) -> str:
    v = s.tail(40).values
    lo, hi = v.min(), v.max()
    pts = " ".join(f"{i * 100 / (len(v) - 1):.1f},{27 - (x - lo) / (hi - lo + 1e-9) * 25:.1f}" for i, x in enumerate(v))
    return (f'<svg viewBox="0 0 100 28" width="100%" height="28" preserveAspectRatio="none">'
            f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{pts}"/></svg>')


def panel(fig: go.Figure, title: str, height: int = 300) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=12)), showlegend=False)
    style(fig, height)
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10))
    return fig


def style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="ui-monospace,Menlo,monospace", size=11, color="#b9b2a3"),
                      hovermode="x unified", legend=dict(orientation="h", y=1.04))
    fig.update_xaxes(gridcolor="#24221e")
    fig.update_yaxes(gridcolor="#24221e")
    return fig


@st.cache_data(ttl=300, show_spinner=False)
def load(symbol: str, rng: str) -> tuple[pd.DataFrame, str]:
    return data.get_prices(symbol, rng)


def price_figure(df, name, unit, short, long) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"],
                                 close=df["close"], name=name,
                                 increasing_line_color=GREEN, decreasing_line_color=RED), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_short"], name=f"{short}-day MA",
                             line=dict(color=COPPER, width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_long"], name=f"{long}-day MA",
                             line=dict(color=BLUE, width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI(14)",
                             line=dict(color="#BB8FCE", width=1.5)), row=2, col=1)
    for lvl, col in ((70, RED), (30, GREEN)):
        fig.add_hline(y=lvl, line=dict(color=col, dash="dot", width=1), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume",
                         marker_color=GREY, opacity=0.6), row=3, col=1)
    fig.update_yaxes(title_text=unit, row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="Vol", row=3, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    return style(fig, 640)


def build_signals(df, sig, pressure: float) -> list[dict]:
    """Rule layer: turn indicators into graded evidence (ok / warn / bad)."""
    out = []
    reg = str(sig["regime"])
    up = any(k in reg.lower() for k in ("up", "bull"))
    dn = any(k in reg.lower() for k in ("down", "bear"))
    out.append(dict(name="Trend regime", state=reg, kind="ok" if up else "warn" if dn else "mut",
                    note=f"{sig['spread_pct']:+.2f}% MA spread"))
    if sig["cross"]:
        out.append(dict(name="MA cross", state=str(sig["cross"]),
                        kind="bad" if "death" in str(sig["cross"]).lower() else "ok", note="last 5 sessions"))
    r = float(df["rsi"].iloc[-1])
    out.append(dict(name="RSI(14)", state=an.rsi_state(r), note=f"{r:.0f}",
                    kind="bad" if r >= 75 or r <= 25 else "warn" if r >= 70 or r <= 30 else "ok"))
    pct = float(df["vol_30d"].rank(pct=True).iloc[-1])
    out.append(dict(name="30d volatility", state=f"{pct * 100:.0f}th pctile", note=f"{df['vol_30d'].iloc[-1] * 100:.1f}% ann.",
                    kind="bad" if pct >= 0.9 else "warn" if pct >= 0.75 else "ok"))
    vr = float(df["vol_ratio"].iloc[-1])
    out.append(dict(name="Volume surge", state=f"{vr:.2f}x avg", note="vs 30d average",
                    kind="bad" if vr >= 2 else "warn" if vr >= 1.5 else "ok"))
    out.append(dict(name="S/D pressure", state=an.pressure_label(pressure), note=f"{pressure:+.0f} / 100",
                    kind="bad" if abs(pressure) >= 60 else "warn" if abs(pressure) >= 30 else "ok"))
    return out


# ---------------------------------------------------------------- sidebar
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

# ------------------------------------------------------------------ data
symbol, unit = METALS[metal]
rng = RANGES[rng_label]
t0 = time.perf_counter()
raw, source = load(symbol, rng)
cu_raw, cu_src = load("HG=F", rng)
au_raw, au_src = load("GC=F", rng)
fetch_ms = (time.perf_counter() - t0) * 1000
closes = {label: load(sym, rng)[0]["close"] for label, (sym, _) in METALS.items()}
df = an.add_indicators(raw, short, long)
cu = an.add_indicators(cu_raw)
ratio = an.copper_gold_ratio(cu["close"], au_raw["close"])
sd = an.supply_demand_pressure(cu, ratio)
now = float(sd["pressure"].iloc[-1])
sig = an.trend_signal(df)
signals = build_signals(df, sig, now)
n_bad = sum(s["kind"] == "bad" for s in signals)
n_warn = sum(s["kind"] == "warn" for s in signals)
risk = (n_bad + 0.5 * n_warn) / len(signals)
live = "simulated" not in (source, cu_src, au_src)

# ---------------------------------------------------------------- header
st.markdown(
    f'<div class="top"><div class="lbl" style="color:#ece6d8;font-size:.8rem">'
    f'<span style="color:{GREEN if live else AMBER}">&#9679;</span>&nbsp; COPPERLENS</div>'
    f'<div class="r">{pill("SNAPSHOT " + f"{df.index[-1]:%d %b %Y}")}'
    f'{pill("LIVE REST" if live else "SIMULATED GBM", "ok" if live else "warn")}'
    f'{pill("NLP RULE-BASED")}</div></div>',
    unsafe_allow_html=True)

last, prev = df["close"].iloc[-1], df["close"].iloc[-2]
k1, k2, k3, k4 = st.columns(4)
k1.markdown(card(f"{metal} · {symbol}", f"{last:,.2f}", f"{(last / prev - 1) * 100:+.2f}% day · {unit}",
                 "ok" if last >= prev else "bad"), unsafe_allow_html=True)
k2.markdown(card("Trend regime", str(sig["regime"]), f"{sig['spread_pct']:+.2f}% MA spread"), unsafe_allow_html=True)
k3.markdown(card("S/D pressure", f"{now:+.0f}", an.pressure_label(now),
                 "bad" if abs(now) >= 60 else "warn" if abs(now) >= 30 else "ok"), unsafe_allow_html=True)
k4.markdown(card("Active alerts", str(n_bad + n_warn), f"{n_bad} critical · {n_warn} watch · risk {risk:.2f}",
                 "bad" if n_bad else "warn" if n_warn else "ok"), unsafe_allow_html=True)
st.markdown('<div class="lbl" style="margin:14px 0 6px">Market watch</div>', unsafe_allow_html=True)
for col, (label, cl) in zip(st.columns(len(closes)), closes.items()):
    chg = cl.iloc[-1] / cl.iloc[-2] - 1
    kc = GREEN if chg >= 0 else RED
    col.markdown(f'<div class="card" style="padding:10px 14px"><div class="lbl">{label}'
                 f'<span class="dot" style="background:{kc}"></span></div>'
                 f'<div style="font-size:1.1rem;font-weight:600">{cl.iloc[-1]:,.2f} '
                 f'<span style="color:{kc};font-size:.78rem">{chg * 100:+.2f}%</span></div>'
                 f'{spark(cl, kc)}</div>', unsafe_allow_html=True)
if not live:
    st.warning("Live API unreachable - showing **simulated** (GBM) prices for demonstration only.")
st.write("")

tab_ops, tab_sd, tab_risk, tab_alerts, tab_pipe, tab_ai = st.tabs(
    ["Operations", "Supply–Demand", "Risk & Stats", "Alerts", "Pipeline", "Report Analyst"])

# ------------------------------------------------------------ operations
with tab_ops:
    inc_col, left, right = st.columns([1, 3.2, 1])
    with inc_col:
        st.markdown('<div class="lbl">Active incidents</div>', unsafe_allow_html=True)
        inc = [x for x in signals if x["kind"] in ("warn", "bad")]
        for x in inc:
            st.markdown(f'<div class="ev"><div>{pill("CRITICAL" if x["kind"] == "bad" else "WATCH", x["kind"])}'
                        f'<div style="margin-top:6px"><b>{x["name"]}</b></div>'
                        f'<small>{x["state"]} · {x["note"]}</small>'
                        f'<small>{df.index[-1]:%d %b %Y} · ACTIVE</small></div></div>', unsafe_allow_html=True)
        if not inc:
            st.markdown(f'<div class="ev"><div>{pill("ALL CLEAR", "ok")}<small>No indicator is outside its '
                        f'normal range.</small></div></div>', unsafe_allow_html=True)
    with left:
        st.plotly_chart(price_figure(df, metal, unit, short, long), width='stretch')
        with st.expander("Raw data"):
            st.dataframe(df.round(4).tail(60).sort_index(ascending=False), width='stretch')
            st.download_button("Download CSV", df.to_csv().encode(), f"{symbol}_{rng}.csv", "text/csv")
    with right:
        st.markdown('<div class="lbl">Live model evidence</div>', unsafe_allow_html=True)
        for s in signals:
            st.markdown(f'<div class="ev"><div><b>{s["name"]}</b><small>{s["note"]}</small></div>'
                        f'{pill(s["state"], s["kind"])}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------- supply-demand
with tab_sd:
    st.caption("Market-implied indicators from price, volume and cross-metal ratios "
               "(no physical-inventory feed). A screening signal, not a forecast.")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Copper/Gold ratio (growth proxy)", "Supply–demand pressure index"))
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio, name="Cu/Au", line=dict(color=COPPER)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio.rolling(30).mean(), name="30d MA",
                             line=dict(color=BLUE, dash="dot")), row=1, col=1)
    fig.add_trace(go.Bar(x=sd.index, y=sd["pressure"], name="Pressure",
                         marker_color=[GREEN if v >= 0 else RED for v in sd["pressure"]]), row=2, col=1)
    fig.update_yaxes(range=[-100, 100], row=2, col=1)
    fig.update_layout(showlegend=False)
    st.plotly_chart(style(fig, 520), width='stretch')

    st.markdown('<div class="lbl">Cross-metal performance (rebased to 100)</div>', unsafe_allow_html=True)
    reb = an.rebase(closes)
    fig2 = go.Figure([go.Scatter(x=reb.index, y=reb[m], name=m,
                                 line=dict(width=3 if m == "Copper" else 1.5,
                                           color=COPPER if m == "Copper" else None)) for m in reb.columns])
    st.plotly_chart(style(fig2, 380), width='stretch')

# -------------------------------------------------------------- risk & stats
with tab_risk:
    ret = df["close"].pct_change().dropna()
    dd = df["close"] / df["close"].cummax() - 1
    ann_ret = (1 + ret).prod() ** (252 / len(ret)) - 1
    ann_vol = ret.std() * 252 ** 0.5
    var95 = ret.quantile(0.05)
    stats = [("Ann. return", f"{ann_ret * 100:+.1f}%", "geometric"),
             ("Ann. volatility", f"{ann_vol * 100:.1f}%", "daily σ × √252"),
             ("Return / risk", f"{ann_ret / ann_vol:.2f}", "risk-free = 0"),
             ("Max drawdown", f"{dd.min() * 100:.1f}%", f"now {dd.iloc[-1] * 100:.1f}%"),
             ("VaR 95% (1d)", f"{var95 * 100:.2f}%", "historical"),
             ("Skew / kurtosis", f"{ret.skew():.2f} / {ret.kurt():.1f}", "daily returns")]
    for col, (l, v, sub_) in zip(st.columns(6), stats):
        col.markdown(card(l, v, sub_), unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    f1 = go.Figure(go.Scatter(x=dd.index, y=dd * 100, fill="tozeroy", line=dict(color=RED, width=1.2)))
    a.plotly_chart(panel(f1, f"{metal} · drawdown from peak (%)"), width='stretch')
    f2 = go.Figure(go.Histogram(x=ret * 100, nbinsx=40, marker_color=COPPER))
    f2.add_vline(x=var95 * 100, line=dict(color=RED, dash="dot"))
    b.plotly_chart(panel(f2, "Daily return distribution (%) · dotted line = VaR 95%"), width='stretch')
    c, d = st.columns(2)
    f3 = go.Figure(go.Scatter(x=df.index, y=df["vol_30d"] * 100, line=dict(color=BLUE, width=1.6)))
    c.plotly_chart(panel(f3, "Rolling 30d annualised volatility (%)"), width='stretch')
    corr = pd.DataFrame(closes).pct_change().dropna().corr()
    f4 = go.Figure(go.Heatmap(z=corr.values, x=list(corr.columns), y=list(corr.columns), zmin=-1, zmax=1,
                              colorscale="RdBu", text=corr.round(2).values, texttemplate="%{text}"))
    d.plotly_chart(panel(f4, "Cross-metal return correlation"), width='stretch')

# ---------------------------------------------------------------- alerts
with tab_alerts:
    rows = [{"Indicator": s["name"], "State": s["state"], "Reading": s["note"],
             "Level": {"ok": "Stable", "warn": "Watch", "bad": "Critical", "mut": "Info"}[s["kind"]]}
            for s in signals]
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    with st.expander("How the alert layer works"):
        st.markdown(
            "- **RSI(14):** watch at 70/30, critical at 75/25.\n"
            "- **Volatility:** percentile of 30d annualised vol within the selected history "
            "(watch ≥ 75th, critical ≥ 90th).\n"
            "- **Volume surge:** latest volume / 30d average (watch ≥ 1.5x, critical ≥ 2x).\n"
            "- **S/D pressure:** composite of 50% 30d momentum, 30% 30d Cu/Au change, 20% signed volume "
            "surge, z-scored and tanh-squashed to −100…+100 (watch |x| ≥ 30, critical ≥ 60).\n"
            "- **Risk score:** (critical + 0.5 × watch) / number of indicators.")

# -------------------------------------------------------------- pipeline
with tab_pipe:
    st.markdown('<div class="lbl">Data and model pipeline</div>', unsafe_allow_html=True)
    stages = [
        ("Ingest", "Yahoo REST", "LIVE" if live else "FALLBACK", "ok" if live else "warn",
         f"HG=F · GC=F · {metal}; {len(raw)} bars; {fetch_ms:.0f} ms; 5-min cache"),
        ("Features", "analytics.py", "OK", "ok", "MA · RSI(14) · 30d vol · volume ratio · Cu/Au ratio"),
        ("Scoring", "S/D pressure", "OK", "ok", "z-scored momentum, ratio change and volume surge → tanh"),
        ("Rules", "Alert layer", "OK", "ok", f"{len(signals)} indicators graded; {n_bad + n_warn} active"),
        ("NLP", "nlp.py", "RULES", "ok", "PDF → text → lexicon + regex → structured brief (offline)"),
        ("Serve", "Streamlit", "OK", "ok", "Plotly charts · CSV export"),
    ]
    html = '<div class="flow">'
    for i, (tag, name, status, kind, desc) in enumerate(stages):
        html += (f'<div class="stage{" fb" if kind == "warn" else ""}"><div class="lbl">{tag}</div>'
                 f'<h4>{name}</h4>{pill(status, kind)}<p>{desc}</p></div>')
        if i < len(stages) - 1:
            html += '<div class="arrow">&#10140;</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)
    st.write("")
    st.caption("Failure handling: API unreachable → labelled GBM simulation.")

# ------------------------------------------------------------ report analyst
SAMPLE = ("Copper rallied as inventories fell for a fourth session and a strike at a major Chilean mine "
          "threatened supply. Data centre and grid investment continued to support demand, while a weaker "
          "dollar added support. Analysts see resistance near $4.80/lb and support at $4.40/lb. "
          "Tariff uncertainty remains a risk.")
SAMPLE_DIR = Path(__file__).parent / "sample_reports"

with tab_ai:
    st.markdown('<div class="lbl">Tier 3 · text analysis layer</div>', unsafe_allow_html=True)
    st.subheader("Market-report analyst")
    if nlp is None:
        st.error("NLP module failed to import. Add `pypdf` to requirements.txt.")
    else:
        samples = {p.stem: p for p in sorted(SAMPLE_DIR.glob("*.pdf"))} if SAMPLE_DIR.exists() else {}
        c1, c2 = st.columns([1, 1])
        choice = c1.selectbox("Sample report", ["(none)"] + list(samples))
        up_file = c1.file_uploader("…or upload a PDF", type="pdf")
        txt = c2.text_area("…or paste report text", SAMPLE, height=215)
        if st.button("Analyse report", type="primary"):
            try:
                if up_file:
                    text = nlp.extract_text(up_file.getvalue())
                elif choice != "(none)":
                    text = nlp.extract_text(str(samples[choice]))
                else:
                    text = txt
                t1 = time.perf_counter()
                st.session_state["brief"] = (nlp.summarise_rule_based(text), (time.perf_counter() - t1) * 1000)
            except Exception as exc:
                st.error(f"Could not analyse report: {exc}")
        if "brief" in st.session_state:
            res, ms = st.session_state["brief"]
            sk = {"bullish": "ok", "bearish": "bad"}.get(res["sentiment"], "mut")
            st.markdown(f"### {res['headline']}")
            conf, ms_txt = f"CONF {res['confidence']:.2f}", f"{ms:.0f} MS"
            st.markdown(f"{pill(res['sentiment'].upper(), sk)} &nbsp; {pill(conf)} &nbsp; "
                        f"{pill(res['engine'].upper())} &nbsp; {pill(ms_txt)}", unsafe_allow_html=True)
            for bullet in res["summary"]:
                st.markdown(f"- {bullet}")
            if res["signals"]:
                st.dataframe(pd.DataFrame(res["signals"]).rename(columns=str.title), width='stretch', hide_index=True)
            if res["price_levels"]:
                st.markdown("**Quoted levels:** " + " · ".join(res["price_levels"]))
            agree = (res["sentiment"] == "bullish" and now > 0) or (res["sentiment"] == "bearish" and now < 0)
            st.info(f"Cross-check: report sentiment is **{res['sentiment']}**, S/D pressure is **{now:+.0f}** → "
                    + ("aligned." if agree else "not aligned; treat the divergence as a flag."))

st.markdown(f'<div class="lbl" style="margin-top:2rem;border-top:1px solid #2b2924;padding-top:10px">'
            f'Source Yahoo Finance REST · cache TTL 5 min · {len(raw)} bars · fetch {fetch_ms:.0f} ms · '
            f'analysis only, not financial advice</div>', unsafe_allow_html=True)
