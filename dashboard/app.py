import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

DATA  = Path(__file__).parent.parent / "data" / "processed"
RAW   = Path(__file__).parent.parent / "data" / "raw"
LOGO  = Path(__file__).parent.parent / "assets" / "StarvestLogo.png"

st.set_page_config(
    page_title="Starvest",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  /* ── Base ── */
  [data-testid="stAppViewContainer"] { background: #f8fafc; }
  [data-testid="stHeader"]           { background: #f8fafc; }
  [data-testid="stSidebar"]          { background: #f1f5f9; }
  .block-container { padding: 1.5rem 2.5rem 2rem; max-width: 1400px; }

  /* ── KPI cards ── */
  .kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    height: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  .kpi-label { color: #64748b; font-size: 0.72rem; text-transform: uppercase; letter-spacing: .08em; margin-bottom: .3rem; }
  .kpi-value { color: #0f172a; font-size: 1.6rem; font-weight: 700; line-height: 1.1; }
  .kpi-sub   { color: #94a3b8; font-size: 0.75rem; margin-top: .3rem; }
  .kpi-bullish { color: #16a34a; font-size: 1.4rem; font-weight: 700; }
  .kpi-bearish { color: #dc2626; font-size: 1.4rem; font-weight: 700; }
  .kpi-neutral { color: #d97706; font-size: 1.4rem; font-weight: 700; }
  .kpi-trend-pos { color: #16a34a; font-size: 1.4rem; font-weight: 700; }
  .kpi-trend-neg { color: #dc2626; font-size: 1.4rem; font-weight: 700; }

  /* ── Section headers ── */
  .section-title {
    color: #64748b;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-bottom: .5rem;
    margin-top: .2rem;
  }

  /* ── Panel cards ── */
  .panel-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  .panel-title {
    color: #f97316;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    font-weight: 600;
    margin-bottom: 1rem;
  }

  /* ── Outlook panel ── */
  .outlook-metric { display: flex; justify-content: space-between; align-items: center; padding: .5rem 0; border-bottom: 1px solid #e2e8f0; }
  .outlook-key    { color: #64748b; font-size: .82rem; }
  .outlook-val    { color: #0f172a; font-size: .9rem; font-weight: 600; }

  /* ── Model perf table ── */
  .perf-row { display: flex; justify-content: space-between; align-items: center; padding: .45rem 0; border-bottom: 1px solid #e2e8f0; }
  .perf-key { color: #64748b; font-size: .8rem; }
  .perf-val { color: #2563eb; font-size: .85rem; font-weight: 600; }

  /* ── Data sources ── */
  .ds-item { color: #64748b; font-size: .8rem; padding: .4rem 0; border-bottom: 1px solid #e2e8f0; display: flex; align-items: center; gap: .5rem; }
  .ds-dot   { width: 7px; height: 7px; border-radius: 50%; background: #f97316; flex-shrink: 0; }

  /* ── Divider ── */
  hr { border-color: #e2e8f0 !important; margin: 1.2rem 0 !important; }

  /* ── Tooltip ── */
  .tooltip-wrap { position: relative; display: inline-block; cursor: help; }
  .tooltip-wrap .tooltip-text {
    visibility: hidden; opacity: 0;
    background: #1e293b; color: #f1f5f9;
    font-size: .72rem; line-height: 1.4;
    border: 1px solid #334155; border-radius: 8px;
    padding: .6rem .8rem; width: 220px;
    position: absolute; z-index: 999;
    bottom: 130%; left: 50%; transform: translateX(-50%);
    transition: opacity .15s ease;
    pointer-events: none;
  }
  .tooltip-wrap:hover .tooltip-text { visibility: visible; opacity: 1; }

  /* ── Hide streamlit chrome ── */
  #MainMenu { visibility: hidden; }
  footer     { visibility: hidden; }
  [data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    dataset = pd.read_csv(DATA / "dataset.csv")
    backtest = pd.read_csv(DATA / "backtest_results.csv")
    with open(DATA / "model_params.json") as f:
        params = json.load(f)
    return dataset, backtest, params

@st.cache_data
def load_ndvi_raw():
    return pd.read_csv(RAW / "ndvi_raw.csv", parse_dates=["date"])

try:
    dataset, backtest, params = load_data()
    ndvi_raw = load_ndvi_raw()
except FileNotFoundError:
    st.error("Run the pipeline first: build_dataset → model → backtest")
    st.stop()

dataset = dataset.sort_values("year").reset_index(drop=True)
latest  = dataset.iloc[-1]
prev    = dataset.iloc[-2] if len(dataset) > 1 else latest

# ── Derived metrics ─────────────────────────────────────────────────────────
ndvi_trend_pct = (latest["mean_ndvi"] - prev["mean_ndvi"]) / prev["mean_ndvi"] * 100
accuracy       = (backtest["actual_pressure"] == backtest["predicted_pressure"]).mean()
pressure       = latest["price_pressure"]
now            = datetime.now()
harvest_month  = "Nov" if now.month <= 10 else "Nov"
harvest_year   = now.year if now.month <= 6 else now.year + 1

CHART_BG    = "#ffffff"
GRID_COLOR  = "#e2e8f0"
FONT_COLOR  = "#475569"

def chart_layout(fig, height=300):
    fig.update_layout(
        height=height,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color=FONT_COLOR, size=11),
        margin=dict(t=10, b=30, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color=FONT_COLOR)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155", font=dict(color="#f1f5f9", size=11)),
        xaxis=dict(
            gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR),
            showspikes=True, spikecolor="#2563eb", spikethickness=1, spikedash="dot", spikemode="across",
        ),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR)),
    )
    return fig

# ── HEADER ───────────────────────────────────────────────────────────────────
h_col1, h_col2 = st.columns([3, 1])
with h_col1:
    st.image(str(LOGO), width=460)
    st.markdown('<p style="color:#64748b;font-size:.82rem;margin-top:-.5rem;">Satellite-Powered Citrus Commodity Forecasting</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8;font-size:.75rem;">NASA MODIS NDVI &nbsp;|&nbsp; USDA Yield Data &nbsp;|&nbsp; OJ Futures</p>', unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ── KPI ROW ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")

pressure_class = {"bullish": "kpi-bullish", "bearish": "kpi-bearish", "neutral": "kpi-neutral"}[pressure]
trend_class    = "kpi-trend-pos" if ndvi_trend_pct >= 0 else "kpi-trend-neg"
trend_arrow    = "▲" if ndvi_trend_pct >= 0 else "▼"

k1.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">NDVI Current Average</div>
  <div class="kpi-value">{latest['mean_ndvi']:.4f}</div>
  <div class="kpi-sub">Season composite</div>
</div>""", unsafe_allow_html=True)

k2.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Predicted Yield</div>
  <div class="kpi-value">{latest['yield_boxes']/1e6:.2f}M</div>
  <div class="kpi-sub">Boxes · {int(latest['year'])}</div>
</div>""", unsafe_allow_html=True)

PRESSURE_TOOLTIPS = {
    "bullish": "Bullish = upward price pressure expected due to lower-than-average citrus supply. OJ futures likely to rise.",
    "bearish": "Bearish = downward price pressure expected due to higher-than-average citrus supply. OJ futures likely to fall.",
    "neutral": "Neutral = yield near historical average. No significant price pressure signal detected.",
}
k3.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Market Pressure
    <span class="tooltip-wrap"> ℹ️
      <span class="tooltip-text">{PRESSURE_TOOLTIPS[pressure]}</span>
    </span>
  </div>
  <div class="{pressure_class}">{pressure.capitalize()}</div>
  <div class="kpi-sub">OJ futures signal</div>
</div>""", unsafe_allow_html=True)

k4.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Model Accuracy</div>
  <div class="kpi-value">{accuracy:.0%}</div>
  <div class="kpi-sub">Backtest · price pressure</div>
</div>""", unsafe_allow_html=True)

k5.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Trend Signal</div>
  <div class="{trend_class}">{trend_arrow} {abs(ndvi_trend_pct):.1f}%</div>
  <div class="kpi-sub">vs. last season</div>
</div>""", unsafe_allow_html=True)

k6.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">Est. Harvest</div>
  <div class="kpi-value">{harvest_month} {harvest_year}</div>
  <div class="kpi-sub">Season start</div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

# ── ROW 1: Map + NDVI Trend ───────────────────────────────────────────────────
c_map, c_ndvi = st.columns(2, gap="medium")

with c_map:
    st.markdown('<p class="section-title">Florida Citrus Belt</p>', unsafe_allow_html=True)
    ndvi_val = float(latest["mean_ndvi"])

    # Draw citrus belt polygon on a dark map
    lats = [27.0, 28.2, 28.2, 27.0, 27.0]
    lons = [-82.0, -82.0, -81.0, -81.0, -82.0]

    fig_map = go.Figure()
    fig_map.add_trace(go.Scattermapbox(
        lat=lats, lon=lons, mode="lines",
        fill="toself",
        fillcolor=f"rgba(34,197,94,{min(ndvi_val * 0.6, 0.35)})",
        line=dict(color="#22c55e", width=2),
        name="Citrus Belt",
        hovertemplate=f"NDVI: {ndvi_val:.4f}<extra></extra>",
    ))
    fig_map.add_trace(go.Scattermapbox(
        lat=[27.6], lon=[-81.5], mode="markers+text",
        marker=dict(size=14, color="#f97316"),
        text=[f"NDVI {ndvi_val:.3f}"], textposition="top right",
        textfont=dict(color="#ffffff", size=12),
        hovertemplate=f"Citrus Belt<br>NDVI: {ndvi_val:.4f}<extra></extra>",
    ))
    fig_map.update_layout(
        mapbox=dict(
            style="white-bg",
            center=dict(lat=27.8, lon=-81.5),
            zoom=5.5,
            layers=[{
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "ESRI World Imagery",
                "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
            }],
        ),
        height=300, margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="#ffffff", showlegend=False,
    )
    st.plotly_chart(fig_map, use_container_width=True)

with c_ndvi:
    st.markdown('<p class="section-title">NDVI Trend — Florida Citrus Belt</p>', unsafe_allow_html=True)
    fig_ndvi = go.Figure()
    fig_ndvi.add_trace(go.Scatter(
        x=ndvi_raw["date"], y=ndvi_raw["mean_ndvi"],
        mode="lines", name="NDVI",
        line=dict(color="#22c55e", width=1.5),
        fill="tozeroy", fillcolor="rgba(34,197,94,0.08)",
        hovertemplate="<b>%{x|%b %Y}</b><br>NDVI: %{y:.4f}<extra></extra>",
    ))
    chart_layout(fig_ndvi)
    fig_ndvi.update_layout(yaxis_title="Mean NDVI", xaxis_title="")
    st.plotly_chart(fig_ndvi, use_container_width=True)

# ── ROW 2: Yield + Backtest ───────────────────────────────────────────────────
c_yield, c_bt = st.columns(2, gap="medium")

with c_yield:
    st.markdown('<p class="section-title">Yield vs Historical Average</p>', unsafe_allow_html=True)
    colors = ["#f97316" if y < params["historical_avg_yield"] else "#38bdf8" for y in dataset["yield_boxes"]]
    fig_yield = go.Figure()
    fig_yield.add_bar(x=dataset["year"], y=dataset["yield_boxes"], marker_color=colors, name="Yield",
                      hovertemplate="<b>%{x}</b><br>Yield: %{y:,.0f} boxes<extra></extra>")
    fig_yield.add_hline(y=params["historical_avg_yield"], line_dash="dash", line_color="#94a3b8",
                        annotation_text="Hist. Avg", annotation_font_color="#64748b")
    chart_layout(fig_yield)
    fig_yield.update_layout(yaxis_title="Boxes", xaxis_title="")
    st.plotly_chart(fig_yield, use_container_width=True)

with c_bt:
    st.markdown('<p class="section-title">Backtest — Predicted vs Actual Yield</p>', unsafe_allow_html=True)
    fig_bt = go.Figure()
    fig_bt.add_scatter(x=backtest["year"], y=backtest["actual_yield"],
                       mode="lines+markers", name="Actual",
                       line=dict(color="#f97316", width=2), marker=dict(size=6),
                       hovertemplate="<b>%{x}</b><br>Actual: %{y:,.0f} boxes<extra></extra>")
    fig_bt.add_scatter(x=backtest["year"], y=backtest["predicted_yield"],
                       mode="lines+markers", name="Predicted",
                       line=dict(color="#38bdf8", width=2, dash="dash"), marker=dict(size=6),
                       hovertemplate="<b>%{x}</b><br>Predicted: %{y:,.0f} boxes<extra></extra>")
    chart_layout(fig_bt)
    fig_bt.update_layout(yaxis_title="Boxes", xaxis_title="")
    st.plotly_chart(fig_bt, use_container_width=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ── BOTTOM ROW: Outlook | Model Perf | Data Sources ───────────────────────────
b1, b2, b3 = st.columns(3, gap="medium")

with b1:
    st.markdown(f"""<div class="panel-card">
  <div class="panel-title">🍊 {harvest_year} Outlook</div>
  <div class="outlook-metric"><span class="outlook-key">Predicted Yield</span><span class="outlook-val">{latest['yield_boxes']/1e6:.2f}M boxes</span></div>
  <div class="outlook-metric"><span class="outlook-key">Market Signal</span><span class="outlook-val" style="color:{'#22c55e' if pressure=='bullish' else '#ef4444' if pressure=='bearish' else '#f59e0b'}">{pressure.capitalize()}</span></div>
  <div class="outlook-metric"><span class="outlook-key">Backtest Accuracy</span><span class="outlook-val">{accuracy:.0%}</span></div>
  <div class="outlook-metric" style="border:none"><span class="outlook-key">Trend</span><span class="outlook-val" style="color:{'#22c55e' if ndvi_trend_pct>=0 else '#ef4444'}">{trend_arrow} {abs(ndvi_trend_pct):.1f}%</span></div>
  <p style="color:#475569;font-size:.7rem;margin-top:1rem;line-height:1.5;">Accuracy is limited because NDVI alone doesn't fully capture citrus greening disease (HLB) impact on yield. Year trend is included as a second feature to partially account for this.</p>
</div>""", unsafe_allow_html=True)

with b2:
    st.markdown(f"""<div class="panel-card">
  <div class="panel-title">📊 Model Performance</div>
  <div class="perf-row"><span class="perf-key">R² Score</span><span class="perf-val">{params['r2_train']:.2f}</span></div>
  <div class="perf-row"><span class="perf-key">MAE</span><span class="perf-val">{params['mae_loo']/1e6:.1f}M boxes</span></div>
  <div class="perf-row"><span class="perf-key">Accuracy</span><span class="perf-val">{accuracy:.0%}</span></div>
  <div class="perf-row" style="border:none"><span class="perf-key">Backtest Years</span><span class="perf-val">{len(backtest)}</span></div>
</div>""", unsafe_allow_html=True)

with b3:
    st.markdown(f"""<div class="panel-card">
  <div class="panel-title">🛰️ Data Sources</div>
  <div class="ds-item"><div class="ds-dot"></div>NASA MODIS NDVI</div>
  <div class="ds-item"><div class="ds-dot"></div>USDA NASS Data</div>
  <div class="ds-item"><div class="ds-dot"></div>CME OJ Futures</div>
  <div class="ds-item" style="border:none"><div class="ds-dot"></div>Google Earth Engine</div>
</div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

with st.expander("📖 View Detailed Data & Methodology"):
    st.markdown("""
**Data Sources**

| Source | Description | Update Frequency |
|--------|-------------|-----------------|
| NASA MODIS MOD13A1 | 500m 16-day NDVI composite via Google Earth Engine | 16-day |
| USDA NASS Quick Stats | Florida orange production (ALL CLASSES, SURVEY) | Annual |
| CME OJ Futures (OJ=F) | Orange juice front-month futures price via yfinance | Daily |

---

**Model Methodology**

Starvest uses a linear regression model with two features:
- **Mean NDVI** — average Normalized Difference Vegetation Index over the Florida citrus belt (Polk, Highlands, DeSoto counties) during the Oct–May growing season. Higher NDVI indicates healthier vegetation.
- **Year** — captures the long-run decline in Florida citrus production driven by Huanglongbing (HLB / citrus greening disease), which NDVI alone cannot distinguish from weather effects.

**Price Pressure Logic**
- **Bullish** — predicted yield is >10% below historical average → supply shock → upward price pressure on OJ futures.
- **Bearish** — predicted yield is >10% above historical average → oversupply → downward price pressure.
- **Neutral** — predicted yield within ±10% of historical average.

**Backtest**
Walk-forward validation: for each year, the model is trained only on prior years and never sees future data. This gives an honest estimate of out-of-sample accuracy.

**Limitations**
- Small dataset (9 years of annual observations). Accuracy will improve as more data accumulates.
- NDVI doesn't directly measure HLB infection severity — the year trend is a proxy, not a mechanistic variable.
- OJ futures are influenced by factors beyond Florida supply (Brazil production, weather, macroeconomics).
""")

