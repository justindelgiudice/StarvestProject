import base64
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import urllib.request

DATA  = Path(__file__).parent.parent / "data" / "processed"
RAW   = Path(__file__).parent.parent / "data" / "raw"
LOGO  = Path(__file__).parent.parent / "Assets" / "StarvestLogo.png"
DS    = Path(__file__).parent.parent / "Data Source"

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

  /* ── Section spacing ── */
  .section-gap { margin-top: 1.8rem; }

  /* ── KPI cards ── */
  .kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    height: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    box-sizing: border-box;
  }
  .kpi-label { color: #64748b; font-size: 0.72rem; text-transform: uppercase; letter-spacing: .08em; margin-bottom: .4rem; }
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

@st.cache_data(ttl=86400)
def load_florida_counties():
    try:
        url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        with urllib.request.urlopen(url, timeout=12) as r:
            data = json.loads(r.read())
        fl = {"type": "FeatureCollection",
              "features": [f for f in data["features"] if f["id"].startswith("12")]}
        return fl
    except Exception:
        return None

def _geojson_latlons(geojson):
    lats, lons = [], []
    for feat in geojson["features"]:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "Polygon" else sum(geom["coordinates"], [])
        for ring in polys:
            for lon, lat in ring:
                lons.append(lon); lats.append(lat)
            lons.append(None); lats.append(None)
    return lats, lons

def _img_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

@st.cache_data
def load_county_seasonal():
    path = DATA / "ndvi_county_seasonal.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

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
CHART_CFG   = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "toImage", "select2d", "lasso2d",
        "zoomIn2d", "zoomOut2d", "autoScale2d",
        "hoverClosestCartesian", "hoverCompareCartesian",
        "toggleSpikelines",
    ],
}

def gap():
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

def hint(text: str):
    st.markdown(
        f'<p style="color:#94a3b8;font-size:.73rem;line-height:1.5;margin-top:.35rem;">{text}</p>',
        unsafe_allow_html=True,
    )

def ndvi_fill_rgba(val: float, alpha: float = 0.45) -> str:
    stops = [(0.0,(239,68,68)),(0.3,(249,115,22)),(0.5,(234,179,8)),(1.0,(34,197,94))]
    for i in range(len(stops) - 1):
        v0, c0 = stops[i]; v1, c1 = stops[i + 1]
        if val <= v1 or i == len(stops) - 2:
            t = max(0.0, min(1.0, (val - v0) / (v1 - v0) if v1 > v0 else 0.0))
            r = int(c0[0] + t * (c1[0] - c0[0]))
            g = int(c0[1] + t * (c1[1] - c0[1]))
            b = int(c0[2] + t * (c1[2] - c0[2]))
            return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(34,197,94,{alpha})"

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
  <p style="color:#94a3b8;font-size:.68rem;line-height:1.45;margin-top:.5rem;">Bullish = supply shortage → prices likely to rise. Bearish = oversupply → prices likely to fall. Neutral = near-average supply, no strong signal.</p>
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

gap()

# ── ROW 1: Map + NDVI Trend ───────────────────────────────────────────────────
c_map, c_ndvi = st.columns(2, gap="medium")

with c_map:
    st.markdown('<p class="section-title">Florida Citrus Belt</p>', unsafe_allow_html=True)

    ALL_CITRUS_FIPS = {"12105","12055","12027","12049","12051","12015","12043","12081"}
    fl_counties      = load_florida_counties()
    county_seasonal  = load_county_seasonal()

    # Default NDVI value (regional average from latest dataset row)
    ndvi_val  = float(latest["mean_ndvi"])
    year_data = None
    sel_year  = None

    # ── Year slider (only shown when per-county data is available) ────────────
    if county_seasonal is not None:
        available_years = sorted(county_seasonal["year"].unique().tolist())
        if "ndvi_map_year" not in st.session_state:
            st.session_state["ndvi_map_year"] = int(max(available_years))

        _sc1, _sc2 = st.columns([5, 1], gap="small")
        with _sc2:
            if st.button("Latest", use_container_width=True, key="btn_ndvi_latest"):
                st.session_state["ndvi_map_year"] = int(max(available_years))
                st.rerun()
        with _sc1:
            sel_year = st.select_slider(
                "Year",
                options=available_years,
                value=st.session_state.get("ndvi_map_year", max(available_years)),
                key="ndvi_map_year",
            )

        year_data = county_seasonal[county_seasonal["year"] == sel_year].copy()
        if not year_data.empty:
            ndvi_val = float(year_data["mean_ndvi"].mean())

    # ── Build map figure ──────────────────────────────────────────────────────
    fig_map = go.Figure()

    # Non-citrus FL county borders (thin white lines)
    if fl_counties:
        non_citrus_geo = {"type": "FeatureCollection",
                          "features": [f for f in fl_counties["features"]
                                       if f["id"] not in ALL_CITRUS_FIPS]}
        nc_lats, nc_lons = _geojson_latlons(non_citrus_geo)
        fig_map.add_trace(go.Scattermapbox(
            lat=nc_lats, lon=nc_lons, mode="lines",
            line=dict(color="rgba(255,255,255,0.35)", width=0.8),
            hoverinfo="skip", showlegend=False,
        ))

    # Per-county NDVI choropleth
    if fl_counties and year_data is not None and not year_data.empty:
        citrus_features = [f for f in fl_counties["features"] if f["id"] in ALL_CITRUS_FIPS]
        citrus_geo      = {"type": "FeatureCollection", "features": citrus_features}
        fig_map.add_trace(go.Choroplethmapbox(
            geojson=citrus_geo,
            locations=year_data["geoid"].astype(str).tolist(),
            z=year_data["mean_ndvi"].tolist(),
            text=year_data["county"].tolist(),
            featureidkey="id",
            colorscale=[[0,"#ef4444"],[0.3,"#f97316"],[0.5,"#eab308"],[1.0,"#22c55e"]],
            zmin=0.0, zmax=1.0,
            showscale=False,
            marker_opacity=0.72,
            marker_line_width=1.8,
            marker_line_color="#f97316",
            hovertemplate="<b>%{text} County</b><br>NDVI: %{z:.4f}<extra></extra>",
        ))
    else:
        # Fallback rectangle until per-county data is fetched
        ndvi_fill = ndvi_fill_rgba(ndvi_val)
        if fl_counties:
            citrus_geo2 = {"type": "FeatureCollection",
                           "features": [f for f in fl_counties["features"]
                                        if f["id"] in ALL_CITRUS_FIPS]}
            c_lats2, c_lons2 = _geojson_latlons(citrus_geo2)
            fig_map.add_trace(go.Scattermapbox(
                lat=c_lats2, lon=c_lons2, mode="lines",
                line=dict(color="#f97316", width=2.2),
                hoverinfo="skip", showlegend=False,
            ))
        fig_map.add_trace(go.Scattermapbox(
            lat=[27.0, 28.2, 28.2, 27.0, 27.0],
            lon=[-82.0, -82.0, -81.0, -81.0, -82.0],
            mode="lines", fill="toself",
            fillcolor=ndvi_fill, line=dict(color="rgba(0,0,0,0)", width=0),
            name="Citrus Belt",
            hovertemplate=f"NDVI: {ndvi_val:.4f}<extra></extra>",
        ))

    # Regional average label
    label_suffix = f" ({sel_year})" if sel_year else ""
    fig_map.add_trace(go.Scattermapbox(
        lat=[27.55], lon=[-81.5], mode="markers+text",
        marker=dict(size=13, color="#f97316"),
        text=[f"Avg NDVI {ndvi_val:.3f}{label_suffix}"],
        textposition="top right",
        textfont=dict(color="#ffffff", size=11),
        hovertemplate=f"Regional avg NDVI: {ndvi_val:.4f}<extra></extra>",
    ))

    fig_map.update_layout(
        mapbox=dict(
            style="white-bg",
            center=dict(lat=27.55, lon=-81.3),
            zoom=6.2,
            layers=[{
                "below": "traces",
                "sourcetype": "raster",
                "sourceattribution": "ESRI World Imagery",
                "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
            }],
        ),
        height=390, margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="#ffffff", showlegend=False,
    )
    st.plotly_chart(fig_map, use_container_width=True, config=CHART_CFG)

    # NDVI gradient legend
    ndvi_pct = min(ndvi_val * 100, 100)
    ndvi_status = "Healthy" if ndvi_val >= 0.5 else ("Moderate" if ndvi_val >= 0.3 else "Stressed")
    ndvi_status_color = "#22c55e" if ndvi_val >= 0.5 else ("#eab308" if ndvi_val >= 0.3 else "#ef4444")
    st.markdown(f"""
<div style="margin-top:.4rem;padding:.5rem .1rem 0;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.3rem;">
    <span style="color:#64748b;font-size:.7rem;font-weight:500;">NDVI Health Scale</span>
    <span style="color:{ndvi_status_color};font-size:.7rem;font-weight:600;">{ndvi_status} &nbsp;({ndvi_val:.4f})</span>
  </div>
  <div style="position:relative;height:7px;border-radius:4px;
              background:linear-gradient(to right,#ef4444 0%,#f97316 30%,#eab308 50%,#22c55e 100%);">
    <div style="position:absolute;top:-4px;left:{ndvi_pct:.1f}%;width:3px;height:15px;
                background:#0f172a;border-radius:2px;transform:translateX(-50%);"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:.25rem;">
    <span style="color:#ef4444;font-size:.65rem;">Stressed (&lt;0.3)</span>
    <span style="color:#eab308;font-size:.65rem;">Moderate (0.3–0.5)</span>
    <span style="color:#22c55e;font-size:.65rem;">Healthy (&gt;0.5)</span>
  </div>
</div>
""", unsafe_allow_html=True)

with c_ndvi:
    st.markdown('<p class="section-title">NDVI Trend — Florida Citrus Belt</p>', unsafe_allow_html=True)
    fig_ndvi = go.Figure()
    fig_ndvi.add_trace(go.Scatter(
        x=ndvi_raw["date"], y=ndvi_raw["mean_ndvi"],
        mode="lines+markers", name="NDVI",
        line=dict(color="#22c55e", width=1.5),
        marker=dict(size=4, color="#22c55e", opacity=0.45,
                    line=dict(color="#16a34a", width=0.5)),
        fill="tozeroy", fillcolor="rgba(34,197,94,0.08)",
        hovertemplate="<b>%{x|%b %Y}</b><br>NDVI: %{y:.4f}<extra></extra>",
    ))
    chart_layout(fig_ndvi, height=420)
    fig_ndvi.update_layout(yaxis_title="Mean NDVI", xaxis_title="")
    st.plotly_chart(fig_ndvi, use_container_width=True, config=CHART_CFG)
    hint("NDVI (Normalized Difference Vegetation Index) measures plant greenness from satellite data. Values above 0.5 indicate healthy, dense vegetation; below 0.3 signals stressed or sparse crops. Each data point is a 16-day average over the Florida citrus belt.")

gap()
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
    st.plotly_chart(fig_yield, use_container_width=True, config=CHART_CFG)
    hint("Orange bars = yield fell below the long-run historical average (potential supply shortage). Blue bars = yield exceeded the average (abundant crop). The dashed line is the average across all years in the dataset.")

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
    st.plotly_chart(fig_bt, use_container_width=True, config=CHART_CFG)
    hint("A backtest checks how well the model would have predicted years it had never seen — trained only on earlier data each time, mimicking a real forecast. The closer the orange and blue lines, the more trustworthy the model's predictions.")

gap()

# ── YIELD vs OJ PRICE IMPACT ─────────────────────────────────────────────────
st.markdown('<p class="section-title">Yield vs. OJ Price Impact</p>', unsafe_allow_html=True)

c_ts, c_sc = st.columns(2, gap="medium")

with c_ts:
    st.markdown('<p class="section-title">Yield & OJ Price — Annual</p>', unsafe_allow_html=True)
    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ts.add_trace(go.Bar(
        x=dataset["year"], y=dataset["yield_boxes"],
        name="Yield (boxes)",
        marker_color="#f97316",
        opacity=0.7,
        hovertemplate="<b>%{x}</b><br>Yield: %{y:,.0f} boxes<extra></extra>",
    ), secondary_y=False)
    fig_ts.add_trace(go.Scatter(
        x=dataset["year"], y=dataset["avg_oj_price"],
        name="OJ Price (¢/lb)",
        line=dict(color="#2563eb", width=2.5),
        mode="lines+markers",
        marker=dict(size=7),
        hovertemplate="<b>%{x}</b><br>OJ Price: ¢%{y:.1f}<extra></extra>",
    ), secondary_y=True)
    chart_layout(fig_ts)
    fig_ts.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=10, b=30, l=10, r=50),
    )
    fig_ts.update_yaxes(
        title_text="Yield (boxes)", secondary_y=False,
        gridcolor=GRID_COLOR, linecolor=GRID_COLOR, tickfont=dict(color=FONT_COLOR),
    )
    fig_ts.update_yaxes(
        title_text="OJ Price (¢/lb)", secondary_y=True,
        gridcolor="rgba(0,0,0,0)", linecolor=GRID_COLOR, tickfont=dict(color="#2563eb"),
        title_font=dict(color="#2563eb"),
    )
    st.plotly_chart(fig_ts, use_container_width=True, config=CHART_CFG)

with c_sc:
    st.markdown('<p class="section-title">Yield vs. OJ Price — Scatter</p>', unsafe_allow_html=True)
    x_sc    = dataset["yield_vs_avg"].values
    y_sc    = dataset["avg_oj_price"].values
    years_sc = dataset["year"].values.astype(int)

    m, b_int = np.polyfit(x_sc, y_sc, 1)
    x_line   = np.linspace(x_sc.min(), x_sc.max(), 50)
    y_line   = m * x_line + b_int

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=x_sc, y=y_sc,
        mode="markers+text",
        text=[str(y) for y in years_sc],
        textposition="top center",
        textfont=dict(size=9, color="#64748b"),
        marker=dict(
            size=11,
            color=years_sc,
            colorscale="Oranges",
            showscale=False,
            line=dict(color="#0f172a", width=0.8),
        ),
        hovertemplate="<b>%{text}</b><br>Yield vs Avg: %{x:.2f}×<br>OJ Price: ¢%{y:.1f}<extra></extra>",
        name="",
    ))
    fig_sc.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        line=dict(color="#ef4444", width=1.5, dash="dash"),
        name="Trend",
        hoverinfo="skip",
    ))
    chart_layout(fig_sc)
    fig_sc.update_layout(
        xaxis_title="Yield vs. Historical Avg (ratio)",
        yaxis_title="OJ Price (¢/lb)",
        showlegend=False,
    )
    st.plotly_chart(fig_sc, use_container_width=True, config=CHART_CFG)

# Correlation summary bar
corr    = float(np.corrcoef(dataset["yield_vs_avg"], dataset["avg_oj_price"])[0, 1])
corr_lbl = "lower yield → higher OJ prices (inverse relationship)" if corr < 0 else "higher yield → higher OJ prices (direct relationship)"
corr_color = "#2563eb" if corr < 0 else "#16a34a"
st.markdown(f"""
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;
            padding:.85rem 1.2rem;margin-top:.5rem;display:flex;align-items:center;gap:.8rem;">
  <span style="color:#64748b;font-size:.8rem;">Correlation (yield ratio vs. OJ price):</span>
  <span style="color:{corr_color};font-size:1rem;font-weight:700;">r = {corr:+.3f}</span>
  <span style="color:#94a3b8;font-size:.78rem;">— {corr_lbl}. Strength: {abs(corr):.0%}.</span>
</div>
""", unsafe_allow_html=True)
hint("r (correlation coefficient) ranges from –1 to +1. r near –1 means lower yield reliably predicts higher OJ prices; r near +1 means higher yield predicts higher prices; r near 0 means no consistent link. The closer |r| is to 1, the stronger the relationship.")

gap()

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
  <p style="color:#94a3b8;font-size:.7rem;line-height:1.55;margin-top:.8rem;border-top:1px solid #e2e8f0;padding-top:.6rem;">
    <b style="color:#64748b;">R²</b> — how much yield variation the model explains (0 = nothing, 1 = perfect fit).<br>
    <b style="color:#64748b;">MAE</b> — average prediction error in boxes; lower is better.<br>
    <b style="color:#64748b;">Accuracy</b> — % of years the bullish/bearish/neutral price signal was predicted correctly.
  </p>
</div>""", unsafe_allow_html=True)

with b3:
    ds_entries = [
        (DS / "NASA-Logo-Large.png",    "NASA MODIS NDVI"),
        (DS / "usda-1-logo.png",        "USDA NASS Data"),
        (DS / "CME_Group_Logo.svg.png", "CME OJ Futures"),
        (DS / "earth_engine_icon.png",  "Google Earth Engine"),
    ]
    rows_html = ""
    for logo_path, label in ds_entries:
        b64 = _img_b64(logo_path)
        rows_html += f"""
<div style="display:flex;align-items:center;gap:.75rem;
            padding:.55rem 0;border-bottom:1px solid #e2e8f0;">
  <div style="width:40px;height:40px;flex-shrink:0;
              display:flex;align-items:center;justify-content:center;">
    <img src="data:image/png;base64,{b64}"
         style="max-width:40px;max-height:40px;object-fit:contain;"/>
  </div>
  <span style="color:#0f172a;font-size:.85rem;font-weight:600;line-height:1.2;">{label}</span>
</div>"""
    st.markdown(f"""
<div class="panel-card">
  <div class="panel-title">🛰️ Data Sources</div>
  {rows_html}
</div>""", unsafe_allow_html=True)

gap()

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

