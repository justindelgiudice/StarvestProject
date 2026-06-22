import base64
import certifi
import json
import ssl
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
        ctx = ssl.create_default_context(cafile=certifi.where())
        url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        with urllib.request.urlopen(url, timeout=15, context=ctx) as r:
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

def county_centroid(feature) -> tuple[float | None, float | None]:
    """Return (lat, lon) centroid of a GeoJSON county feature."""
    try:
        geom = feature["geometry"]
        rings = ([geom["coordinates"][0]] if geom["type"] == "Polygon"
                 else [p[0] for p in geom["coordinates"]])
        all_lons = [c[0] for ring in rings for c in ring]
        all_lats = [c[1] for ring in rings for c in ring]
        return sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)
    except Exception:
        return None, None

def _img_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

@st.cache_data
def load_county_seasonal():
    path = DATA / "ndvi_county_seasonal.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_county_dataset():
    path = DATA / "county_dataset.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_price_model_params():
    path = DATA / "price_model_params.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

@st.cache_data
def load_price_backtest():
    path = DATA / "price_backtest_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

try:
    dataset, backtest, params = load_data()
    ndvi_raw = load_ndvi_raw()
except FileNotFoundError:
    st.error("Run the pipeline first: build_dataset → model → backtest")
    st.stop()

price_params = load_price_model_params()
price_bt     = load_price_backtest()

dataset = dataset.sort_values("year").reset_index(drop=True)
latest  = dataset.iloc[-1]
prev    = dataset.iloc[-2] if len(dataset) > 1 else latest

# ── Derived metrics ─────────────────────────────────────────────────────────
ndvi_trend_pct = (latest["mean_ndvi"] - prev["mean_ndvi"]) / prev["mean_ndvi"] * 100
accuracy       = (backtest["actual_pressure"] == backtest["predicted_pressure"]).mean()
# Use price model's trained signal if available; fall back to rule-based threshold
pressure       = price_params["price_pressure"] if price_params else latest["price_pressure"]
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
    if val >= 0.6:
        return f"rgba(34,197,94,{alpha})"    # green — healthy
    elif val >= 0.4:
        return f"rgba(234,179,8,{alpha})"    # yellow — moderate
    else:
        return f"rgba(239,68,68,{alpha})"    # red — stressed

def chart_layout(fig, height=380):
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
    logo_b64 = _img_b64(LOGO)
    st.markdown(f"""
<div style="display:flex;flex-direction:column;align-items:flex-start;gap:0;line-height:1;">
  <img src="data:image/png;base64,{logo_b64}" style="width:460px;display:block;margin:0;padding:0;"/>
  <p style="color:#64748b;font-size:.82rem;margin:4px 0 2px 0;padding:0;">Satellite-Powered Citrus Commodity Forecasting</p>
  <p style="color:#94a3b8;font-size:.75rem;margin:0;padding:0;">NASA MODIS NDVI &nbsp;|&nbsp; USDA Yield Data &nbsp;|&nbsp; OJ Futures</p>
</div>
""", unsafe_allow_html=True)

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
  <div class="kpi-sub">{"Price model" if price_params else "Rule-based"}</div>
  <p style="color:#94a3b8;font-size:.68rem;line-height:1.45;margin-top:.5rem;">{"Derived from trained price regression — see Price Forecast section." if price_params else "Bullish = yield >10% below avg. Bearish = >10% above. Neutral = within ±10%."}</p>
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

    ALL_CITRUS_FIPS  = {"12105","12055","12027","12049","12051","12015","12043","12081"}
    fl_counties      = load_florida_counties()
    county_seasonal  = load_county_seasonal()
    county_dataset   = load_county_dataset()

    # Default NDVI value (regional average from latest dataset row)
    ndvi_val  = float(latest["mean_ndvi"])
    year_data = None
    sel_year  = None

    # ── Map layer toggle ──────────────────────────────────────────────────────
    map_view = st.radio(
        "Map Layer",
        ["Vegetation Index", "Citrus Output"],
        horizontal=True,
        label_visibility="collapsed",
        key="map_layer_view",
    )

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
                key="ndvi_map_year",
            )

        year_data = county_seasonal[county_seasonal["year"] == sel_year].copy()
        if not year_data.empty:
            ndvi_val = float(year_data["mean_ndvi"].mean())

    # ── Build map figure ──────────────────────────────────────────────────────
    fig_map = go.Figure()
    _z_max = 1.0  # overwritten inside Citrus Output block; kept in scope for the legend

    if fl_counties:
        citrus_features = [f for f in fl_counties["features"] if f["id"] in ALL_CITRUS_FIPS]
        citrus_geo      = {"type": "FeatureCollection", "features": citrus_features}

        # ── All-67-county choropleth ──────────────────────────────────────────
        if map_view == "Citrus Output":
            # Build 67-county yield table: 8 citrus counties get real USDA data, other 59 get 0.
            # Gray (0) vs blue gradient lets the viewer see at a glance where citrus is grown.
            _all_fips  = [f["id"] for f in fl_counties["features"]]
            _all_names = {f["id"]: f["properties"].get("NAME", f["id"])
                          for f in fl_counties["features"]}
            _yield_all = pd.DataFrame({
                "geoid":            _all_fips,
                "county":           [_all_names[g] for g in _all_fips],
                "county_yield_est": 0.0,
            })
            if county_dataset is not None and sel_year is not None:
                _eff_yr = min(sel_year, int(county_dataset["year"].max()))
                _yy = county_dataset[county_dataset["year"] == _eff_yr].copy()
                _yy["geoid"] = _yy["geoid"].astype(str)
                for _, _r in _yy.iterrows():
                    _yield_all.loc[_yield_all["geoid"] == _r["geoid"], "county_yield_est"] = _r["county_yield_est"]
            _z_y   = _yield_all["county_yield_est"].tolist()
            _z_max = max(_z_y) if max(_z_y) > 0 else 1.0
            fig_map.add_trace(go.Choroplethmapbox(
                geojson=fl_counties,
                locations=_yield_all["geoid"].tolist(),
                z=_z_y,
                text=_yield_all["county"].tolist(),
                featureidkey="id",
                colorscale=[
                    [0.0,   "#9ca3af"],
                    [0.001, "#9ca3af"],
                    [0.001, "#bfdbfe"],
                    [1.0,   "#1e40af"],
                ],
                zmin=0, zmax=_z_max,
                showscale=False,
                marker_opacity=0.82,
                marker_line_width=0.5,
                marker_line_color="rgba(255,255,255,0.4)",
                hovertemplate="<b>%{text} County</b><br>Est. Yield: %{z:,.0f} boxes<extra></extra>",
            ))
        elif year_data is not None and not year_data.empty:
            # NDVI layer — all 67 counties with actual satellite data
            fig_map.add_trace(go.Choroplethmapbox(
                geojson=fl_counties,
                locations=year_data["geoid"].astype(str).tolist(),
                z=year_data["mean_ndvi"].tolist(),
                text=year_data["county"].tolist(),
                featureidkey="id",
                colorscale=[
                    [0.0,  "#ef4444"], [0.39, "#ef4444"],
                    [0.40, "#eab308"], [0.59, "#eab308"],
                    [0.60, "#22c55e"], [1.0,  "#22c55e"],
                ],
                zmin=0.0, zmax=1.0,
                showscale=False,
                marker_opacity=0.72,
                marker_line_width=0.5,
                marker_line_color="rgba(255,255,255,0.4)",
                hovertemplate="<b>%{text} County</b><br>NDVI: %{z:.4f}<extra></extra>",
            ))
        else:
            # Fallback: no county data available yet
            ndvi_fill = ndvi_fill_rgba(ndvi_val)
            c_lats2, c_lons2 = _geojson_latlons(citrus_geo)
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

        # ── Bold orange outline for the 8 primary citrus counties ─────────────
        _c_lats, _c_lons = _geojson_latlons(citrus_geo)
        fig_map.add_trace(go.Scattermapbox(
            lat=_c_lats, lon=_c_lons, mode="lines",
            line=dict(color="#f97316", width=2.5),
            hoverinfo="skip", showlegend=False,
        ))
        fig_map.add_trace(go.Scattermapbox(
            lat=[28.05], lon=[-81.55],
            mode="text",
            text=["Florida Citrus Belt"],
            textfont=dict(color="#f97316", size=9),
            hoverinfo="skip", showlegend=False,
        ))

        # ── County name labels for all 67 counties ────────────────────────────
        _lbl_lats, _lbl_lons, _lbl_texts = [], [], []
        for _feat in fl_counties["features"]:
            _clat, _clon = county_centroid(_feat)
            if _clat is not None:
                _lbl_lats.append(_clat)
                _lbl_lons.append(_clon)
                _lbl_texts.append(_feat["properties"].get("NAME", _feat["id"]))
        if _lbl_lats:
            fig_map.add_trace(go.Scattermapbox(
                lat=_lbl_lats, lon=_lbl_lons,
                mode="text",
                text=_lbl_texts,
                textfont=dict(color="rgba(255,255,255,0.85)", size=7),
                hoverinfo="skip",
                showlegend=False,
            ))

    # Regional average label
    label_suffix = f" ({sel_year})" if sel_year else ""
    if map_view == "Citrus Output" and county_dataset is not None and sel_year is not None:
        _eff_yr_lbl = min(sel_year, int(county_dataset["year"].max()))
        yield_year_lbl = county_dataset[county_dataset["year"] == _eff_yr_lbl]
        total_yield_lbl = yield_year_lbl["county_yield_est"].sum() if not yield_year_lbl.empty else 0
        _yr_note = f" ({_eff_yr_lbl} data)" if _eff_yr_lbl != sel_year else label_suffix
        map_label     = f"Est. {total_yield_lbl/1e6:.1f}M boxes{_yr_note}"
        map_hover     = f"Estimated citrus yield: {total_yield_lbl:,.0f} boxes<extra></extra>"
        dot_color     = "#2563eb"
    else:
        map_label = f"Avg NDVI {ndvi_val:.3f}{label_suffix}"
        map_hover = f"Regional avg NDVI: {ndvi_val:.4f}<extra></extra>"
        dot_color = "#f97316"
    fig_map.add_trace(go.Scattermapbox(
        lat=[27.55], lon=[-81.5], mode="markers+text",
        marker=dict(size=13, color=dot_color),
        text=[map_label],
        textposition="top right",
        textfont=dict(color="#ffffff", size=11),
        hovertemplate=map_hover,
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
        height=480, margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="#ffffff", showlegend=False,
    )
    st.plotly_chart(fig_map, use_container_width=True, config=CHART_CFG)

    # Legend switches with the map toggle
    if map_view == "Citrus Output":
        _legend_yr = min(sel_year, int(county_dataset["year"].max())) if (sel_year is not None and county_dataset is not None) else ""
        _yr_lbl = f" — {_legend_yr} USDA estimates" if _legend_yr else ""
        st.markdown(f"""
<div style="margin-top:.4rem;padding:.5rem .1rem 0;">
  <div style="margin-bottom:.45rem;">
    <span style="color:#64748b;font-size:.7rem;font-weight:500;">Citrus Yield by County{_yr_lbl}</span>
  </div>
  <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem;">
    <div style="width:22px;height:9px;background:#9ca3af;border-radius:2px;flex-shrink:0;"></div>
    <span style="color:#64748b;font-size:.67rem;">No citrus production recorded (59 of 67 FL counties)</span>
  </div>
  <div style="height:7px;border-radius:4px;background:linear-gradient(to right,#bfdbfe,#1e40af);"></div>
  <div style="display:flex;justify-content:space-between;margin-top:.25rem;">
    <span style="color:#64748b;font-size:.65rem;">Low yield (0 boxes)</span>
    <span style="color:#1e40af;font-size:.65rem;">High yield ({_z_max/1e6:.1f}M boxes)</span>
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        ndvi_pct = min(ndvi_val * 100, 100)
        ndvi_status = "Healthy" if ndvi_val >= 0.6 else ("Moderate" if ndvi_val >= 0.4 else "Stressed")
        ndvi_status_color = "#22c55e" if ndvi_val >= 0.6 else ("#eab308" if ndvi_val >= 0.4 else "#ef4444")
        st.markdown(f"""
<div style="margin-top:.4rem;padding:.5rem .1rem 0;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.3rem;">
    <span style="color:#64748b;font-size:.7rem;font-weight:500;">NDVI Health Scale</span>
    <span style="color:{ndvi_status_color};font-size:.7rem;font-weight:600;">{ndvi_status} &nbsp;({ndvi_val:.4f})</span>
  </div>
  <div style="position:relative;height:7px;border-radius:4px;
              background:linear-gradient(to right,#ef4444 0%,#ef4444 39%,#eab308 40%,#eab308 59%,#22c55e 60%,#22c55e 100%);">
    <div style="position:absolute;top:-4px;left:{ndvi_pct:.1f}%;width:3px;height:15px;
                background:#0f172a;border-radius:2px;transform:translateX(-50%);"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:.25rem;">
    <span style="color:#ef4444;font-size:.65rem;">Stressed (&lt;0.4)</span>
    <span style="color:#eab308;font-size:.65rem;">Moderate (0.4–0.6)</span>
    <span style="color:#22c55e;font-size:.65rem;">Healthy (&ge;0.6)</span>
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
    chart_layout(fig_ndvi, height=480)
    fig_ndvi.update_layout(yaxis_title="Mean NDVI", xaxis_title="")
    st.plotly_chart(fig_ndvi, use_container_width=True, config=CHART_CFG)
    hint("NDVI (Normalized Difference Vegetation Index) measures plant greenness from satellite data. Values ≥ 0.6 indicate healthy, dense vegetation; 0.4–0.6 = moderate health; below 0.4 signals stressed or sparse crops. Each data point is a 16-day average over the Florida citrus belt.")

gap()
# ── NDVI vs Yield Correlation (county-level) ──────────────────────────────────
st.markdown('<p class="section-title">County NDVI vs. Estimated Yield</p>', unsafe_allow_html=True)
_cdf = load_county_dataset()
if _cdf is not None:
    _cdf_clean = _cdf.dropna(subset=["mean_ndvi", "county_yield_est"])
    _cx  = _cdf_clean["mean_ndvi"].values
    _cy  = _cdf_clean["county_yield_est"].values
    _cyr = _cdf_clean["year"].values.astype(int)
    _cm, _cb = np.polyfit(_cx, _cy, 1)
    _cx_line = np.linspace(_cx.min(), _cx.max(), 50)
    _cy_line = _cm * _cx_line + _cb
    _cr = float(np.corrcoef(_cx, _cy)[0, 1])

    c_ns, c_ns_info = st.columns([2.5, 1], gap="medium")
    with c_ns:
        fig_ns = go.Figure()
        fig_ns.add_trace(go.Scatter(
            x=_cx, y=_cy,
            mode="markers",
            text=[f"{c} {y}" for c, y in zip(_cdf_clean["county"].tolist(), _cyr.tolist())],
            marker=dict(
                size=9,
                color=_cyr,
                colorscale="Oranges",
                showscale=True,
                colorbar=dict(title="Year", thickness=10, len=0.7),
                line=dict(color="#0f172a", width=0.6),
                opacity=0.8,
            ),
            hovertemplate="<b>%{text}</b><br>NDVI: %{x:.4f}<br>Yield: %{y:,.0f} boxes<extra></extra>",
            name="County-year",
        ))
        fig_ns.add_trace(go.Scatter(
            x=_cx_line, y=_cy_line,
            mode="lines",
            line=dict(color="#ef4444", width=1.8, dash="dash"),
            name="Trend",
            hoverinfo="skip",
        ))
        fig_ns.add_annotation(
            x=_cx.max(), y=_cy_line[-1],
            text=f"r = {_cr:+.3f}",
            showarrow=False, xanchor="right",
            font=dict(size=12, color="#2563eb"),
        )
        chart_layout(fig_ns, height=380)
        fig_ns.update_layout(
            xaxis_title="County Mean NDVI (growing season)",
            yaxis_title="County Est. Yield (boxes)",
            showlegend=False,
        )
        st.plotly_chart(fig_ns, use_container_width=True, config=CHART_CFG)
        hint("Each point is one county for one year. NDVI is the growing-season average; yield is the county's USDA-proportioned estimate. The trend line shows the within-season NDVI-to-yield relationship across all counties and years.")
    with c_ns_info:
        _cr_color = "#16a34a" if abs(_cr) >= 0.6 else ("#d97706" if abs(_cr) >= 0.4 else "#dc2626")
        _cr_lbl = "higher NDVI → higher yield" if _cr > 0 else "higher NDVI → lower yield"
        st.markdown(f"""<div class="panel-card" style="margin-top:.1rem;">
  <div class="panel-title">🌿 NDVI–Yield Correlation</div>
  <div class="perf-row"><span class="perf-key">r (Pearson)</span><span class="perf-val" style="color:{_cr_color}">{_cr:+.3f}</span></div>
  <div class="perf-row"><span class="perf-key">County-years</span><span class="perf-val">{len(_cdf_clean)}</span></div>
  <div class="perf-row" style="border:none"><span class="perf-key">Strength</span><span class="perf-val">{abs(_cr):.0%}</span></div>
  <p style="color:#94a3b8;font-size:.7rem;line-height:1.55;margin-top:.8rem;border-top:1px solid #e2e8f0;padding-top:.6rem;">
    {_cr_lbl.capitalize()}.<br>
    <b style="color:#64748b;">r &gt; 0.6</b> = strong positive link between vegetation health and output.
  </p>
</div>""", unsafe_allow_html=True)

gap()
# ── ROW 2: Yield + Backtest ───────────────────────────────────────────────────
c_yield, c_bt = st.columns(2, gap="medium")

with c_yield:
    st.markdown('<p class="section-title">Yield vs Historical Average</p>', unsafe_allow_html=True)
    colors = ["#f97316" if y < params["historical_avg_yield"] else "#38bdf8" for y in dataset["yield_boxes"]]
    fig_yield = go.Figure()
    fig_yield.add_bar(x=dataset["year"], y=dataset["yield_boxes"], marker_color=colors, name="Actual",
                      hovertemplate="<b>%{x}</b><br>Actual Yield: %{y:,.0f} boxes<extra></extra>")

    # Predicted 2025-2027 bars — multiyear_forecasts from price_model.py.
    # Each year's predicted_yield comes from the county panel model (8 citrus counties only)
    # via coverage-fraction scaling. 2027 uses the mean of the last 3 growing seasons as NDVI.
    _multiyear = price_params.get("multiyear_forecasts", []) if price_params else []
    if _multiyear:
        _pred_years  = [fc["year"] for fc in _multiyear]
        _pred_yields = [fc["predicted_yield"] for fc in _multiyear]
        fig_yield.add_bar(
            x=_pred_years, y=_pred_yields,
            marker_color="#7c3aed",
            marker_line_color="#5b21b6",
            marker_line_width=2,
            marker_pattern_shape="/",
            name="Predicted",
            hovertemplate="<b>%{x} (Predicted)</b><br>Forecast Yield: %{y:,.0f} boxes<extra></extra>",
        )
        for _fc in _multiyear:
            fig_yield.add_annotation(
                x=_fc["year"], y=_fc["predicted_yield"],
                text="Predicted",
                showarrow=False, yanchor="bottom",
                font=dict(size=9, color="#5b21b6"),
            )

    fig_yield.add_hline(y=params["historical_avg_yield"], line_dash="dash", line_color="#94a3b8",
                        annotation_text="Hist. Avg", annotation_font_color="#64748b")
    chart_layout(fig_yield)
    fig_yield.update_layout(
        yaxis_title="Boxes", xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode="overlay",
    )
    st.plotly_chart(fig_yield, use_container_width=True, config=CHART_CFG)
    hint("Orange/blue bars = Actual USDA yield (orange = below historical avg, blue = above). Purple hatched bars = model-predicted yield for 2025–2027 (no official USDA data published yet). 2025–2026 use actual satellite NDVI; 2027 uses avg of last 3 seasons. Dashed line = long-run historical average.")

with c_bt:
    st.markdown('<p class="section-title">Backtest — Predicted vs Actual Yield</p>', unsafe_allow_html=True)
    fig_bt = go.Figure()
    fig_bt.add_scatter(x=backtest["year"], y=backtest["actual_yield"],
                       mode="lines+markers", name="Actual",
                       line=dict(color="#f97316", width=2), marker=dict(size=6),
                       hovertemplate="<b>%{x}</b><br>Actual: %{y:,.0f} boxes<extra></extra>")
    fig_bt.add_scatter(x=backtest["year"], y=backtest["predicted_yield"],
                       mode="lines+markers", name="Predicted (backtest)",
                       line=dict(color="#38bdf8", width=2, dash="dash"), marker=dict(size=6),
                       hovertemplate="<b>%{x}</b><br>Predicted: %{y:,.0f} boxes<extra></extra>")
    _multiyear_bt = price_params.get("multiyear_forecasts", []) if price_params else []
    if _multiyear_bt:
        _bt_years  = [fc["year"] for fc in _multiyear_bt]
        _bt_yields = [fc["predicted_yield"] for fc in _multiyear_bt]
        fig_bt.add_scatter(
            x=_bt_years, y=_bt_yields,
            mode="lines+markers+text",
            line=dict(color="#7c3aed", width=1.5, dash="dot"),
            marker=dict(size=11, color="#7c3aed", symbol="star"),
            text=[str(y) for y in _bt_years],
            textposition="top right",
            textfont=dict(color="#7c3aed", size=8),
            name="2025–2027 Forecast",
            hovertemplate="<b>%{text} Forecast</b><br>Predicted: %{y:,.0f} boxes<extra></extra>",
        )
    chart_layout(fig_bt)
    fig_bt.update_layout(
        yaxis_title="Boxes", xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_bt, use_container_width=True, config=CHART_CFG)
    hint("Orange = actual USDA yield per year. Blue dashes = walk-forward backtest predictions (each year trained only on prior years). Purple stars/dotted line = model forecasts for 2025–2027 (no USDA data published yet; 2027 uses average of last 3 growing seasons as NDVI proxy).")

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
        hovertemplate="<b>%{x}</b><br>OJ Price: ¢%{y:.1f}/lb<extra></extra>",
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

# ── OJ PRICE FORECAST ─────────────────────────────────────────────────────────
st.markdown('<p class="section-title">OJ Price Forecast</p>', unsafe_allow_html=True)

if price_params:
    pf_left, pf_mid, pf_right = st.columns([1.5, 2.5, 1.5], gap="medium")

    with pf_left:
        _mf_card = price_params.get("multiyear_forecasts", [])
        _dir_acc = price_params["directional_accuracy"]
        _last_yr = price_params["last_year"]
        _last_pr = price_params["last_price"]

        _fc_rows_html = ""
        for _fc in _mf_card:
            _pp   = _fc["price_pressure"]
            _pc_c = "#22c55e" if _fc["pct_change"] > 0 else "#ef4444"
            _pc_a = "▲" if _fc["pct_change"] > 0 else "▼"
            _pp_c = {"bullish": "#22c55e", "bearish": "#ef4444", "neutral": "#f59e0b"}.get(_pp, "#64748b")
            _src  = "NDVI" if _fc.get("ndvi_source", "") in ("yield_model_county", "yield_model") else "Est."
            _fc_rows_html += f"""
  <div class="perf-row">
    <span class="perf-key">{_fc['year']} ({_src})</span>
    <span class="perf-val" style="font-size:.82rem;">
      ¢{_fc['predicted_price']:.0f}/lb
      &nbsp;<span style="color:{_pc_c};font-size:.72rem;">{_pc_a}{abs(_fc['pct_change']):.0f}%</span>
      &nbsp;<span style="color:{_pp_c};font-size:.72rem;">{_pp.capitalize()}</span>
    </span>
  </div>"""

        if not _fc_rows_html:
            _fp = price_params["predicted_price"]
            _fc_rows_html = f"""<div class="perf-row"><span class="perf-key">Predicted OJ Price</span><span class="perf-val">¢{_fp:.1f}/lb</span></div>"""

        st.markdown(f"""<div class="panel-card">
  <div class="panel-title">💰 2025–2027 Price Forecast</div>
  <div class="perf-row"><span class="perf-key">{_last_yr} Actual</span><span class="perf-val">¢{_last_pr:.0f}/lb</span></div>
  {_fc_rows_html}
  <div class="perf-row" style="border:none"><span class="perf-key">Confidence</span><span class="perf-val">{_dir_acc:.0%} directional</span></div>
  <p style="color:#94a3b8;font-size:.7rem;line-height:1.5;margin-top:.8rem;border-top:1px solid #e2e8f0;padding-top:.6rem;">
    OJ futures quoted in ¢/lb. 2025–2026 use actual satellite NDVI; 2027 uses avg of last 3 growing seasons.
    Each year's predicted price feeds the next year's lagged-price input.
    Confidence = % of backtest years model correctly predicted price direction.
  </p>
</div>""", unsafe_allow_html=True)

    with pf_mid:
        st.markdown('<p class="section-title">Price Backtest — Predicted vs Actual OJ Price</p>', unsafe_allow_html=True)
        fig_pb = go.Figure()
        fig_pb.add_trace(go.Scatter(
            x=dataset["year"], y=dataset["avg_oj_price"],
            mode="lines+markers", name="Actual price",
            line=dict(color="#f97316", width=2.5),
            marker=dict(size=6),
            hovertemplate="<b>%{x}</b><br>Actual: ¢%{y:.1f}/lb<extra></extra>",
        ))
        if price_bt is not None and not price_bt.empty:
            fig_pb.add_trace(go.Scatter(
                x=price_bt["year"], y=price_bt["predicted_price"],
                mode="lines+markers", name="Predicted (backtest)",
                line=dict(color="#2563eb", width=2, dash="dash"),
                marker=dict(size=7, symbol="diamond"),
                hovertemplate="<b>%{x}</b><br>Predicted: ¢%{y:.1f}/lb<extra></extra>",
            ))
        _mf_pb = price_params.get("multiyear_forecasts", [])
        if _mf_pb:
            _fc_x = [price_params["last_year"]] + [f["year"] for f in _mf_pb]
            _fc_y = [price_params["last_price"]] + [f["predicted_price"] for f in _mf_pb]
            _fc_t = [""] + [f"¢{f['predicted_price']:.0f}" for f in _mf_pb]
            fig_pb.add_trace(go.Scatter(
                x=_fc_x, y=_fc_y,
                mode="lines+markers+text",
                line=dict(color="#7c3aed", width=2, dash="dot"),
                marker=dict(size=9, color="#7c3aed", symbol="diamond"),
                text=_fc_t,
                textposition="top right",
                textfont=dict(color="#7c3aed", size=9),
                name="2025–2027 Forecast",
                hovertemplate="<b>%{x}</b><br>Forecast: ¢%{y:.1f}/lb<extra></extra>",
            ))
        else:
            fig_pb.add_trace(go.Scatter(
                x=[price_params["forecast_year"]], y=[price_params["predicted_price"]],
                mode="markers+text",
                marker=dict(size=13, color="#7c3aed", symbol="star"),
                text=[f"¢{price_params['predicted_price']:.0f}"],
                textposition="top right",
                textfont=dict(color="#7c3aed", size=10),
                name=f"{price_params['forecast_year']} Forecast",
                hovertemplate=f"<b>{price_params['forecast_year']} Forecast</b><br>¢{price_params['predicted_price']:.1f}/lb<extra></extra>",
            ))
        chart_layout(fig_pb, height=380)
        fig_pb.update_layout(
            yaxis_title="OJ Price (¢/lb)", xaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_pb, use_container_width=True, config=CHART_CFG)
        hint("Orange = actual annual OJ futures prices. Blue dashes = walk-forward backtest predictions (each year trained only on prior years). Purple star = the model's forecast for the next year.")

    with pf_right:
        r2_col = "#16a34a" if price_params["r2_backtest"] >= 0.6 else ("#d97706" if price_params["r2_backtest"] >= 0.4 else "#dc2626")
        n_bt   = price_params["n_backtest_years"]
        n_dir  = round(price_params["directional_accuracy"] * n_bt)
        st.markdown(f"""<div class="panel-card">
  <div class="panel-title">📈 Price Model Performance</div>
  <div class="perf-row"><span class="perf-key">R² (backtest)</span><span class="perf-val" style="color:{r2_col}">{price_params["r2_backtest"]:.2f}</span></div>
  <div class="perf-row"><span class="perf-key">MAE (backtest)</span><span class="perf-val">¢{price_params["mae_backtest"]:.1f}/lb</span></div>
  <div class="perf-row"><span class="perf-key">Directional Accuracy</span><span class="perf-val">{price_params["directional_accuracy"]:.0%} &nbsp;({n_dir}/{n_bt})</span></div>
  <div class="perf-row" style="border:none"><span class="perf-key">Backtest Years</span><span class="perf-val">{n_bt}</span></div>
  <p style="color:#94a3b8;font-size:.7rem;line-height:1.55;margin-top:.8rem;border-top:1px solid #e2e8f0;padding-top:.6rem;">
    <b style="color:#64748b;">R²</b> — price variance explained (0=none, 1=perfect).<br>
    <b style="color:#64748b;">MAE</b> — average absolute error in ¢/lb across test years.<br>
    <b style="color:#64748b;">Directional</b> — % of years the model correctly predicted whether price went up or down.
  </p>
</div>""", unsafe_allow_html=True)
else:
    st.info("Run `python src/price_model.py` to enable the price forecast model.")

gap()

# ── BOTTOM ROW: Outlook | Model Perf | Data Sources ───────────────────────────
b1, b2, b3 = st.columns(3, gap="medium")

with b1:
    _mf_out = price_params.get("multiyear_forecasts", []) if price_params else []
    _out_rows = ""
    for _fc in _mf_out:
        _fc_pp_c = "#22c55e" if _fc["price_pressure"] == "bullish" else "#ef4444" if _fc["price_pressure"] == "bearish" else "#f59e0b"
        _out_rows += f"""
  <div class="outlook-metric">
    <span class="outlook-key">{_fc['year']} Forecast</span>
    <span class="outlook-val">{_fc['predicted_yield']/1e6:.1f}M boxes &nbsp;·&nbsp; <span style="color:{_fc_pp_c};">¢{_fc['predicted_price']:.0f}/lb</span></span>
  </div>"""
    if not _out_rows:
        _out_rows = f"""
  <div class="outlook-metric"><span class="outlook-key">Predicted Yield</span><span class="outlook-val">{latest['yield_boxes']/1e6:.2f}M boxes</span></div>"""
    st.markdown(f"""<div class="panel-card">
  <div class="panel-title">🍊 2025–2027 Outlook</div>
  {_out_rows}
  <div class="outlook-metric"><span class="outlook-key">Market Signal</span><span class="outlook-val" style="color:{'#22c55e' if pressure=='bullish' else '#ef4444' if pressure=='bearish' else '#f59e0b'}">{pressure.capitalize()}</span></div>
  <div class="outlook-metric" style="border:none"><span class="outlook-key">Backtest Accuracy</span><span class="outlook-val">{accuracy:.0%}</span></div>
  <p style="color:#475569;font-size:.7rem;margin-top:1rem;line-height:1.5;">2025–2026 forecasts use actual NDVI; 2027 uses avg of last 3 growing seasons. Model extrapolates FL citrus's HLB disease decline trend — extreme low-yield years drive elevated price forecasts.</p>
</div>""", unsafe_allow_html=True)

with b2:
    st.markdown(f"""<div class="panel-card">
  <div class="panel-title">🌾 Yield Model Performance</div>
  <div class="perf-row"><span class="perf-key">R² Score</span><span class="perf-val">{params['r2_train']:.2f}</span></div>
  <div class="perf-row"><span class="perf-key">MAE (LOO)</span><span class="perf-val">{params['mae_loo']/1e6:.1f}M boxes</span></div>
  <div class="perf-row"><span class="perf-key">Pressure Accuracy</span><span class="perf-val">{accuracy:.0%}</span></div>
  <div class="perf-row" style="border:none"><span class="perf-key">Backtest Years</span><span class="perf-val">{len(backtest)}</span></div>
  <p style="color:#94a3b8;font-size:.7rem;line-height:1.55;margin-top:.8rem;border-top:1px solid #e2e8f0;padding-top:.6rem;">
    <b style="color:#64748b;">R²</b> — yield variance explained by NDVI + year trend.<br>
    <b style="color:#64748b;">MAE</b> — leave-one-out average error in boxes.<br>
    <b style="color:#64748b;">Pressure</b> — rule-based Bullish/Bearish accuracy (yield model only; see Price Model above for trained price prediction accuracy).
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

Starvest uses two separate linear regression models chained together:

**1. Yield Model** (NDVI + year → boxes harvested)
- **Mean NDVI** — average Normalized Difference Vegetation Index over 8 Florida citrus counties during the Oct–May growing season. Higher NDVI indicates healthier vegetation.
- **Year** — captures the long-run decline driven by Huanglongbing (HLB / citrus greening disease), which NDVI alone cannot distinguish from weather effects.

**2. Price Model** (yield_vs_avg + lagged_price → OJ futures price)
- **yield_vs_avg** — this season's predicted yield divided by the long-run historical average. Values below 1.0 signal a supply shortage; the model learned a strongly negative coefficient here.
- **lagged_price** — prior year's average OJ futures price, capturing price momentum and mean-reversion dynamics.
- The model is trained on all available data and backtested walk-forward. The coefficient on yield_vs_avg is approximately –100 ¢/lb per unit of yield ratio, meaning a 50% supply shortfall drives roughly +50 ¢/lb in expected price.

**Price Signal (Bullish / Bearish / Neutral)**
Derived from the price model's predicted % change vs the prior year:
- **Bullish** — model predicts price will rise more than +5% → buy/long signal for OJ futures.
- **Bearish** — model predicts price will fall more than –5% → sell/short signal.
- **Neutral** — predicted change within ±5%.

**Backtest**
Both models use walk-forward validation: each year is predicted using only data from prior years, never future data. The price model additionally requires at least 4 training years before making its first out-of-sample prediction.

**2025–2027 Forecast Pipeline**
- 2025: NDVI (Oct 2024–May 2025, actual satellite data) → Yield Model → yield_vs_avg → Price Model + 2024 actual price → ¢588 predicted.
- 2026: NDVI (Oct 2025–May 2026, actual satellite data) → Yield Model → yield_vs_avg → Price Model + 2025 predicted price (chained).
- 2027: NDVI estimated as mean of 2024–2026 growing seasons (satellite data not yet available) → same chain.

Note: the yield model's strong negative year coefficient (–536K boxes/county/year) reflects the severe HLB disease decline from 2015–2024. Projecting this linear trend to 2026–2027 produces near-zero yield forecasts, which in turn drives the extreme OJ price predictions. These are the model's honest extrapolations; the actual industry trajectory may differ if HLB impact levels off.

**Limitations**
- Small dataset (~10 years of annual data). Results are directionally informative, not investment-grade.
- The year trend in the yield model extrapolates the HLB disease decline, which may be leveling off in reality — resulting in an aggressive low yield_vs_avg for 2025.
- OJ futures are influenced by factors beyond Florida supply (Brazil production, weather, macroeconomics, currency).
- The 2023 price spike following Hurricane Ian was a black swan event the model could not anticipate.
""")

