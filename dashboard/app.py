"""
TerraRisk — Florida Climate Risk Intelligence Dashboard

Smooth freeform heatmap over Esri satellite imagery driven by actual hazard
event coordinates (not county aggregates).  County polygons are reference only:
thin white outlines with FEMA NRI score popups.

Heatmap layers:
  Overall       All hazard event points combined (equal-weight per hazard)
  Hurricane     HURDAT2 Atlantic track points ≤200 mi of Florida, weight=wind speed
  Tornado       NOAA SPC touchdown points in Florida, weight=EF scale
  Sinkhole      Florida Geological Survey locations, equal weight
  Wildfire      NASA FIRMS VIIRS fire detections, weight=√FRP
  Storm Surge   NOAA SLOSH coastal zone points, weight=Cat 4 surge depth
  Flood         FEMA flood zone distribution (SFHA %), weight=coverage density
"""

import math
import os

import numpy as np
import pandas as pd
import folium
import requests
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RAW  = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")

PATHS = {
    "risk":        os.path.join(PROC, "county_risk_scores.csv"),
    "surge":       os.path.join(RAW,  "storm_surge.csv"),
    "hurricanes":  os.path.join(RAW,  "hurricanes.csv"),
    "tornadoes":   os.path.join(RAW,  "tornadoes.csv"),
    "sinkholes":   os.path.join(RAW,  "sinkholes.csv"),
    "wildfires":   os.path.join(RAW,  "wildfires.csv"),
    "flood_zones": os.path.join(RAW,  "flood_zones.csv"),
}

FL_BOUNDS = [[24.4, -87.7], [31.2, -79.9]]

ESRI_SATELLITE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

# ── Heatmap config ─────────────────────────────────────────────────────────────
HEATMAP_GRADIENT = {0.0: "blue", 0.3: "green", 0.6: "yellow", 0.8: "orange", 1.0: "red"}
HEATMAP_KW       = dict(radius=40, blur=35, min_opacity=0.05, gradient=HEATMAP_GRADIENT)

LAYER_CHOICES = ["Overall", "Hurricane", "Tornado", "Sinkhole", "Wildfire", "Storm Surge", "Flood"]

LAYER_DESC = {
    "Overall":     "All hazard event points combined — density shows multi-hazard composite exposure",
    "Hurricane":   "HURDAT2 Atlantic track points within 200 mi of Florida · weight = wind speed (kt)",
    "Tornado":     "NOAA SPC tornado touchdown points in Florida · weight = EF scale (EF0=low, EF4=high)",
    "Sinkhole":    "Florida Geological Survey (FGS) sinkhole report locations · equal weight",
    "Wildfire":    "NASA FIRMS VIIRS fire detections in Florida · weight = √FRP (fire radiative power)",
    "Storm Surge": "NOAA SLOSH coastal zone point distribution · weight = Cat 4 max surge depth (ft)",
    "Flood":       "Approximate FEMA NFHL flood zone distribution · weight = SFHA coverage %",
}

LAYER_POINT_SOURCES = {
    "Overall":     "HURDAT2 + SPC + FGS + FIRMS + SLOSH + NFHL",
    "Hurricane":   "NOAA HURDAT2",
    "Tornado":     "NOAA SPC",
    "Sinkhole":    "Florida Geological Survey",
    "Wildfire":    "NASA FIRMS VIIRS",
    "Storm Surge": "NOAA NHC SLOSH MEow v4",
    "Flood":       "FEMA NFHL (SFHA % proxy)",
}

# ── County reference data ──────────────────────────────────────────────────────
FLORIDA_FIPS = {
    "Alachua": "12001", "Baker": "12003", "Bay": "12005", "Bradford": "12007",
    "Brevard": "12009", "Broward": "12011", "Calhoun": "12013", "Charlotte": "12015",
    "Citrus": "12017", "Clay": "12019", "Collier": "12021", "Columbia": "12023",
    "DeSoto": "12027", "Dixie": "12029", "Duval": "12031", "Escambia": "12033",
    "Flagler": "12035", "Franklin": "12037", "Gadsden": "12039", "Gilchrist": "12041",
    "Glades": "12043", "Gulf": "12045", "Hamilton": "12047", "Hardee": "12049",
    "Hendry": "12051", "Hernando": "12053", "Highlands": "12055", "Hillsborough": "12057",
    "Holmes": "12059", "Indian River": "12061", "Jackson": "12063", "Jefferson": "12065",
    "Lafayette": "12067", "Lake": "12069", "Lee": "12071", "Leon": "12073",
    "Levy": "12075", "Liberty": "12077", "Madison": "12079", "Manatee": "12081",
    "Marion": "12083", "Martin": "12085", "Miami-Dade": "12086", "Monroe": "12087",
    "Nassau": "12089", "Okaloosa": "12091", "Okeechobee": "12093", "Orange": "12095",
    "Osceola": "12097", "Palm Beach": "12099", "Pasco": "12101", "Pinellas": "12103",
    "Polk": "12105", "Putnam": "12107", "St. Johns": "12109", "St. Lucie": "12111",
    "Santa Rosa": "12113", "Sarasota": "12115", "Seminole": "12117", "Sumter": "12119",
    "Suwannee": "12121", "Taylor": "12123", "Union": "12125", "Volusia": "12127",
    "Wakulla": "12129", "Walton": "12131", "Washington": "12133",
}

COUNTY_CENTROIDS = {
    "Alachua": (29.67, -82.33), "Baker": (30.33, -82.30), "Bay": (30.22, -85.65),
    "Bradford": (29.94, -82.17), "Brevard": (28.26, -80.72), "Broward": (26.07, -80.25),
    "Calhoun": (30.41, -85.20), "Charlotte": (26.95, -82.03), "Citrus": (28.84, -82.50),
    "Clay": (30.00, -81.87), "Collier": (25.90, -81.30), "Columbia": (30.23, -82.62),
    "DeSoto": (27.18, -81.80), "Dixie": (29.58, -83.17), "Duval": (30.37, -81.65),
    "Escambia": (30.61, -87.34), "Flagler": (29.47, -81.27), "Franklin": (29.84, -84.83),
    "Gadsden": (30.58, -84.62), "Gilchrist": (29.72, -82.79), "Glades": (26.96, -81.19),
    "Gulf": (29.92, -85.18), "Hamilton": (30.49, -82.98), "Hardee": (27.49, -81.79),
    "Hendry": (26.50, -81.31), "Hernando": (28.56, -82.46), "Highlands": (27.34, -81.34),
    "Hillsborough": (27.90, -82.35), "Holmes": (30.87, -85.81), "Indian River": (27.70, -80.57),
    "Jackson": (30.72, -85.20), "Jefferson": (30.42, -83.90), "Lafayette": (29.98, -83.20),
    "Lake": (28.77, -81.71), "Lee": (26.54, -81.76), "Leon": (30.46, -84.29),
    "Levy": (29.28, -82.78), "Liberty": (30.24, -84.88), "Madison": (30.47, -83.47),
    "Manatee": (27.47, -82.35), "Marion": (29.21, -82.06), "Martin": (27.07, -80.41),
    "Miami-Dade": (25.55, -80.63), "Monroe": (24.56, -81.36), "Nassau": (30.61, -81.77),
    "Okaloosa": (30.65, -86.51), "Okeechobee": (27.39, -80.90), "Orange": (28.49, -81.26),
    "Osceola": (27.84, -81.11), "Palm Beach": (26.65, -80.30), "Pasco": (28.30, -82.44),
    "Pinellas": (27.88, -82.73), "Polk": (27.94, -81.68), "Putnam": (29.62, -81.74),
    "St. Johns": (29.95, -81.44), "St. Lucie": (27.38, -80.43), "Santa Rosa": (30.68, -86.98),
    "Sarasota": (27.19, -82.37), "Seminole": (28.71, -81.22), "Sumter": (28.71, -82.08),
    "Suwannee": (30.19, -83.00), "Taylor": (30.06, -83.61), "Union": (30.04, -82.37),
    "Volusia": (29.03, -81.18), "Wakulla": (30.10, -84.37), "Walton": (30.58, -86.13),
    "Washington": (30.60, -85.67),
}


# ── Heatmap point loaders ──────────────────────────────────────────────────────

def _spread(lat: float, lon: float, weight: float, n: int, std: float,
            rng: np.random.Generator) -> list:
    """Generate n points with Gaussian jitter around a county centroid."""
    dlat = rng.normal(0.0, std, n)
    dlon = rng.normal(0.0, std, n)
    return [[lat + dlat[i], lon + dlon[i], float(weight)] for i in range(n)]


@st.cache_data(show_spinner=False)
def _hurr() -> list:
    df = pd.read_csv(PATHS["hurricanes"]).dropna(subset=["latitude", "longitude", "wind_knots"])
    df = df.astype({"latitude": float, "longitude": float, "wind_knots": float})
    # ~200 miles around Florida bounding box
    df = df[
        (df["latitude"]  >= 21.0) & (df["latitude"]  <= 34.5) &
        (df["longitude"] >= -91.0) & (df["longitude"] <= -76.0) &
        (df["wind_knots"] > 0)
    ]
    mx = df["wind_knots"].max()
    return [[r.latitude, r.longitude, r.wind_knots / mx] for _, r in df.iterrows()]


@st.cache_data(show_spinner=False)
def _torn() -> list:
    df = pd.read_csv(PATHS["tornadoes"]).dropna(subset=["latitude", "longitude"])
    df = df.astype({"latitude": float, "longitude": float, "ef_scale": int})
    return [[r.latitude, r.longitude, max(0.15, (r.ef_scale + 1) / 5.0)]
            for _, r in df.iterrows()]


@st.cache_data(show_spinner=False)
def _sink() -> list:
    df = pd.read_csv(PATHS["sinkholes"]).dropna(subset=["latitude", "longitude"])
    df = df.astype({"latitude": float, "longitude": float})
    return [[r.latitude, r.longitude, 1.0] for _, r in df.iterrows()]


@st.cache_data(show_spinner=False)
def _fire(max_pts: int = 15_000) -> list:
    df = pd.read_csv(PATHS["wildfires"]).dropna(subset=["latitude", "longitude", "frp"])
    df = df.astype({"latitude": float, "longitude": float, "frp": float})
    df = df[df["frp"] > 0]
    if len(df) > max_pts:
        # Stratified sample — preserve FRP distribution
        df["_q"] = pd.qcut(df["frp"], q=10, duplicates="drop", labels=False)
        df = (
            df.groupby("_q", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), max_pts // 10), random_state=42))
            .reset_index(drop=True)
        )
    mx = df["frp"].max()
    return [[r.latitude, r.longitude, math.sqrt(r.frp / mx)] for _, r in df.iterrows()]


@st.cache_data(show_spinner=False)
def _surge_pts() -> list:
    df = pd.read_csv(PATHS["surge"])
    df = df[df["cat4_ft"].astype(float) > 0]
    rng = np.random.default_rng(seed=42)
    pts: list = []
    for _, r in df.iterrows():
        county = r["county"]
        if county not in COUNTY_CENTROIDS:
            continue
        clat, clon = COUNTY_CENTROIDS[county]
        w = float(r["cat4_ft"]) / 20.0  # max surge = 20 ft (Gulf County Cat 5)
        pts.extend(_spread(clat, clon, w, n=8, std=0.18, rng=rng))
    return pts


@st.cache_data(show_spinner=False)
def _flood_pts() -> list:
    df = pd.read_csv(PATHS["flood_zones"]).astype({"sfha_pct": float})
    rng = np.random.default_rng(seed=7)
    pts: list = []
    for _, r in df.iterrows():
        county = r["county"]
        sfha = float(r["sfha_pct"])
        if sfha <= 0 or county not in COUNTY_CENTROIDS:
            continue
        clat, clon = COUNTY_CENTROIDS[county]
        n_pts = max(2, int(sfha / 2))  # ~1 pt per 2% SFHA coverage
        pts.extend(_spread(clat, clon, sfha / 100.0, n=n_pts, std=0.28, rng=rng))
    return pts


@st.cache_data(show_spinner=False)
def load_all_heatmap_points() -> dict[str, list]:
    """Return {layer_name: [[lat, lon, weight], ...]} for all 7 layers."""
    per_hazard: dict[str, list] = {
        "Hurricane":   _hurr(),
        "Tornado":     _torn(),
        "Sinkhole":    _sink(),
        "Wildfire":    _fire(),
        "Storm Surge": _surge_pts(),
        "Flood":       _flood_pts(),
    }
    # Overall: cap each hazard at 3,000 pts, normalise weights, then combine
    rng = np.random.default_rng(seed=99)
    combined: list = []
    for pts in per_hazard.values():
        if not pts:
            continue
        arr = np.array(pts, dtype=np.float64)
        if len(arr) > 3_000:
            idx = rng.choice(len(arr), size=3_000, replace=False)
            arr  = arr[idx]
        w_max = arr[:, 2].max()
        if w_max > 0:
            arr[:, 2] /= w_max
        combined.extend(arr.tolist())
    return {**per_hazard, "Overall": combined}


# ── NRI / GeoJSON data loaders (for county popup) ──────────────────────────────

@st.cache_data(ttl=3600)
def load_florida_geojson() -> dict:
    fips_set = set(FLORIDA_FIPS.values())
    r = requests.get(
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        timeout=30,
    )
    r.raise_for_status()
    return {
        "type": "FeatureCollection",
        "features": [f for f in r.json()["features"] if f["id"] in fips_set],
    }


@st.cache_data
def load_risk_scores() -> pd.DataFrame:
    df = pd.read_csv(PATHS["risk"])
    df["fips"] = df["county"].map(FLORIDA_FIPS)
    for col in ["eal_score", "risk_score", "hurricane_score", "coastal_flood_score",
                "inland_flood_score", "tornado_score", "wildfire_score", "wind_score",
                "sovi_score", "resl_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


@st.cache_data
def load_surge_data() -> dict:
    if not os.path.exists(PATHS["surge"]):
        return {}
    df = pd.read_csv(PATHS["surge"])
    return {
        r["county"]: {"cat4_ft": int(r.get("cat4_ft", 0) or 0),
                      "cat5_ft": int(r.get("cat5_ft", 0) or 0)}
        for _, r in df.iterrows()
    }


def enrich_geojson(geojson: dict, df_risk: pd.DataFrame, surge_data: dict) -> dict:
    risk_by_fips = {row["fips"]: row.to_dict() for _, row in df_risk.iterrows()}

    def _fmt(d, col, dec=1):
        v = d.get(col)
        try:
            return f"{float(v):.{dec}f}"
        except (TypeError, ValueError):
            return "N/A"

    enriched = []
    for feat in geojson["features"]:
        fips = feat["id"]
        d    = risk_by_fips.get(fips, {})
        county = d.get("county", fips)
        sg = surge_data.get(county, {})
        cat4 = sg.get("cat4_ft", 0)
        surge_label = f"{cat4} ft" if cat4 > 0 else "< 1 ft (inland)"
        enriched.append({
            **feat,
            "properties": {
                "County":               county,
                "EAL Score":            _fmt(d, "eal_score"),
                "EAL Rating":           d.get("eal_rating", ""),
                "Risk Score":           _fmt(d, "risk_score"),
                "Risk Rating":          d.get("risk_rating", ""),
                "Hurricane":            _fmt(d, "hurricane_score"),
                "Coastal Flood":        _fmt(d, "coastal_flood_score"),
                "Inland Flood":         _fmt(d, "inland_flood_score"),
                "Tornado":              _fmt(d, "tornado_score"),
                "Wildfire":             _fmt(d, "wildfire_score"),
                "Strong Wind":          _fmt(d, "wind_score"),
                "Social Vulnerability": _fmt(d, "sovi_score"),
                "Community Resilience": _fmt(d, "resl_score"),
                "Storm Surge (Cat 4)":  surge_label,
            },
        })
    return {"type": "FeatureCollection", "features": enriched}


# ── County outline + popup layer ───────────────────────────────────────────────

def make_county_layer(rich_geojson: dict) -> folium.GeoJson:
    tooltip = folium.GeoJsonTooltip(
        fields=["County", "EAL Score", "Risk Rating"],
        aliases=["County:", "EAL Score:", "NRI Rating:"],
        sticky=True,
        style=(
            "font-family:sans-serif;font-size:12px;"
            "background:rgba(10,10,10,0.85);color:#fff;"
            "border:none;border-radius:5px;padding:6px 10px;"
        ),
    )
    popup = folium.GeoJsonPopup(
        fields=[
            "County",
            "EAL Score", "EAL Rating",
            "Risk Score", "Risk Rating",
            "Hurricane", "Coastal Flood", "Inland Flood",
            "Tornado", "Wildfire", "Strong Wind",
            "Storm Surge (Cat 4)",
            "Social Vulnerability", "Community Resilience",
        ],
        aliases=[
            "County:",
            "EAL Score (0–100):", "EAL Rating:",
            "Risk Score (0–100):", "Risk Rating:",
            "Hurricane:", "Coastal Flooding:", "Inland Flooding:",
            "Tornado:", "Wildfire:", "Strong Wind:",
            "Storm Surge (Cat 4):",
            "Social Vulnerability:", "Community Resilience:",
        ],
        style="font-family:sans-serif;font-size:13px;min-width:260px;max-width:340px;",
        max_width=340,
    )
    return folium.GeoJson(
        rich_geojson,
        style_function=lambda _: {
            "fillColor":   "transparent",
            "fillOpacity": 0.0,
            "color":       "rgba(255,255,255,0.35)",
            "weight":      0.7,
        },
        tooltip=tooltip,
        popup=popup,
    )


# ── Legend ─────────────────────────────────────────────────────────────────────

def add_legend(m: folium.Map, layer: str, n_pts: int) -> None:
    swatches = [
        ("red",    "High density / intensity"),
        ("orange", "Moderate-high"),
        ("yellow", "Moderate"),
        ("green",  "Low-moderate"),
        ("blue",   "Low density / intensity"),
    ]
    rows_html = "".join(
        f'<span style="background:{c};padding:2px 10px;border-radius:2px;'
        f'color:{"#000" if c in ("yellow","green") else "#fff"}">&nbsp;</span>'
        f'&nbsp;{label}<br>'
        for c, label in swatches
    )
    html = f"""
<div style="position:fixed;bottom:40px;left:40px;z-index:1000;
            background:rgba(10,10,10,0.82);color:#fff;
            padding:12px 16px;border-radius:8px;
            font-family:sans-serif;font-size:12px;line-height:2.0;
            pointer-events:none;">
    <b style="font-size:13px;">{layer}</b><br>
    <span style="color:#aaa;font-size:10px;">{n_pts:,} event points · radius=40 blur=35</span><br>
    <hr style="border-color:rgba(255,255,255,0.2);margin:6px 0;">
    {rows_html}
    <span style="color:#888;font-size:10px;">Click county for FEMA NRI scores</span>
</div>
"""
    m.get_root().html.add_child(folium.Element(html))


# ── Map builder ────────────────────────────────────────────────────────────────

def build_map(layer: str, rich_geojson: dict, heatmap_pts: list) -> folium.Map:
    m = folium.Map(location=[27.8, -83.5], zoom_start=7, tiles=None)
    folium.TileLayer(
        tiles=ESRI_SATELLITE, attr="Esri World Imagery",
        name="Satellite", max_zoom=19,
    ).add_to(m)

    if heatmap_pts:
        HeatMap(heatmap_pts, **HEATMAP_KW).add_to(m)

    make_county_layer(rich_geojson).add_to(m)
    add_legend(m, layer, len(heatmap_pts))
    return m


# ── Streamlit app ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="TerraRisk", layout="wide")
st.title("TerraRisk — Florida Climate Risk Intelligence")
st.caption(
    "Heatmap: raw hazard event locations · white lines = county boundaries (click for FEMA NRI scores)"
)

# ── Data load ─────────────────────────────────────────────────────────────────
missing = [k for k, p in PATHS.items() if not os.path.exists(p)]
if missing:
    st.error(
        f"Missing data files: {missing}\n\nRun the full pipeline:\n"
        "```\npython src/fetch_hurricanes.py\npython src/fetch_tornadoes.py\n"
        "python src/fetch_sinkholes.py\npython src/fetch_storm_surge.py\n"
        "python src/build_risk_score.py\n```"
    )
    st.stop()

with st.spinner("Loading hazard event data…"):
    all_pts    = load_all_heatmap_points()
    df_risk    = load_risk_scores()
    surge_data = load_surge_data()
    geojson    = load_florida_geojson()

rich_geojson = enrich_geojson(geojson, df_risk, surge_data)

# ── Metrics row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
top = df_risk.loc[df_risk["risk_score"].idxmax()]
c1.metric("Hurricane Track Pts", f"{len(all_pts['Hurricane']):,}")
c2.metric("Tornado Events",      f"{len(all_pts['Tornado']):,}")
c3.metric("Sinkhole Reports",    f"{len(all_pts['Sinkhole']):,}")
c4.metric("Wildfire Detections", f"{len(all_pts['Wildfire']):,}")
c5.metric("Highest NRI Risk",    f"{top['county']} ({top['risk_score']:.0f})")

# ── Layer toggle ──────────────────────────────────────────────────────────────
layer = st.radio(
    "Hazard Layer", options=LAYER_CHOICES, horizontal=True,
    label_visibility="collapsed",
)

st.caption(f"**{layer}** — {LAYER_DESC[layer]}  |  Source: {LAYER_POINT_SOURCES[layer]}")

# ── Map ───────────────────────────────────────────────────────────────────────
heatmap_pts = all_pts.get(layer, [])
risk_map    = build_map(layer, rich_geojson, heatmap_pts)
st_folium(risk_map, width=None, height=640, returned_objects=[])

# ── Data table ────────────────────────────────────────────────────────────────
with st.expander("FEMA NRI County Score Table (0–100 national percentile)"):
    display_cols = {
        "county":              "County",
        "eal_score":           "EAL Score",
        "eal_rating":          "EAL Rating",
        "risk_score":          "Risk Score",
        "risk_rating":         "Risk Rating",
        "hurricane_score":     "Hurricane",
        "coastal_flood_score": "Coastal Flood",
        "inland_flood_score":  "Inland Flood",
        "tornado_score":       "Tornado",
        "wildfire_score":      "Wildfire",
        "wind_score":          "Strong Wind",
        "sovi_score":          "Social Vuln.",
        "resl_score":          "Resilience",
    }
    show = [c for c in display_cols if c in df_risk.columns]
    st.dataframe(
        df_risk[show]
        .sort_values("eal_score", ascending=False)
        .reset_index(drop=True)
        .rename(columns={c: display_cols[c] for c in show}),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Heatmap: radius=40 blur=35 — density driven by raw event coordinates, not county aggregates. "
    "NRI scores: FEMA National Risk Index v2023 (0–100 national percentile). "
    "Basemap: Esri World Imagery."
)
