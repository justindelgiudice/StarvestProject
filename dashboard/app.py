"""
TerraRisk — Florida Climate Risk Intelligence Dashboard

Layers (radio toggle):
  Overall Risk | Hurricane Tracks | Tornadoes | Sinkholes | Flood Zones | Sea Level Rise

Each layer uses actual event coordinates so the heatmap only lights up
where events occurred — radar-style, with transparent/white space where there is no data.
"""

import os
import pandas as pd
import folium
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import st_folium
import requests

# ── Data paths ─────────────────────────────────────────────────────────────────
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

PATHS = {
    "risk":       os.path.join(ROOT, "data", "processed", "county_risk_scores.csv"),
    "hurricanes": os.path.join(ROOT, "data", "raw", "hurricanes.csv"),
    "tornadoes":  os.path.join(ROOT, "data", "raw", "tornadoes.csv"),
    "sinkholes":  os.path.join(ROOT, "data", "raw", "sinkholes.csv"),
    "flood":      os.path.join(ROOT, "data", "raw", "flood_zones.csv"),
    "sealevel":   os.path.join(ROOT, "data", "raw", "sealevel.csv"),
}

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

# NOAA tide gauge stations: station_id → (lat, lon)
TIDE_GAUGE_COORDS = {
    "8720218": (30.40, -81.43),   # Mayport - Jacksonville
    "8721604": (28.42, -80.59),   # Trident Pier - Cape Canaveral
    "8722670": (26.61, -80.03),   # Lake Worth Pier - West Palm Beach
    "8723214": (25.73, -80.16),   # Virginia Key - Miami
    "8724580": (24.55, -81.81),   # Key West
    "8725520": (26.65, -81.87),   # Fort Myers
    "8726520": (27.77, -82.63),   # St. Petersburg
    "8727520": (29.13, -83.03),   # Cedar Key
    "8728690": (29.73, -84.98),   # Apalachicola
    "8729108": (30.40, -87.21),   # Pensacola
}

ESRI_SATELLITE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

# Radar-style gradient: transparent → blue → cyan → green → yellow → orange → red
RADAR_GRADIENT = {
    0.00: "rgba(0,0,0,0)",
    0.10: "#0000FF",
    0.25: "#00CCFF",
    0.45: "#00FF88",
    0.60: "#FFFF00",
    0.75: "#FF8800",
    0.90: "#FF2200",
    1.00: "#AA0000",
}

LAYER_CHOICES = [
    "Overall Risk",
    "Hurricane Tracks",
    "Tornadoes",
    "Sinkholes",
    "Flood Zones",
    "Sea Level Rise",
]

# HeatMap settings that produce a radar look (NOT a filled blob)
HEATMAP_DEFAULTS = dict(
    radius=18,
    blur=14,
    gradient=RADAR_GRADIENT,
    min_opacity=0.0,
    max_zoom=14,
)


# ── Data loaders ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_florida_geojson():
    resp = requests.get(
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        timeout=30,
    )
    resp.raise_for_status()
    fips_set = set(FLORIDA_FIPS.values())
    return {
        "type": "FeatureCollection",
        "features": [f for f in resp.json()["features"] if f["id"] in fips_set],
    }


@st.cache_data
def load_all_data():
    """Load every raw data file. Returns dict of DataFrames."""
    dfs = {}
    for key, path in PATHS.items():
        if os.path.exists(path):
            dfs[key] = pd.read_csv(path)
        else:
            dfs[key] = pd.DataFrame()
    return dfs


# ── Heatmap point builders ─────────────────────────────────────────────────────

def hurricane_points(df: pd.DataFrame) -> list:
    """
    3,015 individual storm track records from hurricanes.csv.
    Weight = wind_knots / 165 (Cat 5 landfall ≈ 1.0).
    Points cluster along actual historical track corridors.
    """
    pts = []
    for _, r in df.iterrows():
        w = r.get("wind_knots")
        try:
            w = float(w)
        except (TypeError, ValueError):
            w = 0.0
        if w <= 0:
            continue
        pts.append([float(r["latitude"]), float(r["longitude"]), min(1.0, w / 165.0)])
    return pts


def tornado_points(df: pd.DataFrame) -> list:
    """
    NOAA SPC Florida tornado touchdowns.
    Weight = (EF + 0.5) / 5.5  so EF0 shows faintly, EF5 is max.
    """
    if df.empty:
        return []
    pts = []
    for _, r in df.iterrows():
        ef = max(0, int(r.get("ef_scale", 0)))
        weight = (ef + 0.5) / 5.5
        pts.append([float(r["latitude"]), float(r["longitude"]), round(weight, 3)])
    return pts


def sinkhole_points(df: pd.DataFrame) -> list:
    """
    FGS sinkhole locations from county-level statistics.
    Uniform weight 0.7 (presence/absence metric, not magnitude).
    """
    if df.empty:
        return []
    return [[float(r["latitude"]), float(r["longitude"]), 0.7] for _, r in df.iterrows()]


def flood_points(df: pd.DataFrame) -> list:
    """
    FEMA NFHL SFHA percentage per county rendered as heatmap points
    at county centroids. Weight = sfha_pct / 95 (Collier ≈ 95% = max).
    Only counties with real FEMA data (sfha_pct not null) are plotted.
    """
    if df.empty:
        return []
    pts = []
    for _, r in df.iterrows():
        pct = r.get("sfha_pct")
        try:
            pct = float(pct)
        except (TypeError, ValueError):
            continue
        county = r["county"]
        if county not in COUNTY_CENTROIDS:
            continue
        lat, lon = COUNTY_CENTROIDS[county]
        pts.append([lat, lon, min(1.0, pct / 95.0)])
    return pts


def sealevel_points(df: pd.DataFrame) -> list:
    """
    NOAA tide gauge stations along Florida coast.
    Weight = projected_slr_2100_m / 0.85 (Gulf coast max ≈ 0.78m).
    Only the 10 gauge stations with actual coordinates are plotted
    → color appears only along the coastline, not inland.
    """
    if df.empty:
        return []
    pts = []
    for _, r in df.iterrows():
        sid = str(r.get("station_id", "")).strip()
        if sid not in TIDE_GAUGE_COORDS:
            continue
        lat, lon = TIDE_GAUGE_COORDS[sid]
        slr = float(r.get("projected_slr_2100_m", 0))
        pts.append([lat, lon, min(1.0, slr / 0.85)])
    return pts


def combined_points(dfs: dict) -> list:
    """
    Merge all event types into a single point cloud for the Overall Risk layer.
    Each category is weighted by its contribution to the composite risk model
    (hurricane 40%, flood 35%, sea level 25%) plus bonus layers.
    """
    pts = []
    scale = {"hurricane": 0.40, "tornado": 0.30, "sinkhole": 0.25, "flood": 0.35, "sealevel": 0.25}

    for lat, lon, w in hurricane_points(dfs.get("hurricanes", pd.DataFrame())):
        pts.append([lat, lon, w * scale["hurricane"]])
    for lat, lon, w in tornado_points(dfs.get("tornadoes", pd.DataFrame())):
        pts.append([lat, lon, w * scale["tornado"]])
    for lat, lon, w in sinkhole_points(dfs.get("sinkholes", pd.DataFrame())):
        pts.append([lat, lon, w * scale["sinkhole"]])
    for lat, lon, w in flood_points(dfs.get("flood", pd.DataFrame())):
        pts.append([lat, lon, w * scale["flood"] * 4])  # boost: fewer points, need extra density
    for lat, lon, w in sealevel_points(dfs.get("sealevel", pd.DataFrame())):
        pts.append([lat, lon, w * scale["sealevel"] * 6])  # boost: only 10 coastal stations
    return pts


# ── Map builder ────────────────────────────────────────────────────────────────

LAYER_META = {
    "Overall Risk":     {"fn": combined_points,  "radius": 18, "blur": 14, "label": "All risk factors combined"},
    "Hurricane Tracks": {"fn": hurricane_points,  "radius": 14, "blur": 10, "label": "Wind speed (knots), 1950-2025"},
    "Tornadoes":        {"fn": tornado_points,    "radius": 14, "blur": 10, "label": "EF scale, 1950-2023"},
    "Sinkholes":        {"fn": sinkhole_points,   "radius": 10, "blur": 8,  "label": "FGS county reports (synthetic coords)"},
    "Flood Zones":      {"fn": flood_points,      "radius": 28, "blur": 22, "label": "FEMA NFHL SFHA % by county"},
    "Sea Level Rise":   {"fn": sealevel_points,   "radius": 28, "blur": 22, "label": "IPCC AR6 2100 projection, NOAA gauges"},
}


def enrich_geojson(geojson: dict, df_risk: pd.DataFrame) -> dict:
    risk_lookup = dict(zip(df_risk["fips"], df_risk["composite_risk_score"]))
    detail = df_risk.set_index("fips").to_dict("index")
    enriched = []
    for feat in geojson["features"]:
        fips = feat["id"]
        d = detail.get(fips, {})
        score = risk_lookup.get(fips, 0.0)
        enriched.append({
            **feat,
            "properties": {
                "County": d.get("county", fips),
                "Risk Score": f"{score:.1f} / 10",
                "Storms": int(d.get("storm_count", 0)),
                "Flood Zone": f"{d.get('sfha_pct', 0):.1f}%",
                "Sea Level 2100": f"{d.get('slr_2100_m', 0)}m",
            },
        })
    return {"type": "FeatureCollection", "features": enriched}


def build_map(layer_name: str, heat_pts: list, geojson: dict, df_risk: pd.DataFrame) -> folium.Map:
    meta = LAYER_META[layer_name]

    # Satellite base
    m = folium.Map(location=[27.8, -83.5], zoom_start=7, tiles=None)
    folium.TileLayer(
        tiles=ESRI_SATELLITE,
        attr="Esri World Imagery",
        name="Satellite",
        max_zoom=19,
    ).add_to(m)

    # Event-based heatmap (radar style)
    if heat_pts:
        HeatMap(
            heat_pts,
            radius=meta["radius"],
            blur=meta["blur"],
            gradient=RADAR_GRADIENT,
            min_opacity=0.0,
            max_zoom=14,
        ).add_to(m)

    # Thin white county borders + hover tooltips
    rich = enrich_geojson(geojson, df_risk)
    folium.GeoJson(
        rich,
        style_function=lambda _: {
            "fillColor": "transparent",
            "fillOpacity": 0.0,
            "color": "rgba(255,255,255,0.30)",
            "weight": 0.8,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["County", "Risk Score", "Storms", "Flood Zone", "Sea Level 2100"],
            aliases=["County:", "Composite Risk:", "Storms (1950+):", "FEMA SFHA:", "Sea Level 2100:"],
            sticky=True,
            style=(
                "font-family:sans-serif;font-size:13px;"
                "background:rgba(10,10,10,0.85);color:#fff;"
                "border:none;border-radius:6px;padding:8px 12px;"
            ),
        ),
    ).add_to(m)

    # Legend
    legend_html = f"""
    <div style="position:fixed;bottom:40px;left:40px;z-index:1000;
                background:rgba(10,10,10,0.82);color:#fff;
                padding:12px 16px;border-radius:8px;
                font-family:sans-serif;font-size:12px;line-height:1.9;">
        <b style="font-size:13px;">{layer_name}</b><br>
        <span style="color:#aaa;font-size:11px;">{meta['label']}</span><br>
        <hr style="border-color:rgba(255,255,255,0.2);margin:6px 0;">
        <span style="background:#AA0000;padding:2px 10px;border-radius:2px;">&nbsp;</span>&nbsp;Extreme<br>
        <span style="background:#FF2200;padding:2px 10px;border-radius:2px;">&nbsp;</span>&nbsp;High<br>
        <span style="background:#FF8800;padding:2px 10px;border-radius:2px;">&nbsp;</span>&nbsp;Elevated<br>
        <span style="background:#FFFF00;padding:2px 10px;border-radius:2px;color:#000">&nbsp;</span>&nbsp;Moderate<br>
        <span style="background:#00FF88;padding:2px 10px;border-radius:2px;color:#000">&nbsp;</span>&nbsp;Low<br>
        <span style="background:#00CCFF;padding:2px 10px;border-radius:2px;">&nbsp;</span>&nbsp;Minimal<br>
        <span style="color:#aaa;font-size:10px;">No color = no events</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="TerraRisk", layout="wide")
st.title("TerraRisk — Florida Climate Risk Intelligence")
st.caption("Event-based hazard maps for insurance underwriters and real estate investors.")

if not os.path.exists(PATHS["risk"]):
    st.error(
        "Risk score data not found. Run the full pipeline first:\n\n"
        "```\n"
        "python src/fetch_hurricanes.py\n"
        "python src/fetch_flood_zones.py\n"
        "python src/fetch_sealevel.py\n"
        "python src/fetch_tornadoes.py\n"
        "python src/fetch_sinkholes.py\n"
        "python src/build_risk_score.py\n"
        "```"
    )
    st.stop()

dfs = load_all_data()
df_risk = dfs["risk"]
df_risk["fips"] = df_risk["county"].map(FLORIDA_FIPS)

geojson = load_florida_geojson()

# ── Metrics row ────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Counties Scored", len(df_risk))
col2.metric("Avg Risk Score", f"{df_risk['composite_risk_score'].mean():.2f}")
col3.metric("Highest Risk", f"{df_risk.loc[df_risk['composite_risk_score'].idxmax(), 'county']} ({df_risk['composite_risk_score'].max():.1f})")
col4.metric("Hurricane Tracks", f"{len(dfs['hurricanes']):,}" if not dfs["hurricanes"].empty else "—")
col5.metric("Tornado Events", f"{len(dfs['tornadoes']):,}" if not dfs["tornadoes"].empty else "—")

# ── Layer toggle ───────────────────────────────────────────────────────────────
layer = st.radio(
    "**Risk Layer**",
    options=LAYER_CHOICES,
    horizontal=True,
    label_visibility="collapsed",
)

# ── Map ────────────────────────────────────────────────────────────────────────
meta = LAYER_META[layer]
builder_fn = meta["fn"]

if layer == "Overall Risk":
    heat_pts = combined_points(dfs)
else:
    key_map = {
        "Hurricane Tracks": "hurricanes",
        "Tornadoes":        "tornadoes",
        "Sinkholes":        "sinkholes",
        "Flood Zones":      "flood",
        "Sea Level Rise":   "sealevel",
    }
    heat_pts = builder_fn(dfs[key_map[layer]])

risk_map = build_map(layer, heat_pts, geojson, df_risk)
st_folium(risk_map, width=None, height=620)

# ── Data table ─────────────────────────────────────────────────────────────────
with st.expander("County Risk Score Table"):
    st.dataframe(
        df_risk[["county", "composite_risk_score", "hurricane_score", "flood_score",
                 "sealevel_score", "storm_count", "sfha_pct", "slr_2100_m"]]
        .sort_values("composite_risk_score", ascending=False)
        .reset_index(drop=True)
        .rename(columns={
            "composite_risk_score": "Composite (0-10)",
            "hurricane_score": "Hurricane",
            "flood_score": "Flood Zone",
            "sealevel_score": "Sea Level",
            "storm_count": "Storms",
            "sfha_pct": "SFHA %",
            "slr_2100_m": "SLR 2100 (m)",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Sources: NOAA HURDAT2 · NOAA SPC Tornado Database · Florida Geological Survey · "
    "FEMA NFHL · NOAA Tide Gauges · IPCC AR6 · Esri World Imagery"
)
