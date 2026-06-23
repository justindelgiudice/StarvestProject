"""
TerraRisk — Florida Climate Risk Intelligence Dashboard
Satellite base map + continuous IDW-interpolated risk heatmap + county tooltip layer.
"""

import os
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import streamlit as st
from streamlit_folium import st_folium
import requests

RISK_CSV = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "processed", "county_risk_scores.csv")
)
FLORIDA_GEOJSON_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
)

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

# County centroids used as IDW source points for the heatmap
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

ESRI_SATELLITE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

# Weather-radar style gradient: green (low) → yellow → orange → red (critical)
HEATMAP_GRADIENT = {
    0.00: "rgba(0,0,0,0)",
    0.20: "#22CC44",
    0.40: "#AADD00",
    0.55: "#FFDD00",
    0.70: "#FF8800",
    0.85: "#FF3300",
    1.00: "#CC0000",
}


@st.cache_data(ttl=3600)
def load_florida_geojson():
    resp = requests.get(FLORIDA_GEOJSON_URL, timeout=30)
    resp.raise_for_status()
    all_counties = resp.json()
    fips_set = set(FLORIDA_FIPS.values())
    return {
        "type": "FeatureCollection",
        "features": [f for f in all_counties["features"] if f["id"] in fips_set],
    }


@st.cache_data
def load_risk_data():
    df = pd.read_csv(RISK_CSV)
    df["fips"] = df["county"].map(FLORIDA_FIPS)
    return df


@st.cache_data
def generate_heatmap_data(county_scores: tuple) -> list:
    """
    IDW-interpolate county risk scores onto a dense lat/lon grid.
    county_scores: tuple of (county, score) pairs (hashable for caching).
    Returns list of [lat, lon, normalized_weight] for Folium HeatMap.
    """
    src = [(COUNTY_CENTROIDS[c][0], COUNTY_CENTROIDS[c][1], s)
           for c, s in county_scores if c in COUNTY_CENTROIDS]
    src_lats = np.array([p[0] for p in src])
    src_lons = np.array([p[1] for p in src])
    src_scores = np.array([p[2] for p in src])

    # Dense grid over Florida bounding box
    grid_lats = np.arange(24.3, 31.2, 0.07)
    grid_lons = np.arange(-87.9, -79.6, 0.07)

    heat_data = []
    for lat in grid_lats:
        # Vectorized IDW across all longitudes for this latitude row
        for lon in grid_lons:
            d2 = (src_lats - lat) ** 2 + (src_lons - lon) ** 2
            d2 = np.maximum(d2, 1e-8)
            w = 1.0 / d2
            score = float(np.dot(w, src_scores) / w.sum())
            heat_data.append([float(lat), float(lon), min(1.0, score / 10.0)])

    return heat_data


def enrich_geojson(geojson: dict, df: pd.DataFrame) -> dict:
    """Return a new GeoJSON with risk data embedded in feature properties for GeoJsonTooltip."""
    risk_lookup = dict(zip(df["fips"], df["composite_risk_score"]))
    detail = df.set_index("fips").to_dict("index")

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


def build_map(df: pd.DataFrame, geojson: dict) -> folium.Map:
    # ── 1. Satellite base map ──────────────────────────────────────────────────
    m = folium.Map(location=[27.8, -83.5], zoom_start=7, tiles=None)
    folium.TileLayer(
        tiles=ESRI_SATELLITE,
        attr="Esri World Imagery",
        name="Satellite",
        max_zoom=19,
    ).add_to(m)

    # ── 2. Continuous risk heatmap (weather-radar style) ──────────────────────
    county_scores = tuple(zip(df["county"], df["composite_risk_score"]))
    heat_data = generate_heatmap_data(county_scores)
    HeatMap(
        heat_data,
        radius=52,
        blur=42,
        gradient=HEATMAP_GRADIENT,
        min_opacity=0.45,
        max_zoom=12,
    ).add_to(m)

    # ── 3. Transparent county borders + hover tooltips ─────────────────────────
    rich_geojson = enrich_geojson(geojson, df)
    folium.GeoJson(
        rich_geojson,
        style_function=lambda _: {
            "fillColor": "transparent",
            "fillOpacity": 0.0,
            "color": "rgba(255,255,255,0.35)",
            "weight": 1.0,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["County", "Risk Score", "Storms", "Flood Zone", "Sea Level 2100"],
            aliases=["County:", "Risk Score:", "Storms (since 1950):", "FEMA Flood Zone:", "Sea Level Rise 2100:"],
            sticky=True,
            style=(
                "font-family: sans-serif; font-size: 13px;"
                "background: rgba(10,10,10,0.82); color: #fff;"
                "border: none; border-radius: 6px; padding: 8px 12px;"
            ),
        ),
    ).add_to(m)

    # ── 4. Legend ──────────────────────────────────────────────────────────────
    legend_html = """
    <div style="position:fixed;bottom:40px;left:40px;z-index:1000;
                background:rgba(10,10,10,0.78);color:#fff;
                padding:12px 16px;border-radius:8px;
                font-family:sans-serif;font-size:12px;line-height:1.8;">
        <b style="font-size:13px;">Composite Risk Score</b><br>
        <span style="background:#CC0000;padding:2px 10px;border-radius:2px;">&nbsp;</span>&nbsp;8–10&nbsp;&nbsp;Critical<br>
        <span style="background:#FF3300;padding:2px 10px;border-radius:2px;">&nbsp;</span>&nbsp;6–8&nbsp;&nbsp;&nbsp;High<br>
        <span style="background:#FF8800;padding:2px 10px;border-radius:2px;">&nbsp;</span>&nbsp;4–6&nbsp;&nbsp;&nbsp;Elevated<br>
        <span style="background:#FFDD00;padding:2px 10px;border-radius:2px;">&nbsp;</span>&nbsp;2–4&nbsp;&nbsp;&nbsp;Moderate<br>
        <span style="background:#22CC44;padding:2px 10px;border-radius:2px;">&nbsp;</span>&nbsp;0–2&nbsp;&nbsp;&nbsp;Low
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="TerraRisk", layout="wide")
st.title("TerraRisk — Florida Climate Risk Intelligence")
st.caption("Composite county-level hazard exposure for insurance underwriters and real estate investors.")

if not os.path.exists(RISK_CSV):
    st.error(
        "Risk score data not found. Run the data pipeline first:\n\n"
        "```\n"
        "python src/fetch_hurricanes.py\n"
        "python src/fetch_flood_zones.py\n"
        "python src/fetch_sealevel.py\n"
        "python src/build_risk_score.py\n"
        "```"
    )
    st.stop()

df = load_risk_data()
geojson = load_florida_geojson()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Counties Scored", len(df))
col2.metric("Avg Risk Score", f"{df['composite_risk_score'].mean():.2f}")
col3.metric(
    "Highest Risk",
    f"{df.loc[df['composite_risk_score'].idxmax(), 'county']} ({df['composite_risk_score'].max():.1f})",
)
col4.metric("Total Storms", int(df["storm_count"].sum()))

st.subheader("Florida County Composite Risk Map")
risk_map = build_map(df, geojson)
st_folium(risk_map, width=None, height=600)

with st.expander("County Risk Score Table"):
    st.dataframe(
        df[["county", "composite_risk_score", "hurricane_score", "flood_score", "sealevel_score",
            "storm_count", "sfha_pct", "slr_2100_m"]]
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

st.caption("Sources: NOAA HURDAT2 | FEMA NFHL | IPCC AR6 Intermediate Scenario | Esri World Imagery")
