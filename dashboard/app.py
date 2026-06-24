"""
TerraRisk — Florida Climate Risk Intelligence Dashboard

Layers (radio toggle):
  Overall Risk | Hurricane Tracks | Tornadoes | Sinkholes | Flood Zones | Sea Level Rise | Wildfire

Each layer is a pre-rendered RGBA PNG (scipy KDE + Gaussian smoothed, clipped to
Florida land) displayed as a Folium ImageOverlay at 0.6 opacity over Esri satellite.
Produces a smooth weather-radar appearance at any zoom — no heatmap blobs.
County polygons are clickable — popup shows full risk breakdown.
"""

import os

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

PATHS = {
    "risk":     os.path.join(ROOT, "data", "processed", "county_risk_scores.csv"),
    "flood":    os.path.join(ROOT, "data", "raw",       "flood_zones.csv"),
    "tornadoes":os.path.join(ROOT, "data", "raw",       "tornadoes.csv"),
}
IMG_DIR = os.path.join(ROOT, "data", "processed")

# Florida bounding box — must match generate_risk_images.py
FL_BOUNDS = [[24.4, -87.7], [31.2, -79.9]]

# PNG file for each layer toggle
LAYER_TO_PNG = {
    "Overall Risk":     "risk_overall.png",
    "Hurricane Tracks": "risk_hurricane.png",
    "Tornadoes":        "risk_tornado.png",
    "Sinkholes":        "risk_sinkhole.png",
    "Flood Zones":      "risk_flood.png",
    "Sea Level Rise":   "risk_sealevel.png",
    "Wildfire":         "risk_wildfire.png",
}

# ── Constants ──────────────────────────────────────────────────────────────────
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

ESRI_SATELLITE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

LAYER_CHOICES = [
    "Overall Risk",
    "Hurricane Tracks",
    "Tornadoes",
    "Sinkholes",
    "Flood Zones",
    "Sea Level Rise",
    "Wildfire",
]

LAYER_LABEL = {
    "Overall Risk":     "Composite of all risk factors (5-layer model)",
    "Hurricane Tracks": "Wind speed intensity (knots), 1950–2025",
    "Tornadoes":        "EF scale, NOAA SPC 1950–2023",
    "Sinkholes":        "FGS county reports (synthetic coords)",
    "Flood Zones":      "FEMA NFHL SFHA % by county",
    "Sea Level Rise":   "IPCC AR6 intermediate scenario, 2100",
    "Wildfire":         "NASA FIRMS MODIS + FFS historical (2000–2023)",
}


# ── Data loaders ───────────────────────────────────────────────────────────────

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
    return df


@st.cache_data
def load_flood_zones() -> pd.DataFrame:
    path = PATHS["flood"]
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_data
def compute_tornado_counts(_df_tornadoes: pd.DataFrame) -> dict:
    """Approximate per-county tornado count via nearest county centroid."""
    counts: dict = {c: 0 for c in COUNTY_CENTROIDS}
    if _df_tornadoes.empty:
        return counts
    ctr_lats = np.array([v[0] for v in COUNTY_CENTROIDS.values()], dtype=np.float32)
    ctr_lons = np.array([v[1] for v in COUNTY_CENTROIDS.values()], dtype=np.float32)
    ctr_names = list(COUNTY_CENTROIDS.keys())
    for _, row in _df_tornadoes.iterrows():
        dlat = ctr_lats - float(row["latitude"])
        dlon = ctr_lons - float(row["longitude"])
        idx = int(np.argmin(dlat ** 2 + dlon ** 2))
        counts[ctr_names[idx]] += 1
    return counts


# ── Choropleth helpers ─────────────────────────────────────────────────────────

def sfha_color(pct: float) -> str:
    """Map SFHA% (0-100) to a radar-scale hex color."""
    if pct <= 0:
        return "#1a1a6e"   # near-zero flood risk: deep navy
    if pct < 10:
        return "#0000FF"
    if pct < 25:
        return "#00CCFF"
    if pct < 40:
        return "#00FF88"
    if pct < 55:
        return "#FFFF00"
    if pct < 70:
        return "#FF8800"
    if pct < 85:
        return "#FF2200"
    return "#AA0000"


# ── GeoJSON enrichment ─────────────────────────────────────────────────────────

def enrich_geojson(
    geojson: dict,
    df_risk: pd.DataFrame,
    df_flood: pd.DataFrame,
    tornado_counts: dict,
) -> dict:
    """Embed all popup and style fields into each county feature's properties."""
    risk_by_fips: dict = {}
    for _, row in df_risk.iterrows():
        risk_by_fips[row["fips"]] = row.to_dict()

    flood_by_county: dict = {}
    if not df_flood.empty:
        for _, row in df_flood.iterrows():
            flood_by_county[row["county"]] = float(row.get("sfha_pct") or 0)

    enriched = []
    for feat in geojson["features"]:
        fips = feat["id"]
        d = risk_by_fips.get(fips, {})
        county = d.get("county", fips)
        score = float(d.get("composite_risk_score", 0))
        sfha = flood_by_county.get(county, float(d.get("sfha_pct", 0) or 0))
        slr = float(d.get("slr_2100_m", 0) or 0)
        torns = tornado_counts.get(county, 0)
        fires = int(d.get("fire_count", 0))

        enriched.append({
            **feat,
            "properties": {
                "County":            county,
                "Composite Risk":    f"{score:.1f} / 10",
                "Hurricane Score":   f"{float(d.get('hurricane_score', 0)):.2f}",
                "Flood Zone (SFHA)": f"{sfha:.1f}%",
                "Sea Level 2100":    f"{slr:.2f}m",
                "Storms (1950+)":    int(d.get("storm_count", 0)),
                "Tornadoes":         torns,
                "Wildfire Events":   fires,
                "_sfha_pct":         sfha,
                "_score":            score,
            },
        })
    return {"type": "FeatureCollection", "features": enriched}


# ── Map builder ────────────────────────────────────────────────────────────────

def make_county_layer(rich_geojson: dict, layer: str) -> folium.GeoJson:
    """GeoJson layer providing county borders, tooltip (hover), and popup (click)."""
    is_flood = (layer == "Flood Zones")

    def style_fn(feat):
        if is_flood:
            return {
                "fillColor":   sfha_color(feat["properties"].get("_sfha_pct", 0)),
                "fillOpacity": 0.78,
                "color":       "rgba(255,255,255,0.55)",
                "weight":      0.9,
            }
        return {
            "fillColor":   "transparent",
            "fillOpacity": 0.0,
            "color":       "rgba(255,255,255,0.30)",
            "weight":      0.8,
        }

    tooltip = folium.GeoJsonTooltip(
        fields=["County", "Composite Risk", "Flood Zone (SFHA)"],
        aliases=["County:", "Risk:", "SFHA:"],
        sticky=True,
        style=(
            "font-family:sans-serif;font-size:12px;"
            "background:rgba(10,10,10,0.85);color:#fff;"
            "border:none;border-radius:5px;padding:6px 10px;"
        ),
    )

    popup = folium.GeoJsonPopup(
        fields=[
            "County", "Composite Risk", "Hurricane Score",
            "Flood Zone (SFHA)", "Sea Level 2100", "Storms (1950+)", "Tornadoes", "Wildfire Events",
        ],
        aliases=[
            "County:", "Composite Risk:", "Hurricane Score:",
            "Flood Zone (SFHA):", "Sea Level Rise 2100:", "Storms (1950+):", "Tornadoes:", "Wildfire Events:",
        ],
        style=(
            "font-family:sans-serif;font-size:13px;"
            "min-width:230px;max-width:300px;"
        ),
        max_width=300,
    )

    return folium.GeoJson(
        rich_geojson,
        style_function=style_fn,
        tooltip=tooltip,
        popup=popup,
    )


def add_legend(m: folium.Map, layer: str) -> None:
    if layer == "Flood Zones":
        swatches = [
            ("#AA0000", "85–100% SFHA"),
            ("#FF2200", "70–85%"),
            ("#FF8800", "55–70%"),
            ("#FFFF00", "40–55%"),
            ("#00FF88", "25–40%"),
            ("#00CCFF", "10–25%"),
            ("#0000FF", "0–10%"),
            ("#1a1a6e",  "< 1% / no data"),
        ]
    else:
        swatches = [
            ("#AA0000", "Extreme"),
            ("#FF2200", "High"),
            ("#FF8800", "Elevated"),
            ("#FFFF00", "Moderate"),
            ("#00FF88", "Low"),
            ("#00CCFF", "Minimal"),
        ]

    rows = "".join(
        f'<span style="background:{c};padding:2px 10px;border-radius:2px;'
        f'color:{"#000" if c in ("#FFFF00","#00FF88") else "#fff"}">&nbsp;</span>'
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
    <span style="color:#aaa;font-size:11px;">{LAYER_LABEL[layer]}</span><br>
    <hr style="border-color:rgba(255,255,255,0.2);margin:6px 0;">
    {rows}
    <span style="color:#888;font-size:10px;">Click county for full breakdown</span>
</div>
"""
    m.get_root().html.add_child(folium.Element(html))


def build_map(layer: str, rich_geojson: dict) -> folium.Map:
    m = folium.Map(location=[27.8, -83.5], zoom_start=7, tiles=None)

    folium.TileLayer(
        tiles=ESRI_SATELLITE,
        attr="Esri World Imagery",
        name="Satellite",
        max_zoom=19,
    ).add_to(m)

    # Smooth raster overlay — looks like weather radar, works at any zoom
    png_path = os.path.join(IMG_DIR, LAYER_TO_PNG[layer])
    if os.path.exists(png_path):
        folium.raster_layers.ImageOverlay(
            image=png_path,
            bounds=FL_BOUNDS,
            opacity=0.6,
            name="risk_overlay",
            cross_origin=False,
            zindex=1,
        ).add_to(m)

    # County borders + hover tooltip + click popup (transparent polygon fill)
    make_county_layer(rich_geojson, layer).add_to(m)

    add_legend(m, layer)
    return m


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="TerraRisk", layout="wide")
st.title("TerraRisk — Florida Climate Risk Intelligence")
st.caption("Grid-based hazard maps clipped to Florida land · Click any county for full risk breakdown.")

missing_imgs = [
    f for f in LAYER_TO_PNG.values()
    if not os.path.exists(os.path.join(IMG_DIR, f))
]
if not os.path.exists(PATHS["risk"]) or missing_imgs:
    st.error(
        "Required data missing. Run the full pipeline:\n\n"
        "```\n"
        "python src/fetch_hurricanes.py\n"
        "python src/fetch_flood_zones.py\n"
        "python src/fetch_sealevel.py\n"
        "python src/fetch_tornadoes.py\n"
        "python src/fetch_sinkholes.py\n"
        "python src/build_risk_score.py\n"
        "python src/generate_risk_grid.py\n"
        "python src/generate_risk_images.py\n"
        "```"
    )
    st.stop()

df_risk  = load_risk_scores()
df_flood = load_flood_zones()
geojson  = load_florida_geojson()

df_tornadoes   = pd.read_csv(PATHS["tornadoes"]) if os.path.exists(PATHS["tornadoes"]) else pd.DataFrame()
tornado_counts = compute_tornado_counts(df_tornadoes)

rich_geojson = enrich_geojson(geojson, df_risk, df_flood, tornado_counts)

# ── Metrics row ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Counties Scored", len(df_risk))
c2.metric("Avg Risk Score", f"{df_risk['composite_risk_score'].mean():.2f}")
top_row = df_risk.loc[df_risk["composite_risk_score"].idxmax()]
c3.metric("Highest Risk", f"{top_row['county']} ({top_row['composite_risk_score']:.1f})")
c4.metric("Raster Layers", len(LAYER_TO_PNG))
c5.metric("Tornado Events", f"{len(df_tornadoes):,}" if not df_tornadoes.empty else "—")

# ── Layer toggle ───────────────────────────────────────────────────────────────
layer = st.radio(
    "**Risk Layer**",
    options=LAYER_CHOICES,
    horizontal=True,
    label_visibility="collapsed",
)

# ── Map ────────────────────────────────────────────────────────────────────────
risk_map = build_map(layer, rich_geojson)
st_folium(risk_map, width=None, height=620, returned_objects=[])

# ── Data table ─────────────────────────────────────────────────────────────────
with st.expander("County Risk Score Table"):
    display_cols = {
        "county": "County",
        "composite_risk_score": "Composite (0–10)",
        "hurricane_score": "Hurricane",
        "flood_score": "Flood Zone",
        "sealevel_score": "Sea Level",
        "storm_count": "Storms",
        "sfha_pct": "SFHA %",
        "slr_2100_m": "SLR 2100 (m)",
    }
    show = [c for c in display_cols if c in df_risk.columns]
    st.dataframe(
        df_risk[show]
        .sort_values("composite_risk_score", ascending=False)
        .reset_index(drop=True)
        .rename(columns={c: display_cols[c] for c in show}),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Sources: NOAA HURDAT2 · NOAA SPC Tornado Database · Florida Geological Survey · "
    "FEMA NFHL · NOAA Tide Gauges · IPCC AR6 · NASA FIRMS MODIS · "
    "Florida Forest Service · Esri World Imagery"
)
