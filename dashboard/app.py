"""
TerraRisk — Florida Climate Risk Intelligence Dashboard

Risk data: FEMA National Risk Index (NRI) — official federal hazard risk scores
           https://www.fema.gov/emergency-managers/practitioners/resilience-analysis-and-planning-tool

Scores are NRI percentile scores (0–100). Higher = greater risk nationally.
EAL (Expected Annual Loss) is the primary layer — pure physical exposure, no social adjustment.
RISK (Composite) adjusts EAL by community resilience and social vulnerability.

Raster overlays: smooth RGBA PNGs (scipy IDW + Gaussian smoothed, clipped to FL land).
County polygons are clickable — popup shows full NRI risk breakdown.
"""

import os

import folium
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

PATHS = {
    "risk":   os.path.join(ROOT, "data", "processed", "county_risk_scores.csv"),
    "surge":  os.path.join(ROOT, "data", "raw",       "storm_surge.csv"),
}
IMG_DIR = os.path.join(ROOT, "data", "processed")

FL_BOUNDS = [[24.4, -87.7], [31.2, -79.9]]

# Layer name → PNG filename (Storm Surge uses dynamic surge_cat{n}.png)
LAYER_TO_PNG = {
    "Composite Risk":                 "risk_risk.png",
    "Physical Hazard Exposure (EAL)": "risk_eal.png",
    "Hurricane":                      "risk_hurricane.png",
    "Coastal Flooding":               "risk_coastal_flood.png",
    "Inland Flooding":                "risk_inland_flood.png",
    "Tornado":                        "risk_tornado.png",
    "Wildfire":                       "risk_wildfire.png",
    "Strong Wind":                    "risk_wind.png",
    "Storm Surge":                    "surge_cat4.png",
}

SURGE_PNGS = [f"surge_cat{i}.png" for i in range(1, 6)]

LAYER_CHOICES = list(LAYER_TO_PNG.keys())

LAYER_LABEL = {
    "Composite Risk":                 "FEMA NRI Risk Score — EAL adjusted for social vulnerability & resilience",
    "Physical Hazard Exposure (EAL)": "FEMA NRI EAL Score — pure physical hazard exposure, no social adjustment",
    "Hurricane":            "FEMA NRI HRCN Expected Annual Loss Score (0–100 national percentile)",
    "Coastal Flooding":     "FEMA NRI CFLD Expected Annual Loss Score (storm surge + tidal)",
    "Inland Flooding":      "FEMA NRI IFLD Expected Annual Loss Score (riverine flooding)",
    "Tornado":              "FEMA NRI TRND Expected Annual Loss Score",
    "Wildfire":             "FEMA NRI WFIR Expected Annual Loss Score",
    "Strong Wind":          "FEMA NRI SWND Expected Annual Loss Score",
    "Storm Surge":          "NOAA NHC SLOSH MEow v4 — max potential inundation depth (feet)",
}

# Layer → county_risk_scores.csv score column (for popup display)
LAYER_SCORE_COL = {
    "Composite Risk":                 "risk_score",
    "Physical Hazard Exposure (EAL)": "eal_score",
    "Hurricane":            "hurricane_score",
    "Coastal Flooding":     "coastal_flood_score",
    "Inland Flooding":      "inland_flood_score",
    "Tornado":              "tornado_score",
    "Wildfire":             "wildfire_score",
    "Strong Wind":          "wind_score",
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

ESRI_SATELLITE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

NRI_RATING_COLORS = {
    "Very High":          "#AA0000",
    "Relatively High":    "#FF4400",
    "High":               "#FF4400",
    "Relatively Moderate":"#FF8800",
    "Moderate":           "#FF8800",
    "Relatively Low":     "#FFFF00",
    "Low":                "#00FF88",
    "Very Low":           "#00BBFF",
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
    # Ensure numeric score columns are float
    for col in ["eal_score","risk_score","hurricane_score","coastal_flood_score",
                "inland_flood_score","tornado_score","wildfire_score","wind_score",
                "sovi_score","resl_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


@st.cache_data
def load_surge_data() -> dict:
    path = PATHS["surge"]
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {
        row["county"]: {
            "cat1_ft": int(row.get("cat1_ft", 0) or 0),
            "cat4_ft": int(row.get("cat4_ft", 0) or 0),
            "cat5_ft": int(row.get("cat5_ft", 0) or 0),
        }
        for _, row in df.iterrows()
    }


# ── GeoJSON enrichment ─────────────────────────────────────────────────────────

def enrich_geojson(geojson: dict, df_risk: pd.DataFrame, surge_data: dict) -> dict:
    risk_by_fips = {row["fips"]: row.to_dict() for _, row in df_risk.iterrows()}

    enriched = []
    for feat in geojson["features"]:
        fips = feat["id"]
        d = risk_by_fips.get(fips, {})
        county = d.get("county", fips)
        sg = surge_data.get(county, {})
        cat4 = sg.get("cat4_ft", 0)
        surge_label = f"{cat4} ft" if cat4 > 0 else "< 1 ft (inland)"

        def _fmt(col, decimals=1):
            v = d.get(col)
            return f"{float(v):.{decimals}f}" if v not in (None, "", float("nan")) else "N/A"

        enriched.append({
            **feat,
            "properties": {
                "County":              county,
                "EAL Score":           _fmt("eal_score"),
                "EAL Rating":          d.get("eal_rating", ""),
                "Risk Score":          _fmt("risk_score"),
                "Risk Rating":         d.get("risk_rating", ""),
                "Hurricane":           _fmt("hurricane_score"),
                "Coastal Flood":       _fmt("coastal_flood_score"),
                "Inland Flood":        _fmt("inland_flood_score"),
                "Tornado":             _fmt("tornado_score"),
                "Wildfire":            _fmt("wildfire_score"),
                "Strong Wind":         _fmt("wind_score"),
                "Social Vulnerability":_fmt("sovi_score"),
                "Community Resilience":_fmt("resl_score"),
                "Storm Surge (Cat 4)": surge_label,
                "_eal":                float(d.get("eal_score", 0) or 0),
                "_risk":               float(d.get("risk_score", 0) or 0),
                "_rating":             d.get("eal_rating", ""),
            },
        })
    return {"type": "FeatureCollection", "features": enriched}


# ── Map helpers ────────────────────────────────────────────────────────────────

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
            "color":       "rgba(255,255,255,0.30)",
            "weight":      0.8,
        },
        tooltip=tooltip,
        popup=popup,
    )


def add_legend(m: folium.Map, layer: str, surge_cat: int = 4) -> None:
    if layer == "Storm Surge":
        swatches = [
            ("#00052A", "9+ ft (extreme)"),
            ("#001A6E", "6–9 ft (high)"),
            ("#2070D0", "3–6 ft (moderate)"),
            ("#6DB8F0", "1–3 ft (low)"),
            ("#C8E8FF", "< 1 ft (minimal)"),
        ]
        subtitle = f"Category {surge_cat} — max potential depth (feet)"
    else:
        swatches = [
            ("#AA0000", "Very High (top 10%)"),
            ("#FF4400", "Relatively High"),
            ("#FF8800", "Relatively Moderate"),
            ("#FFFF00", "Relatively Low"),
            ("#00FF88", "Low"),
            ("#00BBFF", "Very Low"),
        ]
        subtitle = LAYER_LABEL[layer]

    rows_html = "".join(
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
    <span style="color:#aaa;font-size:10px;">{subtitle}</span><br>
    <hr style="border-color:rgba(255,255,255,0.2);margin:6px 0;">
    {rows_html}
    <span style="color:#888;font-size:10px;">Click county for full NRI breakdown</span>
</div>
"""
    m.get_root().html.add_child(folium.Element(html))


def build_map(layer: str, rich_geojson: dict, surge_cat: int = 4) -> folium.Map:
    m = folium.Map(location=[27.8, -83.5], zoom_start=7, tiles=None)
    folium.TileLayer(tiles=ESRI_SATELLITE, attr="Esri World Imagery",
                     name="Satellite", max_zoom=19).add_to(m)

    png_name = f"surge_cat{surge_cat}.png" if layer == "Storm Surge" else LAYER_TO_PNG[layer]
    png_path = os.path.join(IMG_DIR, png_name)
    if os.path.exists(png_path):
        folium.raster_layers.ImageOverlay(
            image=png_path, bounds=FL_BOUNDS, opacity=0.65,
            name="risk_overlay", cross_origin=False, zindex=1,
        ).add_to(m)

    make_county_layer(rich_geojson).add_to(m)
    add_legend(m, layer, surge_cat)
    return m


# ── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="TerraRisk", layout="wide")
st.title("TerraRisk — Florida Climate Risk Intelligence")
st.caption(
    "Risk scores: FEMA National Risk Index (NRI) — 0–100 national percentile. "
    "Higher = greater risk relative to all US counties. "
    "Click any county for full breakdown."
)

# Check required files
missing_imgs = [
    f for f in list(LAYER_TO_PNG.values()) + SURGE_PNGS
    if not os.path.exists(os.path.join(IMG_DIR, f))
]
if not os.path.exists(PATHS["risk"]) or missing_imgs:
    st.error(
        "Required data missing. Run the full pipeline:\n\n"
        "```\n"
        "python src/fetch_fema_nri.py\n"
        "python src/build_risk_score.py\n"
        "python src/fetch_storm_surge.py\n"
        "python src/generate_risk_grid.py\n"
        "python src/generate_risk_images.py\n"
        "```"
    )
    st.stop()

df_risk    = load_risk_scores()
surge_data = load_surge_data()
geojson    = load_florida_geojson()
rich_geojson = enrich_geojson(geojson, df_risk, surge_data)

# ── Metrics row ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
top_eal = df_risk.loc[df_risk["eal_score"].idxmax()]
top_risk = df_risk.loc[df_risk["risk_score"].idxmax()]
c1.metric("Counties Scored", len(df_risk))
c2.metric("Highest EAL Score", f"{top_eal['county']} ({top_eal['eal_score']:.0f})")
c3.metric("Highest Risk Score", f"{top_risk['county']} ({top_risk['risk_score']:.0f})")
coastal = sum(1 for v in surge_data.values() if v.get("cat4_ft", 0) > 0)
c4.metric("Surge-Exposed Counties", coastal)
c5.metric("Data Source", "FEMA NRI")

# ── Layer toggle ───────────────────────────────────────────────────────────────
layer = st.radio("**Risk Layer**", options=LAYER_CHOICES, horizontal=True,
                 label_visibility="collapsed")

# ── Storm Surge sub-controls ───────────────────────────────────────────────────
surge_cat = 4
if layer == "Storm Surge":
    col_note, col_cat = st.columns([3, 2])
    with col_cat:
        surge_cat = st.select_slider(
            "Hurricane Category", options=[1, 2, 3, 4, 5], value=4,
            format_func=lambda n: f"Cat {n}",
        )
    with col_note:
        st.info(
            "Maximum potential storm surge depth from NOAA NHC SLOSH MEow v4. "
            "Inland counties appear transparent. Cat 4 is the realistic worst-case "
            "for most Florida coastal counties."
        )
elif layer in ("Physical Hazard Exposure (EAL)", "Composite Risk"):
    diff = (
        "EAL Score — pure physical hazard exposure, no adjustment for community factors."
        if layer == "Physical Hazard Exposure (EAL)"
        else "Risk Score = EAL adjusted downward by high community resilience and "
             "upward by high social vulnerability."
    )
    st.info(f"**{layer}** — FEMA NRI 0–100 national percentile score. {diff}")

# ── Map ────────────────────────────────────────────────────────────────────────
risk_map = build_map(layer, rich_geojson, surge_cat)
st_folium(risk_map, width=None, height=620, returned_objects=[])

# ── Data table ─────────────────────────────────────────────────────────────────
with st.expander("County NRI Score Table (FEMA official data, 0–100 national percentile)"):
    display_cols = {
        "county":               "County",
        "eal_score":            "EAL Score",
        "eal_rating":           "EAL Rating",
        "risk_score":           "Risk Score",
        "risk_rating":          "Risk Rating",
        "hurricane_score":      "Hurricane",
        "coastal_flood_score":  "Coastal Flood",
        "inland_flood_score":   "Inland Flood",
        "tornado_score":        "Tornado",
        "wildfire_score":       "Wildfire",
        "wind_score":           "Strong Wind",
        "sovi_score":           "Social Vuln.",
        "resl_score":           "Resilience",
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
    "Risk scores: FEMA National Risk Index (NRI) v2023 — national percentile scores (0–100). "
    "Storm surge: NOAA NHC SLOSH MEow v4. "
    "Raster overlays: scipy IDW interpolation from county centroids + Gaussian smoothing. "
    "Basemap: Esri World Imagery."
)
