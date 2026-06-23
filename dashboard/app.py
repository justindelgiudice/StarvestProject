"""
TerraRisk — Florida Climate Risk Intelligence Dashboard
Shows a Florida county choropleth colored by composite climate risk score.
"""

import os
import json
import requests
import pandas as pd
import folium
import streamlit as st
from streamlit_folium import st_folium

RISK_CSV = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "processed", "county_risk_scores.csv")
)

FLORIDA_GEOJSON_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
)

# FIPS codes for Florida counties (state FIPS 12)
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


@st.cache_data(ttl=3600)
def load_florida_geojson():
    resp = requests.get(FLORIDA_GEOJSON_URL, timeout=30)
    resp.raise_for_status()
    all_counties = resp.json()
    florida_fips_set = set(FLORIDA_FIPS.values())
    florida_features = [
        f for f in all_counties["features"]
        if f["id"] in florida_fips_set
    ]
    return {"type": "FeatureCollection", "features": florida_features}


@st.cache_data
def load_risk_data():
    df = pd.read_csv(RISK_CSV)
    fips_map = FLORIDA_FIPS
    df["fips"] = df["county"].map(fips_map)
    return df


def risk_color(score: float) -> str:
    if score >= 7.5:
        return "#d73027"
    elif score >= 6.0:
        return "#f46d43"
    elif score >= 4.5:
        return "#fdae61"
    elif score >= 3.0:
        return "#fee090"
    elif score >= 1.5:
        return "#e0f3f8"
    else:
        return "#abd9e9"


def build_map(df: pd.DataFrame, geojson: dict) -> folium.Map:
    risk_lookup = dict(zip(df["fips"], df["composite_risk_score"]))
    detail_lookup = df.set_index("fips").to_dict("index")

    m = folium.Map(location=[27.8, -83.5], zoom_start=7, tiles="CartoDB positron")

    def style_fn(feature):
        fips = feature["id"]
        score = risk_lookup.get(fips, 0)
        return {
            "fillColor": risk_color(score),
            "color": "#555555",
            "weight": 0.8,
            "fillOpacity": 0.75,
        }

    def tooltip_fn(feature):
        fips = feature["id"]
        d = detail_lookup.get(fips, {})
        score = risk_lookup.get(fips, 0)
        county = d.get("county", fips)
        storms = d.get("storm_count", "—")
        flood = d.get("sfha_pct", "—")
        slr = d.get("slr_2100_m", "—")
        return folium.Tooltip(
            f"<b>{county} County</b><br>"
            f"Composite Risk: <b>{score:.1f} / 10</b><br>"
            f"Storms (since 1950): {storms}<br>"
            f"FEMA Flood Zone: {flood}%<br>"
            f"Sea Level Rise (2100): {slr}m",
            sticky=True,
        )

    folium.GeoJson(
        geojson,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=[],
            aliases=[],
        ),
    ).add_to(m)

    # Re-add with custom tooltip by iterating features
    for feature in geojson["features"]:
        fips = feature["id"]
        d = detail_lookup.get(fips, {})
        score = risk_lookup.get(fips, 0)
        county = d.get("county", fips)
        storms = int(d.get("storm_count", 0))
        flood = d.get("sfha_pct", 0)
        slr = d.get("slr_2100_m", 0)

        folium.GeoJson(
            feature,
            style_function=lambda f, s=score: {
                "fillColor": risk_color(s),
                "color": "#555555",
                "weight": 0.8,
                "fillOpacity": 0.75,
            },
            tooltip=folium.Tooltip(
                f"<b>{county} County</b><br>"
                f"Composite Risk: <b>{score:.1f} / 10</b><br>"
                f"Storms since 1950: {storms}<br>"
                f"FEMA Flood Zone: {flood}%<br>"
                f"Sea Level Rise 2100: {slr}m",
                sticky=True,
            ),
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 40px; left: 40px; z-index: 1000; background: white;
                padding: 12px 16px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                font-family: sans-serif; font-size: 13px;">
        <b>Composite Risk Score</b><br>
        <span style="background:#d73027;padding:2px 10px;">&nbsp;</span> 7.5–10 &nbsp;Critical<br>
        <span style="background:#f46d43;padding:2px 10px;">&nbsp;</span> 6.0–7.5 &nbsp;High<br>
        <span style="background:#fdae61;padding:2px 10px;">&nbsp;</span> 4.5–6.0 &nbsp;Elevated<br>
        <span style="background:#fee090;padding:2px 10px;">&nbsp;</span> 3.0–4.5 &nbsp;Moderate<br>
        <span style="background:#e0f3f8;padding:2px 10px;">&nbsp;</span> 1.5–3.0 &nbsp;Low<br>
        <span style="background:#abd9e9;padding:2px 10px;">&nbsp;</span> 0–1.5 &nbsp;&nbsp;&nbsp;Minimal
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ── Streamlit UI ──────────────────────────────────────────────────────────────

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
col3.metric("Highest Risk", f"{df.loc[df['composite_risk_score'].idxmax(), 'county']} ({df['composite_risk_score'].max():.1f})")
col4.metric("Total Storms", int(df["storm_count"].sum()))

st.subheader("Florida County Composite Risk Map")
risk_map = build_map(df, geojson)
st_folium(risk_map, width=None, height=580)

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

st.caption("Sources: NOAA HURDAT2 | FEMA NFHL | IPCC AR6 Intermediate Scenario")
