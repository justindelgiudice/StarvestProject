"""
Pull FEMA National Flood Hazard Layer (NFHL) flood zone data for all 67 Florida counties.
Uses the FEMA NFHL REST API to get Special Flood Hazard Area (SFHA) percentages by county.
Saves to data/raw/flood_zones.csv.
"""

import csv
import os
import requests
import time

OUTPUT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "raw", "flood_zones.csv")
)

# FEMA NFHL REST API — flood hazard zones layer
FEMA_API = (
    "https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer/28/query"
)

FLORIDA_COUNTIES = {
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

# SFHA flood zones (A and V series) — areas with 1% annual flood chance
SFHA_ZONES = {"A", "AE", "AH", "AO", "AR", "A99", "V", "VE"}


def fetch_flood_data_for_county(fips: str) -> dict:
    """Query FEMA NFHL for flood zone areas within a county (by FIPS)."""
    params = {
        "where": f"DFIRM_ID LIKE '{fips[:5]}%'",
        "outFields": "FLD_ZONE,SHAPE_Area",
        "returnGeometry": "false",
        "f": "json",
        "resultRecordCount": 2000,
    }
    try:
        resp = requests.get(FEMA_API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features", [])

        total_area = 0.0
        sfha_area = 0.0
        for feat in features:
            attrs = feat.get("attributes", {})
            zone = str(attrs.get("FLD_ZONE", "")).strip().upper()
            area = float(attrs.get("SHAPE_Area", 0) or 0)
            total_area += area
            if zone in SFHA_ZONES:
                sfha_area += area

        pct = round(100.0 * sfha_area / total_area, 2) if total_area > 0 else 0.0
        return {"sfha_area_sqm": round(sfha_area, 2), "total_area_sqm": round(total_area, 2), "sfha_pct": pct}
    except Exception as e:
        print(f"  Warning: FEMA API error for FIPS {fips}: {e}")
        return {"sfha_area_sqm": None, "total_area_sqm": None, "sfha_pct": None}


def main():
    print("Fetching FEMA flood zone data for 67 Florida counties...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    rows = []
    for county, fips in FLORIDA_COUNTIES.items():
        print(f"  {county} ({fips})...", end=" ", flush=True)
        result = fetch_flood_data_for_county(fips)
        rows.append({"county": county, "fips": fips, **result})
        print(f"SFHA: {result['sfha_pct']}%")
        time.sleep(0.3)  # be polite to the API

    fieldnames = ["county", "fips", "sfha_area_sqm", "total_area_sqm", "sfha_pct"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if r["sfha_pct"] is not None]
    print(f"\nSaved flood zone data for {len(valid)}/{len(rows)} counties to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
