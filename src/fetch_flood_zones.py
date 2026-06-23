"""
Pull FEMA National Flood Hazard Layer (NFHL) flood zone data for all 67 Florida counties.
Queries the FEMA ArcGIS REST API (layer 28: Flood Hazard Zones) to compute the percentage
of each county mapped as Special Flood Hazard Area (SFHA = zones A/V and subtypes).
Saves to data/raw/flood_zones.csv.
"""

import csv
import os
import requests
import time

OUTPUT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "raw", "flood_zones.csv")
)

# Correct FEMA NFHL ArcGIS REST endpoint (layer 28 = Flood Hazard Zones)
FEMA_API = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"

# DFIRM_ID format: {state_fips}{county_fips}{suffix} e.g. "12107C"
# So county Putnam (FIPS 12107) has DFIRM_ID LIKE '12107%'
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


def fetch_sfha_pct(fips: str) -> dict:
    """
    Query FEMA NFHL layer 28 for a county by FIPS.
    Returns total mapped area and SFHA area (SFHA_TF='T'), both in square degrees.
    Ratio is the meaningful output; absolute values depend on projection.
    """
    params = {
        "where": f"DFIRM_ID LIKE '{fips}%'",
        "outStatistics": (
            '[{"statisticType":"sum","onStatisticField":"SHAPE.STArea()",'
            '"outStatisticFieldName":"area"}]'
        ),
        "groupByFieldsForStatistics": "SFHA_TF",
        "returnGeometry": "false",
        "f": "json",
    }
    try:
        resp = requests.get(FEMA_API, params=params, timeout=30)
        resp.raise_for_status()
        features = resp.json().get("features", [])

        sfha_area = 0.0
        total_area = 0.0
        for feat in features:
            attrs = feat.get("attributes", {})
            area = float(attrs.get("area") or 0)
            total_area += area
            if attrs.get("SFHA_TF") == "T":
                sfha_area += area

        pct = round(100.0 * sfha_area / total_area, 2) if total_area > 0 else 0.0
        return {"sfha_area": round(sfha_area, 6), "total_area": round(total_area, 6), "sfha_pct": pct}

    except Exception as e:
        print(f"\n    Warning: API error for FIPS {fips}: {e}")
        return {"sfha_area": None, "total_area": None, "sfha_pct": None}


def main():
    print("Fetching FEMA NFHL flood zone data for 67 Florida counties...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    rows = []
    for county, fips in FLORIDA_COUNTIES.items():
        print(f"  {county} ({fips})...", end=" ", flush=True)
        result = fetch_sfha_pct(fips)
        rows.append({"county": county, "fips": fips, **result})
        pct = result["sfha_pct"]
        print(f"SFHA: {pct}%" if pct is not None else "no data")
        time.sleep(0.2)

    fieldnames = ["county", "fips", "sfha_area", "total_area", "sfha_pct"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if r["sfha_pct"] is not None]
    avg = sum(r["sfha_pct"] for r in valid) / len(valid) if valid else 0
    print(f"\nSaved flood zone data for {len(valid)}/{len(rows)} counties.")
    print(f"Average SFHA coverage: {avg:.1f}%")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
