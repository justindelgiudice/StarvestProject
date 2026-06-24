"""
Build Florida county risk scores from FEMA National Risk Index (NRI) data.

Replaces all custom hazard calculations with official federal risk indices.
Source: FEMA NRI via ArcGIS Feature Service
        https://www.fema.gov/emergency-managers/practitioners/resilience-analysis-and-planning-tool

Scores are NRI percentile scores (0-100). Higher = more risk nationally.

Saves to data/processed/county_risk_scores.csv.
"""

import csv
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RAW  = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")

PATHS = {
    "nri":    os.path.join(RAW,  "fema_nri_counties.csv"),
    "surge":  os.path.join(RAW,  "storm_surge.csv"),
    "output": os.path.join(PROC, "county_risk_scores.csv"),
}

# STCOFIPS → standard county name (matches COUNTY_CENTROIDS keys)
FIPS_TO_COUNTY = {
    "12001": "Alachua",     "12003": "Baker",       "12005": "Bay",
    "12007": "Bradford",    "12009": "Brevard",     "12011": "Broward",
    "12013": "Calhoun",     "12015": "Charlotte",   "12017": "Citrus",
    "12019": "Clay",        "12021": "Collier",     "12023": "Columbia",
    "12027": "DeSoto",      "12029": "Dixie",       "12031": "Duval",
    "12033": "Escambia",    "12035": "Flagler",     "12037": "Franklin",
    "12039": "Gadsden",     "12041": "Gilchrist",   "12043": "Glades",
    "12045": "Gulf",        "12047": "Hamilton",    "12049": "Hardee",
    "12051": "Hendry",      "12053": "Hernando",    "12055": "Highlands",
    "12057": "Hillsborough","12059": "Holmes",      "12061": "Indian River",
    "12063": "Jackson",     "12065": "Jefferson",   "12067": "Lafayette",
    "12069": "Lake",        "12071": "Lee",         "12073": "Leon",
    "12075": "Levy",        "12077": "Liberty",     "12079": "Madison",
    "12081": "Manatee",     "12083": "Marion",      "12085": "Martin",
    "12086": "Miami-Dade",  "12087": "Monroe",      "12089": "Nassau",
    "12091": "Okaloosa",    "12093": "Okeechobee",  "12095": "Orange",
    "12097": "Osceola",     "12099": "Palm Beach",  "12101": "Pasco",
    "12103": "Pinellas",    "12105": "Polk",        "12107": "Putnam",
    "12109": "St. Johns",   "12111": "St. Lucie",   "12113": "Santa Rosa",
    "12115": "Sarasota",    "12117": "Seminole",    "12119": "Sumter",
    "12121": "Suwannee",    "12123": "Taylor",      "12125": "Union",
    "12127": "Volusia",     "12129": "Wakulla",     "12131": "Walton",
    "12133": "Washington",
}

# NRI field → our output column name
NRI_FIELD_MAP = {
    "RISK_SCORE":  "risk_score",
    "RISK_RATNG":  "risk_rating",
    "EAL_SCORE":   "eal_score",
    "EAL_RATNG":   "eal_rating",
    "EAL_VALT":    "eal_annual_loss_usd",
    # Per-hazard Expected Annual Loss scores (pure physical exposure, no social adjustment)
    "HRCN_EALS":   "hurricane_score",
    "CFLD_EALS":   "coastal_flood_score",
    "IFLD_EALS":   "inland_flood_score",
    "TRND_EALS":   "tornado_score",
    "WFIR_EALS":   "wildfire_score",
    "SWND_EALS":   "wind_score",
    # Social context
    "SOVI_SCORE":  "sovi_score",
    "SOVI_RATNG":  "sovi_rating",
    "RESL_SCORE":  "resl_score",
    "RESL_RATNG":  "resl_rating",
}

FIELDNAMES = [
    "county", "stcofips",
    "risk_score", "risk_rating",
    "eal_score", "eal_rating", "eal_annual_loss_usd",
    "hurricane_score", "coastal_flood_score", "inland_flood_score",
    "tornado_score", "wildfire_score", "wind_score",
    "surge_cat4_ft", "surge_cat5_ft",
    "sovi_score", "sovi_rating", "resl_score", "resl_rating",
]


def _float(val, default=0.0) -> float:
    try:
        return float(val) if val not in (None, "", "None", "null") else default
    except (TypeError, ValueError):
        return default


def load_nri_data() -> dict:
    result = {}
    with open(PATHS["nri"], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fips = row.get("STCOFIPS", "").strip()
            county = FIPS_TO_COUNTY.get(fips)
            if not county:
                continue
            mapped = {"county": county, "stcofips": fips}
            for nri_col, our_col in NRI_FIELD_MAP.items():
                mapped[our_col] = _float(row.get(nri_col), 0.0)
                if our_col.endswith("_rating"):
                    mapped[our_col] = row.get(nri_col, "") or ""
            result[county] = mapped
    return result


def load_surge_data() -> dict:
    """Return {county: {cat4_ft, cat5_ft}}."""
    result = {}
    if not os.path.exists(PATHS["surge"]):
        return result
    with open(PATHS["surge"], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            county = row.get("county", "").strip()
            result[county] = {
                "cat4_ft": int(row.get("cat4_ft") or 0),
                "cat5_ft": int(row.get("cat5_ft") or 0),
            }
    return result


def main():
    print("Building county risk scores from FEMA NRI data...")
    nri = load_nri_data()
    surge = load_surge_data()
    print(f"  NRI data loaded: {len(nri)} counties")

    os.makedirs(PROC, exist_ok=True)
    rows = []

    for county, d in nri.items():
        sg = surge.get(county, {"cat4_ft": 0, "cat5_ft": 0})
        row = dict(d)
        row["surge_cat4_ft"] = sg["cat4_ft"]
        row["surge_cat5_ft"] = sg["cat5_ft"]
        # Ensure string rating columns are preserved
        for col in FIELDNAMES:
            if col not in row:
                row[col] = ""
        rows.append(row)

    rows.sort(key=lambda r: float(r.get("eal_score", 0) or 0), reverse=True)

    with open(PATHS["output"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} counties → {PATHS['output']}")
    print("\nTop 15 by EAL Score (FEMA NRI):")
    for r in rows[:15]:
        print(f"  {r['county']:20s}  EAL={float(r['eal_score']):.1f}  RISK={float(r['risk_score']):.1f}  "
              f"HRCN={float(r['hurricane_score']):.1f}  CFLD={float(r['coastal_flood_score']):.1f}  "
              f"rating={r['eal_rating']}")


if __name__ == "__main__":
    main()
