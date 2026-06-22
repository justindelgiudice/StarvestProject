"""
Fetch Florida citrus yield data from USDA NASS API.

NOTE ON COUNTY-LEVEL DATA:
USDA NASS does not publish county-level orange production (boxes). Production
figures are only available at the STATE level (annual survey) and in TONNES
for the Census of Agriculture.  What IS available at county level is:
  - Bearing acres (every 5 years, Census of Agriculture: 2002, 2007, 2012, 2017, 2022)

County production estimates are derived in build_dataset.py by allocating
statewide production proportionally to each county's share of FL bearing
acreage. This is a standard method in agricultural economics.

Outputs
-------
data/raw/yield_raw.csv              — statewide annual production (boxes)
data/raw/yield_county_acres_raw.csv — county bearing acres (Census years)
data/raw/yield_state_acres_raw.csv  — statewide annual bearing acres (for allocation)
"""

import os
import requests
import pandas as pd
from pathlib import Path

NASS_API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
ROOT = Path(__file__).parent.parent

OUTPUT_PATH        = ROOT / "data" / "raw" / "yield_raw.csv"
COUNTY_ACRES_PATH  = ROOT / "data" / "raw" / "yield_county_acres_raw.csv"
STATE_ACRES_PATH   = ROOT / "data" / "raw" / "yield_state_acres_raw.csv"

# NASS county name → canonical form used in NDVI data
NASS_TO_CANONICAL = {
    "DE SOTO":    "DeSoto",
    "HENDRY":     "Hendry",
    "HARDEE":     "Hardee",
    "HIGHLANDS":  "Highlands",
    "CHARLOTTE":  "Charlotte",
    "POLK":       "Polk",
    "GLADES":     "Glades",
    "MANATEE":    "Manatee",
}

NASS_PRODUCTION_PARAMS = {
    "commodity_desc":    "ORANGES",
    "state_alpha":       "FL",
    "statisticcat_desc": "PRODUCTION",
    "unit_desc":         "BOXES",
    "freq_desc":         "ANNUAL",
    "domain_desc":       "TOTAL",
    "class_desc":        "ALL CLASSES",
    "source_desc":       "SURVEY",
    "format":            "JSON",
}

NASS_COUNTY_ACRES_PARAMS = {
    "commodity_desc":    "ORANGES",
    "state_alpha":       "FL",
    "agg_level_desc":    "COUNTY",
    "statisticcat_desc": "AREA BEARING",
    "unit_desc":         "ACRES",
    "format":            "JSON",
}

NASS_STATE_ACRES_PARAMS = {
    "commodity_desc":    "ORANGES",
    "state_alpha":       "FL",
    "agg_level_desc":    "STATE",
    "statisticcat_desc": "AREA BEARING",
    "unit_desc":         "ACRES",
    "class_desc":        "ALL CLASSES",
    "source_desc":       "SURVEY",
    "format":            "JSON",
}


def _get_api_key() -> str:
    key = os.environ.get("NASS_API_KEY")
    if not key:
        raise EnvironmentError("Set NASS_API_KEY in your .env file")
    return key


def _nass_get(params: dict, api_key: str) -> list[dict]:
    resp = requests.get(NASS_API_URL, params={**params, "key": api_key}, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if "error" in result:
        raise RuntimeError(f"NASS API error: {result['error']}")
    return result.get("data", [])


def fetch_yield() -> pd.DataFrame:
    """Statewide Florida orange production (boxes), annual SURVEY."""
    rows = _nass_get(NASS_PRODUCTION_PARAMS, _get_api_key())
    df = pd.DataFrame(rows)[["year", "Value", "load_time"]]
    df["year"] = df["year"].astype(int)
    df["Value"] = pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce")
    df = df.dropna()
    df = df.sort_values("load_time").groupby("year")["Value"].last().reset_index()
    return df.rename(columns={"Value": "yield_boxes"}).sort_values("year").reset_index(drop=True)


def fetch_county_bearing_acres() -> pd.DataFrame:
    """
    County-level orange bearing acres from NASS (Census of Agriculture).
    Available for census years only: 2002, 2007, 2012, 2017, 2022.

    For each county-year, prefers the 'ALL CLASSES' aggregate.  When that is
    suppressed for privacy, falls back to summing the available variety classes
    (MID & NAVEL + VALENCIA) as a lower bound.
    """
    rows = _nass_get(NASS_COUNTY_ACRES_PARAMS, _get_api_key())
    df = pd.DataFrame(rows)
    df["year"]  = df["year"].astype(int)
    df["acres"] = pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce")
    df = df[df["county_name"].isin(NASS_TO_CANONICAL)].copy()

    records = []
    for (nass_name, year), grp in df.groupby(["county_name", "year"]):
        all_cls = grp[grp["class_desc"] == "ALL CLASSES"]["acres"].dropna()
        if not all_cls.empty:
            acres = float(all_cls.iloc[0])
            source = "ALL CLASSES"
        else:
            sub = grp[grp["class_desc"] != "ALL CLASSES"]["acres"].dropna()
            if sub.empty:
                continue
            acres = float(sub.sum())
            source = "sum of " + " + ".join(sorted(grp[grp["class_desc"] != "ALL CLASSES"]["class_desc"].unique()))
        records.append({
            "county":        NASS_TO_CANONICAL[nass_name],
            "county_name_nass": nass_name,
            "year":          year,
            "bearing_acres": acres,
            "acres_source":  source,
        })

    return (pd.DataFrame(records)
            .sort_values(["county", "year"])
            .reset_index(drop=True))


def fetch_state_bearing_acres() -> pd.DataFrame:
    """
    Statewide FL orange bearing acres (SURVEY, annual).
    Used as the denominator when computing county production shares.
    """
    rows = _nass_get(NASS_STATE_ACRES_PARAMS, _get_api_key())
    df = pd.DataFrame(rows)
    df["year"]  = df["year"].astype(int)
    df["acres"] = pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce")
    df = df.dropna(subset=["acres"])
    df = df.sort_values("year").groupby("year")["acres"].last().reset_index()
    return df.rename(columns={"acres": "state_bearing_acres"})


def main():
    api_key = _get_api_key()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching statewide production …")
    ydf = fetch_yield()
    ydf.to_csv(OUTPUT_PATH, index=False)
    print(f"  {len(ydf)} annual yield records → {OUTPUT_PATH}")

    print("Fetching state bearing acres (annual) …")
    sadf = fetch_state_bearing_acres()
    sadf.to_csv(STATE_ACRES_PATH, index=False)
    print(f"  {len(sadf)} state bearing-acre records → {STATE_ACRES_PATH}")

    print("Fetching county bearing acres (Census) …")
    cadf = fetch_county_bearing_acres()
    cadf.to_csv(COUNTY_ACRES_PATH, index=False)
    print(f"  {len(cadf)} county bearing-acre records → {COUNTY_ACRES_PATH}")
    print()

    # ── Verification summary ──────────────────────────────────────────────────
    print("County bearing acres by census year:")
    pivot = cadf.pivot(index="county", columns="year", values="bearing_acres")
    print(pivot.to_string())
    print()
    print("Note: NASS does not publish county-level production (boxes). County yield")
    print("estimates are computed in build_dataset.py as statewide_yield × county_share.")


if __name__ == "__main__":
    main()
