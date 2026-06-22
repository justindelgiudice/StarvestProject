"""
Fetch residential building permit data from the U.S. Census Bureau
Building Permits Survey (BPS) for Florida metro areas.

API endpoint: https://api.census.gov/data/timeseries/eits/bps
  - Series: PERMITS (total authorized units)
  - Geography: MSA level, filtered to FL metros
  - Frequency: monthly

Note: Census BPS API requires a free API key.
  Register at https://api.census.gov/data/key_signup.html
  Set CENSUS_API_KEY in .env

Outputs data/raw/permits_raw.csv:
  metro | date | permits_total | permits_single | permits_multi
"""

import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

ROOT   = Path(__file__).parent.parent
OUTPUT = ROOT / "data" / "raw" / "permits_raw.csv"

load_dotenv(ROOT / ".env")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")

BPS_URL = "https://api.census.gov/data/timeseries/eits/bps"

# FL MSA FIPS codes for target metros
FL_MSA_FIPS = {
    "Miami":           "33100",  # Miami-Fort Lauderdale-Pompano Beach, FL
    "Tampa":           "45300",  # Tampa-St. Petersburg-Clearwater, FL
    "Orlando":         "36740",  # Orlando-Kissimmee-Sanford, FL
    "Jacksonville":    "27260",  # Jacksonville, FL
    "Fort Lauderdale": "22744",  # Fort Lauderdale-Pompano Beach-Deerfield Beach, FL
}

START_YEAR = 2018


def fetch_permits(start_year: int = START_YEAR) -> pd.DataFrame:
    rows = []
    for metro, fips in FL_MSA_FIPS.items():
        print(f"  {metro} (FIPS {fips})...", flush=True)
        for year in range(start_year, 2027):
            params = {
                "get":        "cell_value,time_slot_id,seasonally_adj",
                "for":        f"metropolitan statistical area/micropolitan statistical area:{fips}",
                "time":       str(year),
                "category_code": "TOTAL",
                "data_type_code": "PERMITS",
            }
            if CENSUS_API_KEY:
                params["key"] = CENSUS_API_KEY
            try:
                resp = requests.get(BPS_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                for record in data[1:]:
                    rows.append({
                        "metro":   metro,
                        "date":    record[1],
                        "permits": record[0],
                    })
            except Exception as e:
                print(f"    {year} failed: {e}")

    df = pd.DataFrame(rows)
    df["date"]    = pd.to_datetime(df["date"])
    df["permits"] = pd.to_numeric(df["permits"], errors="coerce")
    return df.sort_values(["metro", "date"]).reset_index(drop=True)


def main():
    if not CENSUS_API_KEY:
        print("Warning: CENSUS_API_KEY not set in .env — requests may be rate-limited")
    print("Fetching Census building permits...")
    df = fetch_permits()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df)} rows → {OUTPUT}")


if __name__ == "__main__":
    main()
