"""
Fetch Florida citrus yield data from USDA NASS API.
Outputs data/raw/yield_raw.csv with columns: year, yield_boxes
"""

import os
import requests
import pandas as pd
from pathlib import Path

NASS_API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "yield_raw.csv"

NASS_PARAMS = {
    "commodity_desc": "ORANGES",
    "state_alpha": "FL",
    "statisticcat_desc": "PRODUCTION",
    "unit_desc": "BOXES",
    "freq_desc": "ANNUAL",
    "domain_desc": "TOTAL",
    "class_desc": "ALL CLASSES",
    "source_desc": "SURVEY",
    "format": "JSON",
}


def fetch_yield() -> pd.DataFrame:
    api_key = os.environ.get("NASS_API_KEY")
    if not api_key:
        raise EnvironmentError("Set NASS_API_KEY in your .env file")

    resp = requests.get(NASS_API_URL, params={**NASS_PARAMS, "key": api_key})
    resp.raise_for_status()
    data = resp.json().get("data", [])

    df = pd.DataFrame(data)[["year", "Value", "load_time"]]
    df["year"] = df["year"].astype(int)
    df["Value"] = pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce")
    df = df.dropna()
    # Take the final (most recently loaded) estimate per year
    df = df.sort_values("load_time").groupby("year")["Value"].last().reset_index()
    df = df.rename(columns={"Value": "yield_boxes"}).sort_values("year").reset_index(drop=True)
    return df


def main():
    df = fetch_yield()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} yield records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
