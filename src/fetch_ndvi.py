"""
Pull MODIS NDVI from Google Earth Engine for Florida citrus-belt counties.

Counties: Polk, Highlands, DeSoto, Hardee, Hendry, Charlotte, Glades, Manatee

Outputs
-------
data/raw/ndvi_county_raw.csv  — long format: date, county, geoid, mean_ndvi
data/raw/ndvi_raw.csv         — regional average across counties (backward-compatible)
"""

import ee
import pandas as pd
from pathlib import Path
from datetime import datetime

COUNTY_RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "ndvi_county_raw.csv"
REGIONAL_PATH   = Path(__file__).parent.parent / "data" / "raw" / "ndvi_raw.csv"

# Florida citrus-belt county FIPS codes (state 12 = Florida)
CITRUS_FIPS = [
    "12105",  # Polk
    "12055",  # Highlands
    "12027",  # DeSoto
    "12049",  # Hardee
    "12051",  # Hendry
    "12015",  # Charlotte
    "12043",  # Glades
    "12081",  # Manatee
]

MODIS_COLLECTION = "MODIS/061/MOD13A1"
NDVI_SCALE       = 0.0001
START            = "2015-01-01"
END              = datetime.today().strftime("%Y-%m-%d")


def fetch_ndvi() -> pd.DataFrame:
    ee.Initialize(project="starvest")

    counties_fc = (
        ee.FeatureCollection("TIGER/2018/Counties")
        .filter(ee.Filter.inList("GEOID", CITRUS_FIPS))
    )

    modis = (
        ee.ImageCollection(MODIS_COLLECTION)
        .filterDate(START, END)
        .select("NDVI")
    )

    n          = modis.size().getInfo()
    image_list = modis.toList(n)
    print(f"Processing {n} MODIS images across {len(CITRUS_FIPS)} counties …")

    records = []
    for i in range(n):
        if i % 25 == 0:
            print(f"  {i + 1}/{n} …")

        img     = ee.Image(image_list.get(i))
        date_ms = img.get("system:time_start").getInfo()
        date_str = pd.Timestamp(date_ms, unit="ms").strftime("%Y-%m-%d")

        try:
            stats = img.reduceRegions(
                collection=counties_fc,
                reducer=ee.Reducer.mean(),
                scale=500,
            ).getInfo()["features"]
        except Exception as exc:
            print(f"    Warning: image {i} ({date_str}) skipped — {exc}")
            continue

        for feat in stats:
            p = feat["properties"]
            if p.get("mean") is not None:
                records.append({
                    "date":     date_str,
                    "county":   p.get("NAME", "Unknown"),
                    "geoid":    p.get("GEOID", ""),
                    "mean_ndvi": float(p["mean"]) * NDVI_SCALE,
                })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "county"]).reset_index(drop=True)


def main():
    df = fetch_ndvi()
    COUNTY_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(COUNTY_RAW_PATH, index=False)
    print(f"Saved {len(df)} county records → {COUNTY_RAW_PATH}")
    print(f"Counties : {sorted(df['county'].unique())}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # Regional average for backward compatibility with build_dataset.py
    regional = df.groupby("date")["mean_ndvi"].mean().reset_index()
    regional.to_csv(REGIONAL_PATH, index=False)
    print(f"Saved regional average ({len(regional)} records) → {REGIONAL_PATH}")


if __name__ == "__main__":
    main()
