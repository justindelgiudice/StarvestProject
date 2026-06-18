"""
Fetch MODIS NDVI data for Florida citrus belt via Google Earth Engine.
Outputs data/raw/ndvi_raw.csv with columns: date, mean_ndvi
"""

import ee
import pandas as pd
from pathlib import Path

MODIS_COLLECTION = "MODIS/061/MOD13A1"
NDVI_BAND = "NDVI"
NDVI_SCALE = 0.0001  # raw values are scaled by 10000

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "ndvi_raw.csv"


def fetch_ndvi(start_year: int = 2015, end_year: int = 2024) -> pd.DataFrame:
    ee.Initialize(project="starvest")

    # Florida citrus belt — Polk, Highlands, DeSoto counties
    citrus_region = ee.Geometry.Rectangle([-82.0, 27.0, -81.0, 28.2])

    collection = (
        ee.ImageCollection(MODIS_COLLECTION)
        .filterDate(f"{start_year}-01-01", f"{end_year}-12-31")
        .filterBounds(citrus_region)
        .select(NDVI_BAND)
    )

    def image_mean(image):
        mean = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=citrus_region,
            scale=500,
            maxPixels=1e9,
        )
        return ee.Feature(None, {
            "date": image.date().format("YYYY-MM-dd"),
            "mean_ndvi": ee.Number(mean.get(NDVI_BAND)).multiply(NDVI_SCALE),
        })

    features = collection.map(image_mean).getInfo()["features"]
    records = [f["properties"] for f in features]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def main():
    df = fetch_ndvi()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} NDVI records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
