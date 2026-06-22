"""
Fetch Sentinel-2 imagery from Google Earth Engine to detect construction activity
across Florida metro areas.

Strategy:
  - Use Sentinel-2 SR (surface reflectance) L2A, cloud-masked
  - Compute Normalized Difference Built-up Index (NDBI) and Bare Soil Index (BSI)
    over quarterly composites for each metro bounding box
  - Track quarter-over-quarter change in built-up / bare-soil area as a proxy
    for active construction
  - Output: data/raw/satellite_raw.csv
      metro | quarter | ndbi_mean | bsi_mean | built_pct | bare_pct

Auth: earthengine authenticate  (one-time browser flow)

FL metro bounding boxes (lon_min, lat_min, lon_max, lat_max):
  Miami:           -80.68, 25.47, -80.03, 25.97
  Tampa:           -82.77, 27.68, -82.28, 28.08
  Orlando:         -81.60, 28.35, -81.15, 28.75
  Jacksonville:    -81.87, 30.09, -81.45, 30.50
  Fort Lauderdale: -80.31, 26.00, -80.00, 26.30
"""

import ee
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

ROOT   = Path(__file__).parent.parent
OUTPUT = ROOT / "data" / "raw" / "satellite_raw.csv"

FL_METROS = {
    "Miami":           [-80.68, 25.47, -80.03, 25.97],
    "Tampa":           [-82.77, 27.68, -82.28, 28.08],
    "Orlando":         [-81.60, 28.35, -81.15, 28.75],
    "Jacksonville":    [-81.87, 30.09, -81.45, 30.50],
    "Fort Lauderdale": [-80.31, 26.00, -80.00, 26.30],
}

# Quarters to fetch (extend as needed)
START_YEAR = 2018


def _mask_clouds(img):
    qa = img.select("QA60")
    cloud_bit    = 1 << 10
    cirrus_bit   = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return img.updateMask(mask).divide(10000)


def _quarter_composite(metro_name: str, bbox: list, year: int, quarter: int) -> dict:
    month_start = (quarter - 1) * 3 + 1
    start = f"{year}-{month_start:02d}-01"
    end_month = month_start + 3
    end_year  = year + 1 if end_month > 12 else year
    end_month = end_month - 12 if end_month > 12 else end_month
    end = f"{end_year}-{end_month:02d}-01"

    region = ee.Geometry.Rectangle(bbox)
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(_mask_clouds)
    )

    composite = col.median()

    # NDBI = (SWIR - NIR) / (SWIR + NIR)  — positive = built-up
    ndbi = composite.normalizedDifference(["B11", "B8"]).rename("NDBI")
    # BSI  = ((SWIR + Red) - (NIR + Blue)) / ((SWIR + Red) + (NIR + Blue))
    bsi  = composite.expression(
        "((SWIR + Red) - (NIR + Blue)) / ((SWIR + Red) + (NIR + Blue))",
        {"SWIR": composite.select("B11"), "Red": composite.select("B4"),
         "NIR":  composite.select("B8"),  "Blue": composite.select("B2")},
    ).rename("BSI")

    stats = ndbi.addBands(bsi).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=20,
        maxPixels=1e9,
    ).getInfo()

    return {
        "metro":     metro_name,
        "quarter":   f"{year}Q{quarter}",
        "ndbi_mean": stats.get("NDBI"),
        "bsi_mean":  stats.get("BSI"),
    }


def fetch_satellite(start_year: int = START_YEAR) -> pd.DataFrame:
    ee.Initialize()

    rows = []
    current_year  = date.today().year
    current_q     = (date.today().month - 1) // 3 + 1

    for metro, bbox in FL_METROS.items():
        print(f"  {metro}...", flush=True)
        for year in range(start_year, current_year + 1):
            for q in range(1, 5):
                if year == current_year and q >= current_q:
                    break
                try:
                    row = _quarter_composite(metro, bbox, year, q)
                    rows.append(row)
                except Exception as e:
                    print(f"    {year}Q{q} failed: {e}")

    return pd.DataFrame(rows)


def main():
    print("Fetching Sentinel-2 construction indices (requires GEE auth)...")
    df = fetch_satellite()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df)} rows → {OUTPUT}")


if __name__ == "__main__":
    main()
