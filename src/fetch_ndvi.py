"""
Pull MODIS NDVI from Google Earth Engine for Florida citrus-belt counties.

Strategy: for each county we call modis.map(fn).getInfo() ONCE, letting GEE
process all ~270 images server-side.  That gives 8 total network round-trips
instead of 270, so it finishes in ~5 minutes instead of hanging.

Counties: Polk, Highlands, DeSoto, Hardee, Hendry, Charlotte, Glades, Manatee

Outputs
-------
data/raw/ndvi_county_raw.csv  — long format: date, county, geoid, mean_ndvi
data/raw/ndvi_raw.csv         — regional average (backward-compatible)
"""

import ee
import pandas as pd
from pathlib import Path
from datetime import datetime

COUNTY_RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "ndvi_county_raw.csv"
REGIONAL_PATH   = Path(__file__).parent.parent / "data" / "raw" / "ndvi_raw.csv"

# Approximate bounding boxes per county [lon_min, lat_min, lon_max, lat_max].
# Using bboxes avoids a dependency on the TIGER FeatureCollection lookup and
# keeps the closure simple for GEE serialisation.
COUNTY_BBOXES = {
    "Polk":      ("12105", [-82.00, 27.60, -81.20, 28.30]),
    "Highlands": ("12055", [-81.50, 27.00, -80.88, 27.60]),
    "DeSoto":    ("12027", [-81.95, 27.10, -81.45, 27.52]),
    "Hardee":    ("12049", [-82.10, 27.30, -81.45, 27.72]),
    "Hendry":    ("12051", [-81.65, 26.32, -80.88, 26.95]),
    "Charlotte": ("12015", [-82.30, 26.77, -81.73, 27.10]),
    "Glades":    ("12043", [-81.45, 26.68, -80.88, 27.20]),
    "Manatee":   ("12081", [-82.62, 27.39, -82.18, 27.87]),
}

MODIS_COLLECTION = "MODIS/061/MOD13A1"
NDVI_SCALE       = 0.0001
START            = "2015-01-01"
END              = datetime.today().strftime("%Y-%m-%d")


def _fetch_county(modis_coll, geom, geoid: str, name: str) -> list[dict]:
    """
    Fetch NDVI time-series for one county.

    GEE maps the reduction over all images server-side and returns the whole
    result in a single .getInfo() call (one HTTP round-trip per county).
    Falls back to year-by-year chunks if the full request times out.
    """
    def img_to_feat(img, _geom=geom, _geoid=geoid, _name=name):
        mean = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=_geom,
            scale=500,
            maxPixels=1e9,
        ).get("NDVI")
        return ee.Feature(None, {
            "date":  img.date().format("YYYY-MM-dd"),
            "mean":  mean,
            "geoid": _geoid,
            "name":  _name,
        })

    try:
        return modis_coll.map(img_to_feat).getInfo()["features"]
    except Exception as exc:
        print(f"    Full-range failed ({exc}); retrying year-by-year …")
        feats = []
        for yr in range(2015, datetime.today().year + 1):
            try:
                yr_coll = modis_coll.filterDate(f"{yr}-01-01", f"{yr}-12-31")
                feats.extend(yr_coll.map(img_to_feat).getInfo()["features"])
            except Exception as e2:
                print(f"      Year {yr} skipped: {e2}")
        return feats


def fetch_ndvi() -> pd.DataFrame:
    ee.Initialize(project="starvest")

    modis = (
        ee.ImageCollection(MODIS_COLLECTION)
        .filterDate(START, END)
        .select("NDVI")
    )

    n = modis.size().getInfo()
    print(f"{n} MODIS images found ({START} → {END})")
    print(f"Fetching NDVI for {len(COUNTY_BBOXES)} counties …\n")

    records = []
    for county_name, (geoid, bbox) in COUNTY_BBOXES.items():
        print(f"  {county_name} ({geoid}) …", end="", flush=True)
        geom  = ee.Geometry.Rectangle(bbox)
        feats = _fetch_county(modis, geom, geoid, county_name)

        county_records = []
        for feat in feats:
            p = feat["properties"]
            if p.get("mean") is not None:
                county_records.append({
                    "date":     p["date"],
                    "county":   p["name"],
                    "geoid":    p["geoid"],
                    "mean_ndvi": float(p["mean"]) * NDVI_SCALE,
                })
        records.extend(county_records)
        print(f" {len(county_records)} records")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "county"]).reset_index(drop=True)


def main():
    df = fetch_ndvi()

    if df.empty:
        print("ERROR: no data returned — check GEE auth and project name")
        return

    COUNTY_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(COUNTY_RAW_PATH, index=False)
    print(f"\nSaved {len(df)} county-level NDVI records → {COUNTY_RAW_PATH}")

    # Quick sanity check: show per-county range
    summary = df.groupby("county")["mean_ndvi"].agg(["count","min","mean","max"]).round(4)
    print("\nPer-county NDVI summary (all dates):")
    print(summary.to_string())

    # Regional average for backward compatibility with build_dataset.py
    regional = df.groupby("date")["mean_ndvi"].mean().reset_index()
    regional.to_csv(REGIONAL_PATH, index=False)
    print(f"\nRegional average ({len(regional)} records) → {REGIONAL_PATH}")


if __name__ == "__main__":
    main()
