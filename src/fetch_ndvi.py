"""
Pull MODIS NDVI from Google Earth Engine for ALL 67 Florida counties.

Strategy: query GEE's TIGER/2018/Counties FeatureCollection to get actual
county boundaries for all 67 FL counties, then fetch NDVI for each with
concurrent workers (4 by default) to keep total runtime under 15 minutes.

Outputs
-------
data/raw/ndvi_county_raw.csv  — long format: date, county, geoid, mean_ndvi
data/raw/ndvi_raw.csv         — regional average (backward-compatible)
"""

import ee
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

COUNTY_RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "ndvi_county_raw.csv"
REGIONAL_PATH   = Path(__file__).parent.parent / "data" / "raw" / "ndvi_raw.csv"

# The 8 primary citrus-belt counties used for yield modelling.
CITRUS_FIPS = {"12105", "12055", "12027", "12049", "12051", "12015", "12043", "12081"}

MODIS_COLLECTION = "MODIS/061/MOD13A1"
NDVI_SCALE       = 0.0001
START            = "2015-01-01"
END              = datetime.today().strftime("%Y-%m-%d")
WORKERS          = 4   # concurrent GEE requests


def get_fl_counties() -> list[dict]:
    """
    Return all 67 FL counties as [{geoid, name, geom}] using GEE TIGER boundaries.

    Only property names are transferred to the client; the geometry object
    stays server-side as a lazy EE expression (resolved when reduceRegion runs).
    """
    fc = ee.FeatureCollection("TIGER/2018/Counties").filter(ee.Filter.eq("STATEFP", "12"))

    # Pull only identifiers — no geometry data sent to Python client
    info = fc.map(lambda f: ee.Feature(None, {
        "GEOID": f.get("GEOID"),
        "NAME":  f.get("NAME"),
    })).getInfo()

    result = []
    for feat in info["features"]:
        props = feat["properties"]
        geoid = str(props["GEOID"]).zfill(5)
        name  = props["NAME"]
        # Server-side geometry reference — never serialised to the client
        geom  = fc.filter(ee.Filter.eq("GEOID", geoid)).first().geometry()
        result.append({"geoid": geoid, "name": name, "geom": geom})

    return sorted(result, key=lambda x: x["name"])


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


def _fetch_county_task(modis_coll, county: dict) -> tuple[str, list[dict]]:
    """Wrapper for ThreadPoolExecutor; returns (name, records)."""
    feats = _fetch_county(modis_coll, county["geom"], county["geoid"], county["name"])
    records = []
    for feat in feats:
        p = feat["properties"]
        if p.get("mean") is not None:
            records.append({
                "date":      p["date"],
                "county":    p["name"],
                "geoid":     p["geoid"],
                "mean_ndvi": float(p["mean"]) * NDVI_SCALE,
            })
    return county["name"], records


def fetch_ndvi() -> pd.DataFrame:
    ee.Initialize(project="starvest")

    modis = (
        ee.ImageCollection(MODIS_COLLECTION)
        .filterDate(START, END)
        .select("NDVI")
    )

    n = modis.size().getInfo()
    print(f"{n} MODIS images found ({START} → {END})")

    print("Loading all 67 FL county geometries from GEE TIGER …")
    counties = get_fl_counties()
    print(f"  Got {len(counties)} counties.")
    print(f"Fetching NDVI with {WORKERS} parallel workers …\n")

    all_records = []
    completed   = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_fetch_county_task, modis, c): c
            for c in counties
        }
        for future in as_completed(futures):
            county = futures[future]
            try:
                name, records = future.result()
                all_records.extend(records)
                completed += 1
                citrus_flag = " ★" if county["geoid"] in CITRUS_FIPS else ""
                print(f"  [{completed:2d}/{len(counties)}] {name} ({county['geoid']}){citrus_flag} … {len(records)} records")
            except Exception as e:
                completed += 1
                print(f"  [{completed:2d}/{len(counties)}] ERROR {county['name']}: {e}")

    df = pd.DataFrame(all_records)
    if df.empty:
        return df
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
    print(f"  Unique counties: {df['county'].nunique()}")

    # Per-county summary
    summary = df.groupby("county")["mean_ndvi"].agg(["count", "min", "mean", "max"]).round(4)
    print(f"\nPer-county NDVI summary ({df['county'].nunique()} counties, all dates):")
    print(summary.to_string())

    # Sample of non-citrus counties to verify spatial variation
    citrus_names = {"Polk", "Highlands", "DeSoto", "Hardee", "Hendry", "Charlotte", "Glades", "Manatee"}
    non_citrus = summary[~summary.index.isin(citrus_names)]
    if not non_citrus.empty:
        print(f"\nSample non-citrus counties ({len(non_citrus)} total):")
        print(non_citrus.head(12).to_string())

    # Regional average for backward compatibility
    regional = df.groupby("date")["mean_ndvi"].mean().reset_index()
    regional.to_csv(REGIONAL_PATH, index=False)
    print(f"\nRegional average ({len(regional)} records) → {REGIONAL_PATH}")


if __name__ == "__main__":
    main()
