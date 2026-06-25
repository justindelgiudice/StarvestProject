"""
Starvest Data Pipeline
======================
Pulls three datasets and merges them into a single CSV:
  1. MODIS NDVI (Jan-Mar average) via Google Earth Engine
  2. Florida orange production + bearing acreage via USDA NASS API
  3. OJ futures April close price via yfinance

Output: starvest_data.csv with one row per year (2005-2025)

Run from your project root:
    python data_pipeline.py

Requirements:
    pip install earthengine-api yfinance pandas requests python-dotenv
"""

import os
import time
import json
import requests
import pandas as pd
import yfinance as yf
import ee
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

START_YEAR = 2005
END_YEAR   = 2025

# Florida citrus belt counties (FIPS codes for geometry lookup)
# Polk, Highlands, Hardee, DeSoto, Indian River, St. Lucie, Hendry
CITRUS_COUNTIES = [
    "12105",  # Polk
    "12055",  # Highlands
    "12049",  # Hardee
    "12027",  # DeSoto
    "12061",  # Indian River
    "12111",  # St. Lucie
    "12051",  # Hendry
]

NASS_API_KEY = os.getenv("NASS_API_KEY")
NASS_BASE    = "https://quickstats.nass.usda.gov/api/api_GET/"

OUTPUT_FILE  = "starvest_data.csv"


# ── 1. MODIS NDVI via Google Earth Engine ─────────────────────────────────────

def get_ndvi_time_series():
    """
    Returns a dict {year: mean_ndvi} for Jan-Mar of each year 2005-2025.
    Uses MODIS MOD13Q1 16-day composite at 250m over FL citrus belt counties.
    """
    print("\n[1/3] Fetching MODIS NDVI from Google Earth Engine...")

    try:
        ee.Initialize()
    except Exception as e:
        print(f"  GEE auth error: {e}")
        print("  Run: earthengine authenticate")
        return {}

    # Florida citrus belt bounding box
    # Covers the main growing region in south-central Florida
    citrus_belt = ee.Geometry.Rectangle([-82.5, 26.5, -80.0, 28.5])

    ndvi_by_year = {}

    for year in range(START_YEAR, END_YEAR + 1):
        start = f"{year}-01-01"
        end   = f"{year}-03-31"

        collection = (
            ee.ImageCollection("MODIS/006/MOD13Q1")
            .filterDate(start, end)
            .filterBounds(citrus_belt)
            .select("NDVI")
        )

        # Mean NDVI over the region, scaled by 0.0001 (MODIS scale factor)
        mean_image = collection.mean()
        stats = mean_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=citrus_belt,
            scale=250,
            maxPixels=1e9
        )

        try:
            ndvi_raw = stats.getInfo()["NDVI"]
            ndvi_scaled = ndvi_raw * 0.0001
            ndvi_by_year[year] = round(ndvi_scaled, 4)
            print(f"  {year}: NDVI = {ndvi_scaled:.4f}")
        except Exception as e:
            print(f"  {year}: ERROR - {e}")
            ndvi_by_year[year] = None

        time.sleep(0.3)  # Be polite to GEE

    return ndvi_by_year


# ── 2. USDA NASS — Orange Production + Bearing Acreage ───────────────────────

def get_nass_data():
    """
    Returns a dict {year: {production_boxes, bearing_acres}} for 2005-2025.
    Production in 1000 boxes, acreage in 1000 acres.
    """
    print("\n[2/3] Fetching USDA NASS data...")

    if not NASS_API_KEY:
        print("  ERROR: NASS_API_KEY not found in .env")
        return {}

    results = {}

    # ── Production (1000 boxes) ──
    prod_params = {
        "key": NASS_API_KEY,
        "commodity_desc": "ORANGES",
        "state_alpha": "FL",
        "statisticcat_desc": "PRODUCTION",
        "unit_desc": "1000 BOXES",
        "freq_desc": "ANNUAL",
        "year__GE": str(START_YEAR),
        "year__LE": str(END_YEAR),
        "format": "JSON"
    }

    try:
        r = requests.get(NASS_BASE, params=prod_params, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])

        for row in data:
            year = int(row["year"])
            # Filter to total oranges (not by variety breakdown)
            if row.get("class_desc", "").upper() in ("", "ALL CLASSES"):
                val_str = row["Value"].replace(",", "").strip()
                if val_str not in ("(D)", "(Z)", "(NA)", ""):
                    results.setdefault(year, {})["production_1000_boxes"] = float(val_str)
                    print(f"  Production {year}: {val_str} thousand boxes")

    except Exception as e:
        print(f"  Production fetch error: {e}")

    # ── Bearing Acreage ──
    acre_params = {
        "key": NASS_API_KEY,
        "commodity_desc": "ORANGES",
        "state_alpha": "FL",
        "statisticcat_desc": "AREA BEARING",
        "unit_desc": "ACRES",
        "freq_desc": "ANNUAL",
        "year__GE": str(START_YEAR),
        "year__LE": str(END_YEAR),
        "format": "JSON"
    }

    try:
        r = requests.get(NASS_BASE, params=acre_params, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])

        for row in data:
            year = int(row["year"])
            val_str = row["Value"].replace(",", "").strip()
            if val_str not in ("(D)", "(Z)", "(NA)", ""):
                results.setdefault(year, {})["bearing_acres"] = float(val_str)
                print(f"  Bearing acres {year}: {val_str}")

    except Exception as e:
        print(f"  Acreage fetch error: {e}")

    return results


# ── 3. OJ Futures via yfinance ────────────────────────────────────────────────

def get_oj_prices():
    """
    Returns a dict {year: {apr_close, sep_close, direction}}
    apr_close = April average close (signal entry point)
    sep_close = September average close (signal exit / measurement)
    direction = 1 if price went UP Apr->Sep, -1 if DOWN
    """
    print("\n[3/3] Fetching OJ futures from yfinance...")

    try:
        oj = yf.Ticker("OJ=F")
        hist = oj.history(
            start=f"{START_YEAR}-01-01",
            end=f"{END_YEAR}-10-01",
            interval="1mo"
        )

        if hist.empty:
            print("  ERROR: No data returned from yfinance")
            return {}

        # Remove timezone info for clean indexing
        hist.index = hist.index.tz_localize(None) if hist.index.tz else hist.index

        prices = {}
        for year in range(START_YEAR, END_YEAR + 1):
            year_data = hist[hist.index.year == year]

            apr = year_data[year_data.index.month == 4]["Close"]
            sep = year_data[year_data.index.month == 9]["Close"]

            if not apr.empty and not sep.empty:
                apr_close = round(float(apr.values[0]), 2)
                sep_close = round(float(sep.values[0]), 2)
                direction = 1 if sep_close > apr_close else -1
                prices[year] = {
                    "apr_close": apr_close,
                    "sep_close": sep_close,
                    "price_direction": direction
                }
                arrow = "↑" if direction == 1 else "↓"
                print(f"  {year}: Apr={apr_close:.1f}¢  Sep={sep_close:.1f}¢  {arrow}")
            else:
                print(f"  {year}: missing Apr or Sep data")

        return prices

    except Exception as e:
        print(f"  yfinance error: {e}")
        return {}


# ── 4. Merge & Compute Yield Surprise Signal ──────────────────────────────────

def build_dataset(ndvi, nass, oj):
    """
    Merges all three sources and computes:
      - ndvi_3yr_avg: rolling 3-year NDVI baseline
      - ndvi_surprise: current NDVI minus 3yr avg (relative deviation)
      - yield_surprise: actual production vs prior year (% change)
      - ndvi_x_acres: composite signal (NDVI × bearing acres)
    """
    print("\n[4/4] Merging datasets and computing signals...")

    rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        row = {"year": year}

        # NDVI
        row["ndvi_jan_mar"] = ndvi.get(year)

        # NASS
        nass_year = nass.get(year, {})
        row["production_1000_boxes"] = nass_year.get("production_1000_boxes")
        row["bearing_acres"]         = nass_year.get("bearing_acres")

        # OJ prices
        oj_year = oj.get(year, {})
        row["apr_close"]       = oj_year.get("apr_close")
        row["sep_close"]       = oj_year.get("sep_close")
        row["price_direction"] = oj_year.get("price_direction")

        rows.append(row)

    df = pd.DataFrame(rows).set_index("year")

    # Rolling 3-year NDVI baseline (excludes current year)
    df["ndvi_3yr_avg"] = (
        df["ndvi_jan_mar"]
        .shift(1)
        .rolling(window=3, min_periods=2)
        .mean()
        .round(4)
    )

    # NDVI surprise: how much greener/browner vs recent trend
    df["ndvi_surprise"] = (df["ndvi_jan_mar"] - df["ndvi_3yr_avg"]).round(4)

    # Yield surprise: YoY % change in production
    df["yield_yoy_pct"] = (
        df["production_1000_boxes"]
        .pct_change()
        .mul(100)
        .round(2)
    )

    # Composite signal: NDVI weighted by grove health (bearing acres)
    # Normalize acres to 0-1 range relative to 2005 baseline
    if df["bearing_acres"].notna().any():
        baseline_acres = df["bearing_acres"].iloc[0]
        df["acres_normalized"] = (df["bearing_acres"] / baseline_acres).round(4)
        df["ndvi_x_acres"] = (df["ndvi_jan_mar"] * df["acres_normalized"]).round(4)
    else:
        df["acres_normalized"] = None
        df["ndvi_x_acres"] = None

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  STARVEST DATA PIPELINE")
    print("  Pulling 2005-2025 | MODIS + NASS + OJ Futures")
    print("=" * 55)

    ndvi = get_ndvi_time_series()
    nass = get_nass_data()
    oj   = get_oj_prices()

    df = build_dataset(ndvi, nass, oj)

    df.to_csv(OUTPUT_FILE)
    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"\n{'='*55}")
    print(df.to_string())
    print(f"{'='*55}")

    # Quick data completeness check
    print("\nData completeness:")
    for col in ["ndvi_jan_mar", "production_1000_boxes", "bearing_acres",
                "apr_close", "sep_close", "price_direction"]:
        n_valid = df[col].notna().sum()
        print(f"  {col}: {n_valid}/21 years")


if __name__ == "__main__":
    main()
