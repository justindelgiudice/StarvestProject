"""
Starvest Data Pipeline
======================
Pulls four datasets and merges them into a single CSV:
  1. MODIS NDVI (Jan-Mar average) via Google Earth Engine
  2. Florida orange production + bearing acreage via USDA NASS API
  3. OJ futures April/September close price via yfinance
  4. FL citrus-belt hard freeze flag (Jan-Mar) via Open-Meteo archive

Output: starvest_data.csv with one row per year (2005-2025)

Run from your project root:
    python data_pipeline.py

Requirements:
    pip install earthengine-api yfinance pandas requests python-dotenv

GEE setup:
    earthengine authenticate
    Set GEE_PROJECT=your-project-id in .env
"""

import os
import time
import requests
import pandas as pd
import yfinance as yf
import ee
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

START_YEAR  = 2005
END_YEAR    = 2025
GEE_PROJECT = os.getenv("GEE_PROJECT")
NASS_API_KEY = os.getenv("NASS_API_KEY")
NASS_BASE    = "https://quickstats.nass.usda.gov/api/api_GET/"
OUTPUT_FILE  = "starvest_data.csv"


# ── 1. MODIS NDVI via Google Earth Engine ─────────────────────────────────────

def get_ndvi_time_series():
    """
    Returns {year: mean_ndvi} for Jan-Mar of each year.
    Uses MODIS MOD13Q1 16-day composite at 250m over FL citrus belt.
    """
    print("\n[1/4] Fetching MODIS NDVI from Google Earth Engine...")

    if not GEE_PROJECT:
        print("  SKIPPED: set GEE_PROJECT=<your-cloud-project> in .env")
        return {}

    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception as e:
        print(f"  GEE auth error: {e}")
        print("  Run: earthengine authenticate")
        return {}

    # South-central Florida citrus belt
    citrus_belt = ee.Geometry.Rectangle([-82.5, 26.5, -80.0, 28.5])
    ndvi_by_year = {}

    for year in range(START_YEAR, END_YEAR + 1):
        collection = (
            ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterDate(f"{year}-01-01", f"{year}-03-31")
            .filterBounds(citrus_belt)
            .select("NDVI")
        )
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
        time.sleep(0.3)

    return ndvi_by_year


# ── 2. USDA NASS — Orange Production + Bearing Acreage ───────────────────────

def _nass_get(params: dict) -> list:
    params = {"key": NASS_API_KEY, "format": "JSON", **params}
    r = requests.get(NASS_BASE, params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("data", [])


def get_nass_data():
    """
    Returns {year: {production_boxes, bearing_acres}} for 2005-2025.
    Production in boxes (e.g. 149_800_000), acreage in acres (e.g. 541_800).
    Only final-actual rows are used (reference_period_desc == "YEAR").
    """
    print("\n[2/4] Fetching USDA NASS data...")

    if not NASS_API_KEY:
        print("  ERROR: NASS_API_KEY not found in .env")
        return {}

    results = {}

    # ── Production (boxes) ──
    try:
        rows = _nass_get({
            "commodity_desc":        "ORANGES",
            "state_alpha":           "FL",
            "statisticcat_desc":     "PRODUCTION",
            "unit_desc":             "BOXES",
            "class_desc":            "ALL CLASSES",
            "agg_level_desc":        "STATE",
            "reference_period_desc": "YEAR",
            "year__GE":              str(START_YEAR),
            "year__LE":              str(END_YEAR),
        })
        # Multiple programs may have reference_period_desc="YEAR"; keep the max
        # per year (the largest reported figure is the official state total).
        prod_max: dict = {}
        for row in rows:
            if row.get("agg_level_desc", "").upper() != "STATE":
                continue
            year = int(row["year"])
            val_str = row["Value"].replace(",", "").strip()
            if val_str not in ("(D)", "(Z)", "(NA)", ""):
                val = float(val_str)
                if val > prod_max.get(year, 0):
                    prod_max[year] = val
        for year, val in sorted(prod_max.items()):
            results.setdefault(year, {})["production_boxes"] = val
            print(f"  Production {year}: {val/1e6:.2f}M boxes")
    except Exception as e:
        print(f"  Production fetch error: {e}")

    # ── Bearing Acreage ──
    try:
        rows = _nass_get({
            "commodity_desc":        "ORANGES",
            "state_alpha":           "FL",
            "statisticcat_desc":     "AREA BEARING",
            "unit_desc":             "ACRES",
            "class_desc":            "ALL CLASSES",
            "agg_level_desc":        "STATE",
            "reference_period_desc": "YEAR",
            "year__GE":              str(START_YEAR),
            "year__LE":              str(END_YEAR),
        })
        # Census years (2007/2012/2017/2022) return county rows despite the
        # STATE filter. Prefer SURVEY source; fall back to CENSUS if needed.
        acre_survey: dict = {}
        acre_census: dict = {}
        for row in rows:
            if row.get("agg_level_desc", "").upper() != "STATE":
                continue
            year = int(row["year"])
            val_str = row["Value"].replace(",", "").strip()
            if val_str in ("(D)", "(Z)", "(NA)", ""):
                continue
            val = float(val_str)
            if row.get("source_desc", "").upper() == "SURVEY":
                acre_survey[year] = val
            else:
                acre_census[year] = val
        # Merge: survey wins over census
        acre_final = {**acre_census, **acre_survey}
        for year, val in sorted(acre_final.items()):
            results.setdefault(year, {})["bearing_acres"] = val
            print(f"  Bearing acres {year}: {val:,.0f}")
    except Exception as e:
        print(f"  Acreage fetch error: {e}")

    return results


# ── 3. OJ Futures via yfinance ────────────────────────────────────────────────

def get_oj_prices():
    """
    Returns {year: {apr_close, sep_close, price_direction}}
    Uses daily data aggregated to monthly means for better coverage.
    """
    print("\n[3/4] Fetching OJ futures from yfinance...")

    try:
        oj = yf.Ticker("OJ=F")
        hist = oj.history(
            start=f"{START_YEAR}-01-01",
            end=f"{END_YEAR}-10-15",
            interval="1d"
        )

        if hist.empty:
            print("  ERROR: No data returned from yfinance")
            return {}

        hist.index = hist.index.tz_localize(None) if hist.index.tz else hist.index

        # Resample to monthly mean close
        monthly = hist["Close"].resample("ME").mean()

        prices = {}
        for year in range(START_YEAR, END_YEAR + 1):
            year_monthly = monthly[monthly.index.year == year]
            apr = year_monthly[year_monthly.index.month == 4]
            sep = year_monthly[year_monthly.index.month == 9]

            if not apr.empty and not sep.empty:
                apr_close = round(float(apr.iloc[0]), 2)
                sep_close = round(float(sep.iloc[0]), 2)
                direction = 1 if sep_close > apr_close else -1
                prices[year] = {
                    "apr_close":       apr_close,
                    "sep_close":       sep_close,
                    "price_direction": direction,
                }
                arrow = "↑" if direction == 1 else "↓"
                print(f"  {year}: Apr={apr_close:.1f}¢  Sep={sep_close:.1f}¢  {arrow}")
            else:
                print(f"  {year}: missing Apr or Sep data")

        return prices

    except Exception as e:
        print(f"  yfinance error: {e}")
        return {}


# ── 4. FL Citrus-Belt Hard Freeze Flag via Open-Meteo ────────────────────────

# Hard freeze threshold for citrus damage (standard USDA/NASS definition)
HARD_FREEZE_F = 28.0  # °F

# Five monitoring points spanning the FL citrus belt
BELT_POINTS = [
    (27.9, -81.7),  # Polk County (largest citrus county)
    (27.5, -81.3),  # Highlands County
    (27.6, -80.5),  # Indian River County
    (27.4, -80.4),  # St. Lucie County
    (26.6, -81.4),  # Hendry County (southern belt)
]


def get_freeze_data() -> dict:
    """
    Returns {year: {freeze_flag, freeze_days, min_temp_janmar_f}} for each year.

    freeze_flag        – 1 if any belt point hit ≤ 28°F on any Jan-Mar day, else 0
    freeze_days        – number of days where belt minimum ≤ 28°F
    min_temp_janmar_f  – coldest single reading across belt and period

    Source: Open-Meteo Historical Archive (free, no API key).
    Batches the full date range into one request per monitoring point (5 total).
    """
    print("\n[4/4] Fetching hard freeze data from Open-Meteo historical archive...")

    from collections import defaultdict

    # date_str → coldest tmin across all belt points that day
    belt_daily_min: dict[str, float] = defaultdict(lambda: float("inf"))

    for lat, lon in BELT_POINTS:
        try:
            r = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude":         lat,
                    "longitude":        lon,
                    "start_date":       f"{START_YEAR}-01-01",
                    "end_date":         f"{END_YEAR}-03-31",
                    "daily":            "temperature_2m_min",
                    "temperature_unit": "fahrenheit",
                    "timezone":         "America/New_York",
                },
                timeout=30,
            )
            r.raise_for_status()
            payload = r.json()
            for date_str, tmin in zip(
                payload["daily"]["time"],
                payload["daily"]["temperature_2m_min"],
            ):
                if tmin is not None:
                    belt_daily_min[date_str] = min(belt_daily_min[date_str], tmin)
            print(f"  Fetched ({lat}, {lon})")
            time.sleep(0.5)
        except Exception as e:
            print(f"  Open-Meteo error ({lat}, {lon}): {e}")

    if not belt_daily_min:
        print("  ERROR: No data retrieved — check network or Open-Meteo availability")
        return {}

    results = {}
    for year in range(START_YEAR, END_YEAR + 1):
        # Keep only Jan-Mar dates for this year
        year_days = {
            d: t for d, t in belt_daily_min.items()
            if d[:4] == str(year) and d[5:7] in ("01", "02", "03")
        }
        if not year_days:
            results[year] = {"freeze_flag": None, "freeze_days": None, "min_temp_janmar_f": None}
            continue

        min_temp    = min(year_days.values())
        freeze_days = sum(1 for t in year_days.values() if t <= HARD_FREEZE_F)
        results[year] = {
            "freeze_flag":       1 if min_temp <= HARD_FREEZE_F else 0,
            "freeze_days":       freeze_days,
            "min_temp_janmar_f": round(min_temp, 1),
        }
        marker = "❄ FREEZE" if min_temp <= HARD_FREEZE_F else "— no freeze"
        print(f"  {year}: min={min_temp:.1f}°F, {freeze_days} freeze day(s) → {marker}")

    return results


# ── 5. Merge & Compute Yield Surprise Signal ──────────────────────────────────

def build_dataset(ndvi: dict, nass: dict, oj: dict, freeze: dict) -> pd.DataFrame:
    """
    Merges all four sources and computes:
      ndvi_3yr_avg       – rolling 3-year NDVI baseline (excludes current year)
      ndvi_surprise      – current NDVI minus 3yr avg
      yield_yoy_pct      – YoY % change in production
      acres_norm         – bearing acres relative to 2005 baseline
      ndvi_x_acres       – NDVI weighted by grove health proxy
      freeze_flag        – 1 if hard freeze (≤28°F) in Jan-Mar citrus belt
      freeze_days        – count of hard-freeze days in Jan-Mar
      min_temp_janmar_f  – coldest belt temperature in Jan-Mar (°F)
    """
    print("\n[5/5] Merging datasets and computing signals...")

    rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        row = {"year": year}
        row["ndvi_jan_mar"] = ndvi.get(year)

        nass_year = nass.get(year, {})
        row["production_boxes"] = nass_year.get("production_boxes")
        row["bearing_acres"]    = nass_year.get("bearing_acres")

        oj_year = oj.get(year, {})
        row["apr_close"]       = oj_year.get("apr_close")
        row["sep_close"]       = oj_year.get("sep_close")
        row["price_direction"] = oj_year.get("price_direction")

        freeze_year = freeze.get(year, {})
        row["freeze_flag"]       = freeze_year.get("freeze_flag")
        row["freeze_days"]       = freeze_year.get("freeze_days")
        row["min_temp_janmar_f"] = freeze_year.get("min_temp_janmar_f")

        rows.append(row)

    df = pd.DataFrame(rows).set_index("year")

    df["ndvi_3yr_avg"] = (
        df["ndvi_jan_mar"].shift(1)
        .rolling(window=3, min_periods=2)
        .mean()
        .round(4)
    )
    df["ndvi_surprise"] = (df["ndvi_jan_mar"] - df["ndvi_3yr_avg"]).round(4)

    df["yield_yoy_pct"] = (
        df["production_boxes"].pct_change().mul(100).round(2)
    )

    if df["bearing_acres"].notna().any():
        baseline = df["bearing_acres"].iloc[0]
        df["acres_norm"]    = (df["bearing_acres"] / baseline).round(4)
        df["ndvi_x_acres"]  = (df["ndvi_jan_mar"] * df["acres_norm"]).round(4)
    else:
        df["acres_norm"]   = None
        df["ndvi_x_acres"] = None

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  STARVEST DATA PIPELINE")
    print("  Pulling 2005-2025 | MODIS + NASS + OJ Futures + Freeze")
    print("=" * 55)

    ndvi   = get_ndvi_time_series()
    nass   = get_nass_data()
    oj     = get_oj_prices()
    freeze = get_freeze_data()

    df = build_dataset(ndvi, nass, oj, freeze)

    df.to_csv(OUTPUT_FILE)
    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"\n{'='*55}")
    print(df.to_string())
    print(f"{'='*55}")

    print("\nData completeness:")
    for col in ["ndvi_jan_mar", "production_boxes", "bearing_acres",
                "apr_close", "sep_close", "price_direction",
                "freeze_flag", "freeze_days", "min_temp_janmar_f"]:
        n_valid = df[col].notna().sum()
        print(f"  {col}: {n_valid}/21 years")


if __name__ == "__main__":
    main()
