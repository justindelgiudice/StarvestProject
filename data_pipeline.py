"""
Starvest Data Pipeline
======================
Pulls five datasets and merges them into a single CSV:
  1. MODIS NDVI (Jan-Mar average) via Google Earth Engine
  2. Florida orange production + bearing acreage via USDA NASS API
  3. OJ futures April/September close price via yfinance
  4. FL citrus-belt hard freeze flag (Jan-Mar) via NOAA GHCN-Daily
  5. Brazil orange production via USDA FAS PSD bulk download

Output: starvest_data.csv with one row per year (2005-2025)

Run from your project root:
    python data_pipeline.py

Requirements:
    pip install earthengine-api yfinance pandas requests python-dotenv

GEE setup:
    earthengine authenticate
    Set GEE_PROJECT=your-project-id in .env
"""

import io
import os
import time
import zipfile
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


# ── 4. FL Citrus-Belt Hard Freeze Flag via NOAA GHCN-Daily ───────────────────

# Citrus soft-freeze damage threshold (30°F catches more events than 28°F,
# especially the multi-night events that accumulate grove damage)
HARD_FREEZE_F = 30.0  # °F

# NOAA GHCN-D station IDs — FL citrus belt and reliable nearby airports
# Files live at ncei.noaa.gov/pub/data/ghcn/daily/by_station/<ID>.csv.gz
# Mix of cooperative (USC) and airport (USW) stations for coverage through 2025
GHCN_STATIONS = {
    "USC00080228": "Avon Park 2W, Highlands Co.",  # freeze-prone belt, 2005-2021
    "USW00012842": "Tampa Intl Airport",           # active through present, 30 mi W
    "USW00012815": "Orlando McCoy Airport",        # active through present, eastern FL
}

GHCN_BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station"


def get_freeze_data() -> dict:
    """
    Returns {year: {freeze_flag, freeze_days, min_temp_janmar_f}} for each year.

    freeze_flag        – 1 if any station hit ≤ 30°F on any Jan-Mar day, else 0
    freeze_days        – count of days where the cross-station minimum ≤ 30°F
    min_temp_janmar_f  – coldest single reading across all stations and period

    Source: NOAA GHCN-Daily by-station CSV files (no auth required).
    Downloads the full station history once per station; filters in Python.
    TMIN values in GHCN-D are tenths of °C → converted to °F here.
    """
    print("\n[4/4] Fetching NOAA GHCN-D hard freeze data...")

    from collections import defaultdict

    # YYYYMMDD → coldest TMIN (°F) across all stations that day
    belt_daily_min: dict[str, float] = defaultdict(lambda: float("inf"))

    for station_id, label in GHCN_STATIONS.items():
        url = f"{GHCN_BASE}/{station_id}.csv.gz"
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()

            df_st = pd.read_csv(
                io.BytesIO(r.content),
                compression="gzip",
                header=None,
                names=["station", "date", "element", "value",
                       "mflag", "qflag", "sflag", "obs_time"],
                dtype={"date": str, "value": "float64"},
            )

            # TMIN only; blank qflag = passed all quality checks
            tmin = df_st[
                (df_st["element"] == "TMIN") &
                (df_st["qflag"].isna() | (df_st["qflag"].str.strip() == ""))
            ].copy()

            # GHCN value = tenths of °C → convert to °F
            tmin["temp_f"] = tmin["value"] / 10.0 * 9.0 / 5.0 + 32.0

            # Filter to Jan-Mar within our year range
            tmin["year"]  = tmin["date"].str[:4].astype(int)
            tmin["month"] = tmin["date"].str[4:6]
            janmar = tmin[
                (tmin["year"].between(START_YEAR, END_YEAR)) &
                (tmin["month"].isin(["01", "02", "03"]))
            ]

            for _, row in janmar.iterrows():
                belt_daily_min[row["date"]] = min(
                    belt_daily_min[row["date"]], row["temp_f"]
                )

            print(f"  {label}: {len(janmar)} Jan-Mar readings loaded")

        except Exception as e:
            print(f"  {label} ({station_id}): error — {e}")

    if not belt_daily_min:
        print("  ERROR: No GHCN data loaded")
        return {}

    results = {}
    for year in range(START_YEAR, END_YEAR + 1):
        year_str = str(year)
        year_days = {
            d: t for d, t in belt_daily_min.items()
            if d[:4] == year_str and d[4:6] in ("01", "02", "03")
        }
        if not year_days:
            results[year] = {
                "freeze_flag": None, "freeze_days": None, "min_temp_janmar_f": None
            }
            continue

        min_temp    = min(year_days.values())
        freeze_days = sum(1 for t in year_days.values() if t <= HARD_FREEZE_F)
        results[year] = {
            "freeze_flag":       1 if min_temp <= HARD_FREEZE_F else 0,
            "freeze_days":       freeze_days,
            "min_temp_janmar_f": round(min_temp, 1),
        }
        marker = "❄ FREEZE" if min_temp <= HARD_FREEZE_F else "— no freeze"
        print(f"  {year}: min={min_temp:.1f}°F, {freeze_days} day(s) → {marker}")

    return results


# ── 5. Brazil Orange Production via USDA FAS PSD Bulk Download ───────────────

FAS_BULK_URL = "https://apps.fas.usda.gov/psdonline/downloads/psd_alldata_csv.zip"


def get_brazil_orange_production() -> dict:
    """
    Returns {market_year: production_mt} for Brazil fresh oranges 2005-2025.

    Source: USDA FAS PSD (Production, Supply & Distribution) bulk CSV.
    Commodity: 'Oranges, Fresh' (country_code='BR', attribute_id=28).
    Unit: values are in 1000 MT in the source; returned here in metric tons.

    The FAS REST API (/psdonline/api/v1/) is currently unavailable (returns 404),
    so we use the equivalent bulk download which contains the same dataset.
    When multiple revisions exist for a market year, the most recent is kept.
    """
    print("\n[5/5] Fetching Brazil orange production from USDA FAS PSD...")

    try:
        r = requests.get(FAS_BULK_URL, timeout=90, headers={"User-Agent": "Starvest/1.0"})
        r.raise_for_status()

        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open("psd_alldata.csv"))

        prod = df[
            (df["Country_Code"] == "BR") &
            (df["Commodity_Description"] == "Oranges, Fresh") &
            (df["Attribute_ID"] == 28) &
            (df["Market_Year"].between(START_YEAR, END_YEAR))
        ].copy()

        if prod.empty:
            print("  WARNING: No Brazil orange production rows found in FAS bulk data")
            return {}

        # Keep the most recent revision per market year (highest Calendar_Year + Month)
        prod = (
            prod.sort_values(["Calendar_Year", "Month"])
            .groupby("Market_Year")
            .last()
            .reset_index()
        )

        result = {}
        for _, row in prod.iterrows():
            year = int(row["Market_Year"])
            mt   = float(row["Value"]) * 1000  # source is in 1000 MT
            result[year] = mt
            print(f"  {year}: {mt/1e6:.2f}M MT")

        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        return {}


# ── 6. Merge & Compute Yield Surprise Signal ──────────────────────────────────

def build_dataset(ndvi: dict, nass: dict, oj: dict, freeze: dict, brazil: dict) -> pd.DataFrame:
    """
    Merges all five sources and computes:
      ndvi_3yr_avg         – rolling 3-year NDVI baseline (excludes current year)
      ndvi_surprise        – current NDVI minus 3yr avg
      yield_yoy_pct        – YoY % change in FL production
      acres_norm           – bearing acres relative to 2005 baseline
      ndvi_x_acres         – NDVI weighted by grove health proxy
      freeze_flag          – 1 if hard freeze (≤30°F) in Jan-Mar citrus belt
      freeze_days          – count of hard-freeze days in Jan-Mar
      min_temp_janmar_f    – coldest belt temperature in Jan-Mar (°F)
      brazil_production_mt – Brazil annual orange production in metric tons (USDA FAS)
      brazil_yoy_pct       – YoY % change in Brazil production
    """
    print("\n[6/6] Merging datasets and computing signals...")

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

        row["brazil_production_mt"] = brazil.get(year)

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

    df["brazil_yoy_pct"] = (
        df["brazil_production_mt"].pct_change().mul(100).round(2)
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
    print("  Pulling 2005-2025 | MODIS + NASS + OJ Futures + Freeze + Brazil FAS")
    print("=" * 55)

    ndvi   = get_ndvi_time_series()
    nass   = get_nass_data()
    oj     = get_oj_prices()
    freeze = get_freeze_data()
    brazil = get_brazil_orange_production()

    df = build_dataset(ndvi, nass, oj, freeze, brazil)

    df.to_csv(OUTPUT_FILE)
    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"\n{'='*55}")
    print(df.to_string())
    print(f"{'='*55}")

    print("\nData completeness:")
    for col in ["ndvi_jan_mar", "production_boxes", "bearing_acres",
                "apr_close", "sep_close", "price_direction",
                "freeze_flag", "freeze_days", "min_temp_janmar_f",
                "brazil_production_mt", "brazil_yoy_pct"]:
        n_valid = df[col].notna().sum()
        print(f"  {col}: {n_valid}/21 years")


if __name__ == "__main__":
    main()
