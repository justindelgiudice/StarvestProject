"""
Merge NDVI, yield, and price data into modelling datasets.

Outputs
-------
data/processed/dataset.csv             — statewide annual dataset (existing pipeline)
data/processed/ndvi_county_seasonal.csv — per-county seasonal NDVI
data/processed/county_dataset.csv      — county-level panel: NDVI × yield estimate
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED = Path(__file__).parent.parent / "data" / "processed"

OUTPUT_PATH          = PROCESSED / "dataset.csv"
COUNTY_SEASONAL_PATH = PROCESSED / "ndvi_county_seasonal.csv"
COUNTY_DATASET_PATH  = PROCESSED / "county_dataset.csv"

COUNTY_RAW_PATH      = RAW / "ndvi_county_raw.csv"
COUNTY_ACRES_PATH    = RAW / "yield_county_acres_raw.csv"
STATE_ACRES_PATH     = RAW / "yield_state_acres_raw.csv"


# ── Regional / statewide helpers ──────────────────────────────────────────────

def growing_season_ndvi(ndvi_df: pd.DataFrame) -> pd.DataFrame:
    """Average NDVI over Oct–May growing season, labelled by harvest year."""
    df = ndvi_df.copy()
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year
    df["harvest_year"] = df.apply(
        lambda r: r["year"] + 1 if r["month"] >= 10 else r["year"], axis=1
    )
    season = df[df["month"].isin([10, 11, 12, 1, 2, 3, 4, 5])]
    return (season.groupby("harvest_year")["mean_ndvi"].mean()
                  .reset_index()
                  .rename(columns={"harvest_year": "year"}))


def county_season_ndvi(county_df: pd.DataFrame) -> pd.DataFrame:
    """Compute growing-season (Oct–May) mean NDVI per county per harvest year."""
    df = county_df.copy()
    df["date"]  = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year
    df["harvest_year"] = df.apply(
        lambda r: r["year"] + 1 if r["month"] >= 10 else r["year"], axis=1
    )
    season = df[df["month"].isin([10, 11, 12, 1, 2, 3, 4, 5])]
    return (
        season.groupby(["harvest_year", "county", "geoid"])["mean_ndvi"]
        .mean()
        .reset_index()
        .rename(columns={"harvest_year": "year"})
        .sort_values(["year", "county"])
        .reset_index(drop=True)
    )


def annual_avg_price(prices_df: pd.DataFrame) -> pd.DataFrame:
    df = prices_df.copy()
    df["year"] = df["date"].dt.year
    return (df.groupby("year")["close"].mean()
              .reset_index()
              .rename(columns={"close": "avg_oj_price"}))


def build_dataset() -> pd.DataFrame:
    ndvi     = pd.read_csv(RAW / "ndvi_raw.csv", parse_dates=["date"])
    yield_df = pd.read_csv(RAW / "yield_raw.csv")
    prices   = pd.read_csv(RAW / "prices_raw.csv", parse_dates=["date"])

    ndvi_season  = growing_season_ndvi(ndvi)
    price_annual = annual_avg_price(prices)

    dataset = (ndvi_season
               .merge(yield_df,    on="year")
               .merge(price_annual, on="year")
               .sort_values("year")
               .reset_index(drop=True))

    hist_avg = dataset["yield_boxes"].mean()
    dataset["yield_vs_avg"]   = dataset["yield_boxes"] / hist_avg
    dataset["price_pressure"] = dataset["yield_vs_avg"].apply(
        lambda r: "bullish" if r < 0.9 else ("bearish" if r > 1.1 else "neutral")
    )
    return dataset


# ── County-level yield estimation ─────────────────────────────────────────────

def _interpolate_county_acres(acres_raw: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """
    Linearly interpolate / extrapolate bearing acres for each county to fill
    every year in `years`, starting from Census anchor points.

    - Interpolate between Census years using linear interpolation.
    - Extrapolate beyond the last Census year using the 2017→2022 slope, capped at 0.
    - Years before the first available Census point use the first Census value.
    """
    records = []
    for county, grp in acres_raw.groupby("county"):
        grp = grp.set_index("year")["bearing_acres"].dropna().sort_index()
        if grp.empty:
            continue
        census_years = grp.index.tolist()

        for yr in years:
            if yr <= census_years[0]:
                val = grp.iloc[0]
            elif yr >= census_years[-1]:
                # Extrapolate from last two census points
                y1, y2 = census_years[-2], census_years[-1]
                v1, v2 = grp[y1], grp[y2]
                val = v2 + (yr - y2) / (y2 - y1) * (v2 - v1)
                val = max(0.0, val)
            else:
                # Linear interpolation between bounding census years
                y1 = max(cy for cy in census_years if cy <= yr)
                y2 = min(cy for cy in census_years if cy >= yr)
                if y1 == y2:
                    val = grp[y1]
                else:
                    val = grp[y1] + (yr - y1) / (y2 - y1) * (grp[y2] - grp[y1])
            records.append({"county": county, "year": yr, "bearing_acres": val})

    return pd.DataFrame(records).sort_values(["county", "year"]).reset_index(drop=True)


def county_yield_estimates(
    county_ndvi_df:    pd.DataFrame,
    county_acres_raw:  pd.DataFrame,
    state_acres_df:    pd.DataFrame,
    state_yield_df:    pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive county-level production estimates (boxes) by allocating statewide
    production in proportion to each county's share of Florida bearing acreage.

    Formula:
        county_yield_est[c,t] = statewide_yield[t]
                               × bearing_acres[c,t] / state_bearing_acres[t]

    This is a standard agricultural economics allocation method, used here
    because NASS does not publish county-level production in boxes.
    """
    model_years = sorted(county_ndvi_df["year"].unique().tolist())

    # Interpolate/extrapolate county acres to all model years
    county_acres_annual = _interpolate_county_acres(county_acres_raw, model_years)

    # Pivot state acres to a lookup dict (use annual Survey data)
    state_acres_lu = dict(zip(state_acres_df["year"], state_acres_df["state_bearing_acres"]))
    state_yield_lu = dict(zip(state_yield_df["year"], state_yield_df["yield_boxes"]))

    records = []
    for _, row in county_acres_annual.iterrows():
        yr     = int(row["year"])
        county = row["county"]
        c_acres = row["bearing_acres"]
        s_acres = state_acres_lu.get(yr)
        s_yield = state_yield_lu.get(yr)

        if s_acres is None or s_yield is None or s_acres == 0:
            continue

        records.append({
            "year":              yr,
            "county":            county,
            "bearing_acres":     c_acres,
            "county_share":      c_acres / s_acres,
            "county_yield_est":  s_yield * c_acres / s_acres,
            "state_yield":       s_yield,
        })

    return pd.DataFrame(records)


def build_county_dataset() -> pd.DataFrame:
    """
    Build county-level panel dataset: NDVI × yield estimate per county × year.
    Returns a DataFrame with one row per (county, year).
    """
    # ── Load inputs ───────────────────────────────────────────────────────────
    county_ndvi = pd.read_csv(COUNTY_SEASONAL_PATH)

    if not COUNTY_ACRES_PATH.exists():
        raise FileNotFoundError(f"Run fetch_yield.py first — {COUNTY_ACRES_PATH} not found")
    if not STATE_ACRES_PATH.exists():
        raise FileNotFoundError(f"Run fetch_yield.py first — {STATE_ACRES_PATH} not found")

    county_acres = pd.read_csv(COUNTY_ACRES_PATH)
    state_acres  = pd.read_csv(STATE_ACRES_PATH)
    state_yield  = pd.read_csv(RAW / "yield_raw.csv")

    # ── Compute county yield estimates ────────────────────────────────────────
    yield_est = county_yield_estimates(county_ndvi, county_acres, state_acres, state_yield)

    # ── Merge NDVI with yield estimates ───────────────────────────────────────
    merged = county_ndvi.merge(yield_est, on=["county", "year"], how="inner")
    merged = merged.sort_values(["year", "county"]).reset_index(drop=True)

    # Add price info for completeness
    prices = pd.read_csv(RAW / "prices_raw.csv", parse_dates=["date"])
    price_annual = annual_avg_price(prices)
    merged = merged.merge(price_annual, on="year", how="left")

    return merged


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # ── Statewide dataset (backward compatible) ───────────────────────────────
    df = build_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)}-row statewide dataset → {OUTPUT_PATH}")
    print(df.tail())

    # ── County NDVI seasonal (from raw county time series) ───────────────────
    if COUNTY_RAW_PATH.exists():
        county_df = pd.read_csv(COUNTY_RAW_PATH)
        county_seasonal = county_season_ndvi(county_df)
        county_seasonal.to_csv(COUNTY_SEASONAL_PATH, index=False)
        print(f"\nSaved {len(county_seasonal)}-row county seasonal NDVI → {COUNTY_SEASONAL_PATH}")
    else:
        print("\nNo county NDVI raw file found — skipping ndvi_county_seasonal.csv")

    # ── County-level panel dataset ────────────────────────────────────────────
    if COUNTY_ACRES_PATH.exists() and COUNTY_SEASONAL_PATH.exists():
        county_ds = build_county_dataset()
        county_ds.to_csv(COUNTY_DATASET_PATH, index=False)
        print(f"Saved {len(county_ds)}-row county dataset → {COUNTY_DATASET_PATH}")

        print("\nCounty-level yield estimates (2015–2024 excerpt):")
        pivot = county_ds[county_ds["year"].between(2015, 2024)].pivot_table(
            index="county", columns="year", values="county_yield_est",
            aggfunc="first"
        )
        print((pivot / 1e6).round(2).to_string())
        print("(values in millions of boxes)")
    else:
        print("\nSkipping county dataset — run fetch_yield.py first")


if __name__ == "__main__":
    main()
