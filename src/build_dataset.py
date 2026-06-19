"""
Merge NDVI, yield, and price data into a single modelling dataset.
Outputs data/processed/dataset.csv
"""

import pandas as pd
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw"
PROCESSED = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_PATH          = PROCESSED / "dataset.csv"
COUNTY_RAW_PATH      = RAW / "ndvi_county_raw.csv"
COUNTY_SEASONAL_PATH = PROCESSED / "ndvi_county_seasonal.csv"


def growing_season_ndvi(ndvi_df: pd.DataFrame) -> pd.DataFrame:
    """Average NDVI over Oct–May growing season, labelled by harvest year."""
    df = ndvi_df.copy()
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    # Oct–Dec belong to the *next* harvest year
    df["harvest_year"] = df.apply(
        lambda r: r["year"] + 1 if r["month"] >= 10 else r["year"], axis=1
    )
    season = df[df["month"].isin([10, 11, 12, 1, 2, 3, 4, 5])]
    return season.groupby("harvest_year")["mean_ndvi"].mean().reset_index()\
                 .rename(columns={"harvest_year": "year"})


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
    return df.groupby("year")["close"].mean().reset_index()\
             .rename(columns={"close": "avg_oj_price"})


def build_dataset() -> pd.DataFrame:
    ndvi = pd.read_csv(RAW / "ndvi_raw.csv", parse_dates=["date"])
    yield_df = pd.read_csv(RAW / "yield_raw.csv")
    prices = pd.read_csv(RAW / "prices_raw.csv", parse_dates=["date"])

    ndvi_season = growing_season_ndvi(ndvi)
    price_annual = annual_avg_price(prices)

    dataset = ndvi_season.merge(yield_df, on="year")\
                         .merge(price_annual, on="year")\
                         .sort_values("year")\
                         .reset_index(drop=True)

    hist_avg_yield = dataset["yield_boxes"].mean()
    dataset["yield_vs_avg"] = dataset["yield_boxes"] / hist_avg_yield
    dataset["price_pressure"] = dataset["yield_vs_avg"].apply(
        lambda r: "bullish" if r < 0.9 else ("bearish" if r > 1.1 else "neutral")
    )
    return dataset


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)}-row dataset to {OUTPUT_PATH}")
    print(df.tail())

    if COUNTY_RAW_PATH.exists():
        county_df = pd.read_csv(COUNTY_RAW_PATH)
        county_seasonal = county_season_ndvi(county_df)
        county_seasonal.to_csv(COUNTY_SEASONAL_PATH, index=False)
        print(f"Saved {len(county_seasonal)}-row county seasonal NDVI → {COUNTY_SEASONAL_PATH}")
    else:
        print("No county NDVI file found — skipping ndvi_county_seasonal.csv")


if __name__ == "__main__":
    main()
