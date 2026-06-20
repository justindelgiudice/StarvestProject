"""
Walk-forward backtest for the yield model.

Uses the county-level panel dataset when available (preferred) and falls
back to the statewide dataset otherwise.

County backtest logic:
  For each held-out year t (starting once we have MIN_TRAIN_YEARS of history):
    - Train county model on all (county, year') where year' < t
    - Predict county yield for each of the 8 counties in year t
    - Sum county predictions → raw statewide prediction
    - Scale by coverage fraction (average proportion of FL production our
      8 counties represent across training years)
    - Compare against actual statewide yield

Outputs data/processed/backtest_results.csv with the same schema as before
so the dashboard and price model don't require changes.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

ROOT             = Path(__file__).parent.parent
DATASET_PATH     = ROOT / "data" / "processed" / "dataset.csv"
COUNTY_DS_PATH   = ROOT / "data" / "processed" / "county_dataset.csv"
PARAMS_PATH      = ROOT / "data" / "processed" / "model_params.json"
OUTPUT_PATH      = ROOT / "data" / "processed" / "backtest_results.csv"

FEATURES       = ["mean_ndvi", "year"]
MIN_TRAIN_YEARS = 3   # minimum years of history before first prediction


# ── Statewide backtest (fallback) ─────────────────────────────────────────────

def run_statewide_backtest(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.sort_values("year").reset_index(drop=True)
    hist = df["yield_boxes"].mean()
    results = []

    for i in range(2, len(df)):
        train = df.iloc[:i]
        test  = df.iloc[i]
        model = LinearRegression().fit(train[FEATURES], train["yield_boxes"])
        pred  = model.predict([[test["mean_ndvi"], test["year"]]])[0]
        th    = train["yield_boxes"].mean()
        results.append({
            "year":               int(test["year"]),
            "actual_yield":       float(test["yield_boxes"]),
            "predicted_yield":    float(pred),
            "error_boxes":        float(pred - test["yield_boxes"]),
            "pct_error":          float((pred - test["yield_boxes"]) / test["yield_boxes"] * 100),
            "actual_pressure":    str(test["price_pressure"]),
            "predicted_pressure": (
                "bullish" if pred / th < 0.9 else
                "bearish" if pred / th > 1.1 else "neutral"
            ),
            "method": "statewide",
        })

    return pd.DataFrame(results)


# ── County backtest (primary) ─────────────────────────────────────────────────

def run_county_backtest(county_df: pd.DataFrame, statewide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward backtest using county-level panel.

    For each held-out year, trains on all prior years' county data,
    predicts all counties, sums to statewide, scales by coverage fraction.
    """
    years = sorted(county_df["year"].unique())
    statewide_lu = dict(zip(statewide_df["year"], statewide_df["yield_boxes"]))
    pressure_lu  = dict(zip(statewide_df["year"], statewide_df["price_pressure"]))

    results = []

    for idx, test_year in enumerate(years):
        train_years = [y for y in years if y < test_year]
        if len(train_years) < MIN_TRAIN_YEARS:
            continue

        train = county_df[county_df["year"].isin(train_years)].dropna(
            subset=["mean_ndvi", "county_yield_est"]
        )
        test  = county_df[county_df["year"] == test_year].dropna(subset=["mean_ndvi"])

        model = LinearRegression().fit(train[FEATURES].values, train["county_yield_est"].values)

        test_pred_county = model.predict(test[FEATURES].values)

        # Coverage fraction: county share of statewide in training years
        cov = float(
            (train.groupby("year")["county_yield_est"].sum()
             / train.groupby("year")["state_yield"].first()).mean()
        )

        predicted_statewide = float(test_pred_county.sum()) / cov
        actual_statewide    = float(statewide_lu.get(test_year, np.nan))

        if np.isnan(actual_statewide):
            continue

        th = float(statewide_df[statewide_df["year"].isin(train_years)]["yield_boxes"].mean())
        results.append({
            "year":               test_year,
            "actual_yield":       actual_statewide,
            "predicted_yield":    predicted_statewide,
            "error_boxes":        predicted_statewide - actual_statewide,
            "pct_error":          (predicted_statewide - actual_statewide) / actual_statewide * 100,
            "actual_pressure":    pressure_lu.get(test_year, "unknown"),
            "predicted_pressure": (
                "bullish" if predicted_statewide / th < 0.9 else
                "bearish" if predicted_statewide / th > 1.1 else "neutral"
            ),
            "method": "county_panel",
        })

    return pd.DataFrame(results)


def main():
    statewide_df = pd.read_csv(DATASET_PATH)

    if COUNTY_DS_PATH.exists():
        county_df = pd.read_csv(COUNTY_DS_PATH)
        results   = run_county_backtest(county_df, statewide_df)
        method    = "county panel"
    else:
        print("county_dataset.csv not found — falling back to statewide backtest")
        results = run_statewide_backtest(statewide_df)
        method  = "statewide"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    correct = (results["actual_pressure"] == results["predicted_pressure"]).sum()
    r2  = r2_score(results["actual_yield"], results["predicted_yield"])
    mae = mean_absolute_error(results["actual_yield"], results["predicted_yield"])

    print(f"Backtest method: {method}")
    print(f"Years tested: {len(results)}  ({results['year'].min()}–{results['year'].max()})")
    print(f"R² (backtest): {r2:.3f}")
    print(f"MAE (backtest): {mae/1e6:.2f}M boxes")
    print(f"Pressure accuracy: {correct}/{len(results)} = {correct/len(results):.0%}")
    print()
    print(results[["year","actual_yield","predicted_yield","pct_error",
                   "actual_pressure","predicted_pressure"]].to_string(index=False))


if __name__ == "__main__":
    main()
