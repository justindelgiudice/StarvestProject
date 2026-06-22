"""
Train yield regression model(s) and save parameters.

Two model variants:
  1. Statewide  — NDVI (regional avg) + year → statewide yield (10 rows)
  2. County     — county NDVI + year → county yield estimate (panel, ~88 rows)

The county model is trained as a pooled linear regression on the county-level
panel dataset (county_dataset.csv).  Its county-level predictions are summed
and scaled to statewide for evaluation.  Coverage fraction (what share of FL
production our 8 counties represent) is saved and used by backtest.py.

model_params.json contains the COUNTY model by default (better generalisation)
with statewide-equivalent R² and MAE for dashboard display.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut

ROOT             = Path(__file__).parent.parent
DATASET_PATH     = ROOT / "data" / "processed" / "dataset.csv"
COUNTY_DS_PATH   = ROOT / "data" / "processed" / "county_dataset.csv"
PARAMS_PATH      = ROOT / "data" / "processed" / "model_params.json"

FEATURES = ["mean_ndvi", "year"]


# ── Statewide model (reference) ───────────────────────────────────────────────

def train_statewide(df: pd.DataFrame) -> dict:
    X = df[FEATURES].values
    y = df["yield_boxes"].values

    model = LinearRegression().fit(X, y)

    loo = LeaveOneOut()
    preds = [
        LinearRegression().fit(X[tr], y[tr]).predict(X[te])[0]
        for tr, te in loo.split(X)
    ]

    return {
        "r2_train":            r2_score(y, model.predict(X)),
        "mae_loo":             mean_absolute_error(y, preds),
        "intercept":           float(model.intercept_),
        "coef_ndvi":           float(model.coef_[0]),
        "coef_year":           float(model.coef_[1]),
        "historical_avg_yield": float(df["yield_boxes"].mean()),
    }


# ── County model (primary) ────────────────────────────────────────────────────

def train_county(df: pd.DataFrame, statewide_df: pd.DataFrame) -> dict:
    """
    Train pooled linear regression on county-level panel.

    Cross-validation uses LeaveOneYearOut to mimic walk-forward:
    each held-out group = all county observations for one year.
    This is the fair analogue of the statewide LOO-by-year CV.

    Returns params in the SAME format as the statewide model so the
    dashboard and price model don't need changing.
    """
    df = df.dropna(subset=["mean_ndvi", "county_yield_est"]).copy()

    X      = df[FEATURES].values
    y      = df["county_yield_est"].values
    groups = df["year"].values

    model = LinearRegression().fit(X, y)

    # Leave-one-year-out CV
    logo        = LeaveOneGroupOut()
    county_preds = np.empty(len(df))
    for tr, te in logo.split(X, y, groups):
        county_preds[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])

    county_r2  = r2_score(y, model.predict(X))
    county_mae = mean_absolute_error(y, county_preds)

    # ── Aggregate to statewide for evaluation ─────────────────────────────────
    df2 = df.copy()
    df2["pred_county"] = county_preds

    agg = df2.groupby("year").agg(
        county_yield_sum=("county_yield_est", "sum"),
        pred_sum        =("pred_county",      "sum"),
        state_yield     =("state_yield",      "first"),
    ).reset_index()

    # Coverage fraction: what share of FL production our 8 counties represent
    agg["coverage"] = agg["county_yield_sum"] / agg["state_yield"]
    avg_coverage    = float(agg["coverage"].mean())

    agg["pred_statewide"] = agg["pred_sum"] / avg_coverage

    state_r2  = r2_score(agg["state_yield"], agg["pred_statewide"])
    state_mae = mean_absolute_error(agg["state_yield"], agg["pred_statewide"])

    return {
        # Keys expected by dashboard / price model (statewide equivalents)
        "intercept":              float(model.intercept_),
        "coef_ndvi":              float(model.coef_[0]),
        "coef_year":              float(model.coef_[1]),
        "r2_train":               round(state_r2, 4),
        "mae_loo":                round(state_mae, 0),
        "historical_avg_yield":   float(statewide_df["yield_boxes"].mean()),
        "coverage_fraction":      round(avg_coverage, 4),
        "last_training_year":     int(statewide_df["year"].max()),
        # County-level detail
        "county_r2_train":        round(county_r2, 4),
        "county_mae_loo":         round(county_mae, 0),
        "n_county_obs":           int(len(df)),
        "training_mode":          "county_panel",
    }


def predict_yield(mean_ndvi: float, year: int, params: dict) -> float:
    return (params["intercept"]
            + params["coef_ndvi"] * mean_ndvi
            + params["coef_year"] * year)


def main():
    statewide_df = pd.read_csv(DATASET_PATH)

    # ── Statewide reference ───────────────────────────────────────────────────
    sw = train_statewide(statewide_df)
    print("Statewide model (10 obs, LOO-by-row CV):")
    print(f"  R² (train):   {sw['r2_train']:.3f}")
    print(f"  MAE (LOO):    {sw['mae_loo']:,.0f} boxes")

    if not COUNTY_DS_PATH.exists():
        print("\ncounty_dataset.csv not found — saving statewide model")
        PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PARAMS_PATH, "w") as f:
            json.dump(sw, f, indent=2)
        return

    county_df = pd.read_csv(COUNTY_DS_PATH)

    # ── County model ──────────────────────────────────────────────────────────
    cp = train_county(county_df, statewide_df)
    print(f"\nCounty model ({cp['n_county_obs']} obs, LOO-by-year CV):")
    print(f"  County R² (train):    {cp['county_r2_train']:.3f}")
    print(f"  County MAE (LOYO):    {cp['county_mae_loo']:,.0f} boxes/county")
    print(f"  Statewide R² (agg):   {cp['r2_train']:.3f}")
    print(f"  Statewide MAE (agg):  {cp['mae_loo']:,.0f} boxes")
    print(f"  Coverage fraction:    {cp['coverage_fraction']:.1%} of FL production")

    print(f"\nR² change: {sw['r2_train']:.3f} → {cp['r2_train']:.3f} "
          f"({'▲' if cp['r2_train'] > sw['r2_train'] else '▼'}"
          f" {abs(cp['r2_train'] - sw['r2_train']):.3f})")
    print(f"MAE change: {sw['mae_loo']/1e6:.2f}M → {cp['mae_loo']/1e6:.2f}M boxes")

    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_PATH, "w") as f:
        json.dump(cp, f, indent=2)
    print(f"\nSaved county model params → {PARAMS_PATH}")


if __name__ == "__main__":
    main()
