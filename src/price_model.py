"""
Price regression model: yield_vs_avg + lagged_price → predicted OJ futures price.

Features:
  - yield_vs_avg : relative supply signal (yield this season / long-run avg)
  - lagged_price : prior year's avg OJ futures price (captures momentum / mean reversion)

Walk-forward backtest (MIN_TRAIN=4) avoids look-ahead bias.  Requires at least
4 training samples before the first out-of-sample prediction so the model is
over-determined (2 features + intercept = 3 params; 4 points gives 1 df of error).

Outputs
-------
data/processed/price_model_params.json   — coefficients + forecast for next year
data/processed/price_backtest_results.csv — year-by-year backtest predictions
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

ROOT          = Path(__file__).parent.parent
DATASET_PATH  = ROOT / "data" / "processed" / "dataset.csv"
NDVI_RAW      = ROOT / "data" / "raw" / "ndvi_raw.csv"
YIELD_PARAMS  = ROOT / "data" / "processed" / "model_params.json"
PARAMS_PATH   = ROOT / "data" / "processed" / "price_model_params.json"
BACKTEST_PATH = ROOT / "data" / "processed" / "price_backtest_results.csv"

PRICE_FEATURES = ["yield_vs_avg", "lagged_price"]
MIN_TRAIN = 4

# Must match COUNTY_BBOXES / CITRUS_FIPS in fetch_ndvi.py.
# Only these 8 counties are used for yield modelling; the other 59 FL counties
# are now fetched for NDVI visualisation only and must NOT be fed into the model.
CITRUS_FIPS = {"12105", "12055", "12027", "12049", "12051", "12015", "12043", "12081"}

# Minimum plausible FL statewide orange production — prevents linear trend extrapolation
# from forecasting physically implausible near-zero harvests in forward-looking years.
YIELD_FLOOR = 8_000_000  # boxes


def add_lagged_price(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("year").reset_index(drop=True).copy()
    df["lagged_price"] = df["avg_oj_price"].shift(1)
    return df.dropna(subset=["lagged_price"]).reset_index(drop=True)


def backtest_price(df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward: train on prior years only, predict held-out year."""
    df_lag = add_lagged_price(df)
    results = []

    for i in range(MIN_TRAIN, len(df_lag)):
        train = df_lag.iloc[:i]
        test  = df_lag.iloc[i]

        model = LinearRegression()
        model.fit(train[PRICE_FEATURES].values, train["avg_oj_price"].values)

        predicted_price = float(model.predict(test[PRICE_FEATURES].values.reshape(1, -1))[0])
        actual_price    = float(test["avg_oj_price"])
        lag_price       = float(test["lagged_price"])

        pct_change_actual    = (actual_price - lag_price) / lag_price * 100
        pct_change_predicted = (predicted_price - lag_price) / lag_price * 100
        direction_correct    = (pct_change_actual > 0) == (pct_change_predicted > 0)

        results.append({
            "year":                 int(test["year"]),
            "lagged_price":         round(lag_price, 2),
            "actual_price":         round(actual_price, 2),
            "predicted_price":      round(predicted_price, 2),
            "price_error":          round(predicted_price - actual_price, 2),
            "pct_error":            round((predicted_price - actual_price) / actual_price * 100, 2),
            "actual_pct_change":    round(pct_change_actual, 2),
            "predicted_pct_change": round(pct_change_predicted, 2),
            "direction_correct":    bool(direction_correct),
        })

    return pd.DataFrame(results)


def _season_ndvi(harvest_year: int) -> float | None:
    """Growing-season (Oct–May) mean NDVI for a given harvest year from ndvi_raw.csv."""
    try:
        ndvi = pd.read_csv(NDVI_RAW, parse_dates=["date"])
        ndvi["month"] = ndvi["date"].dt.month
        ndvi["year"]  = ndvi["date"].dt.year
        # Oct–Dec of (harvest_year−1) and Jan–May of harvest_year = that harvest season
        ndvi["harvest_year"] = ndvi.apply(
            lambda r: r["year"] + 1 if r["month"] >= 10 else r["year"], axis=1
        )
        season = ndvi[
            (ndvi["harvest_year"] == harvest_year) &
            (ndvi["month"].isin([10, 11, 12, 1, 2, 3, 4, 5]))
        ]
        if len(season) < 3:
            return None
        return float(season["mean_ndvi"].mean())
    except Exception:
        return None


def forecast_yield_vs_avg(forecast_year: int, historical_avg_yield: float) -> tuple[float, str]:
    """
    Predict yield_vs_avg for forecast_year using the yield model + current NDVI.

    Handles two model types:
    - county_panel: predicts each county separately using per-county NDVI from
      ndvi_county_seasonal.csv, then aggregates to statewide via coverage fraction.
    - statewide (legacy): applies coefficients directly to regional-average NDVI.

    Returns (yield_vs_avg, source).
    """
    if not YIELD_PARAMS.exists():
        return None, "unavailable"

    with open(YIELD_PARAMS) as f:
        yp = json.load(f)

    if yp.get("training_mode") == "county_panel":
        county_seasonal_path = ROOT / "data" / "processed" / "ndvi_county_seasonal.csv"
        if county_seasonal_path.exists():
            cs = pd.read_csv(county_seasonal_path)
            # Filter to the 8 citrus-belt counties the model was trained on.
            # The seasonal CSV now covers all 67 FL counties for visualisation,
            # but summing non-citrus counties would break the coverage_fraction math.
            cs_citrus = cs[cs["geoid"].astype(str).isin(CITRUS_FIPS)]
            yr = cs_citrus[cs_citrus["year"] == forecast_year].dropna(subset=["mean_ndvi"])
            if not yr.empty:
                _trend_year = min(forecast_year, yp.get("last_training_year", forecast_year))
                county_preds = (
                    yp["intercept"]
                    + yp["coef_ndvi"] * yr["mean_ndvi"].values
                    + yp["coef_year"] * _trend_year
                )
                predicted_yield = max(YIELD_FLOOR, float(county_preds.sum()) / yp["coverage_fraction"])
                return float(predicted_yield / yp["historical_avg_yield"]), "yield_model_county"

    # Statewide / fallback: single regional-average NDVI
    ndvi = _season_ndvi(forecast_year)
    if ndvi is not None:
        predicted_yield = (
            yp["intercept"]
            + yp["coef_ndvi"] * ndvi
            + yp["coef_year"] * forecast_year
        )
        return float(predicted_yield / yp["historical_avg_yield"]), "yield_model"

    return None, "unavailable"


def pressure_from_pct_change(pct_change: float) -> str:
    if pct_change > 5:
        return "bullish"
    elif pct_change < -5:
        return "bearish"
    return "neutral"


def _county_yield_vs_avg(forecast_year: int, yp: dict, cs_citrus: pd.DataFrame) -> tuple[float, float | None, str]:
    """
    Apply the county panel model to the 8 citrus counties for a given forecast year.
    For years with no NDVI data, falls back to the mean of the last 3 available seasons.

    Returns (yield_vs_avg, mean_ndvi_used, source_label).
    """
    yr = cs_citrus[cs_citrus["year"] == forecast_year].dropna(subset=["mean_ndvi"])

    # Freeze the year component at the last training year so the linear HLB decline
    # trend isn't extrapolated into the future — only NDVI varies across forecast years.
    _trend_year = min(forecast_year, yp.get("last_training_year", forecast_year))

    if not yr.empty:
        county_preds = (
            yp["intercept"]
            + yp["coef_ndvi"] * yr["mean_ndvi"].values
            + yp["coef_year"] * _trend_year
        )
        predicted_yield = max(YIELD_FLOOR, float(county_preds.sum()) / yp["coverage_fraction"])
        return (
            predicted_yield / yp["historical_avg_yield"],
            float(yr["mean_ndvi"].mean()),
            "yield_model_county",
        )

    # No NDVI available — estimate using the mean of the last 3 seasons
    recent_yrs = sorted(cs_citrus["year"].unique())[-3:]
    recent = cs_citrus[cs_citrus["year"].isin(recent_yrs)].copy()
    # Average per county across those seasons, then apply model
    avg_by_county = recent.groupby("geoid")["mean_ndvi"].mean().reset_index()
    county_preds = (
        yp["intercept"]
        + yp["coef_ndvi"] * avg_by_county["mean_ndvi"].values
        + yp["coef_year"] * _trend_year
    )
    predicted_yield = max(YIELD_FLOOR, float(county_preds.sum()) / yp["coverage_fraction"])
    return (
        predicted_yield / yp["historical_avg_yield"],
        float(avg_by_county["mean_ndvi"].mean()),
        "avg_last_3_seasons",
    )


def forecast_multiyear(
    model,
    df_lag: pd.DataFrame,
    yp: dict,
    start_year: int,
    n_years: int = 3,
) -> list[dict]:
    """
    Chained forward forecast for n_years starting at start_year.

    Each year's predicted price becomes the lagged_price for the next year.
    NDVI for years without satellite data is estimated from the last 3 seasons.
    """
    cs_path = ROOT / "data" / "processed" / "ndvi_county_seasonal.csv"
    if not cs_path.exists():
        return []

    cs_full   = pd.read_csv(cs_path)
    cs_citrus = cs_full[cs_full["geoid"].astype(str).isin(CITRUS_FIPS)]

    # Seed the chain with the last actual price
    df_sorted  = df_lag.sort_values("year")
    prev_price = float(df_sorted["avg_oj_price"].iloc[-1])

    results = []
    for offset in range(n_years):
        year = start_year + offset

        yva, ndvi_used, yva_source = _county_yield_vs_avg(year, yp, cs_citrus)

        predicted_price = float(model.predict([[yva, prev_price]])[0])
        pct_change      = (predicted_price - prev_price) / prev_price * 100

        results.append({
            "year":                  year,
            "forecast_ndvi":         round(ndvi_used, 4) if ndvi_used is not None else None,
            "ndvi_source":           yva_source,
            "forecast_yield_vs_avg": round(yva, 4),
            "predicted_yield":       round(yva * yp["historical_avg_yield"]),
            "lagged_price":          round(prev_price, 2),
            "predicted_price":       round(predicted_price, 2),
            "pct_change":            round(pct_change, 2),
            "price_pressure":        pressure_from_pct_change(pct_change),
        })

        prev_price = predicted_price  # chain: predicted → next year's lagged

    return results


def main():
    df     = pd.read_csv(DATASET_PATH)
    df_lag = add_lagged_price(df)

    # ── Full-data model for deployment ────────────────────────────────────────
    model = LinearRegression()
    model.fit(df_lag[PRICE_FEATURES].values, df_lag["avg_oj_price"].values)

    # ── Walk-forward backtest ─────────────────────────────────────────────────
    bt      = backtest_price(df)
    r2      = r2_score(bt["actual_price"], bt["predicted_price"])
    mae     = mean_absolute_error(bt["actual_price"], bt["predicted_price"])
    dir_acc = float(bt["direction_correct"].mean())

    BACKTEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    bt.to_csv(BACKTEST_PATH, index=False)

    # ── Forecast next year ────────────────────────────────────────────────────
    df_sorted     = df.sort_values("year")
    last_year     = int(df_sorted["year"].iloc[-1])
    last_price    = float(df_sorted["avg_oj_price"].iloc[-1])
    hist_avg      = float(df_sorted["yield_boxes"].mean())
    forecast_year = last_year + 1

    yva, yva_source = forecast_yield_vs_avg(forecast_year, hist_avg)
    if yva is None:
        # Fallback: assume similar supply conditions to most recent year
        yva        = float(df_sorted["yield_vs_avg"].iloc[-1])
        yva_source = "last_known"

    predicted_price   = float(model.predict([[yva, last_price]])[0])
    predicted_pct_chg = (predicted_price - last_price) / last_price * 100
    price_pressure    = pressure_from_pct_change(predicted_pct_chg)

    ndvi_2025 = _season_ndvi(forecast_year)

    # ── Multi-year chained forecast (2025, 2026, 2027) ───────────────────────
    with open(YIELD_PARAMS) as f:
        yp_full = json.load(f)

    multiyear = forecast_multiyear(model, df_lag, yp_full, start_year=forecast_year, n_years=3)

    params = {
        "intercept":                    float(model.intercept_),
        "coef_yield_vs_avg":            float(model.coef_[0]),
        "coef_lagged_price":            float(model.coef_[1]),
        "r2_backtest":                  round(r2, 4),
        "mae_backtest":                 round(mae, 2),
        "directional_accuracy":         round(dir_acc, 4),
        "n_backtest_years":             len(bt),
        "last_year":                    last_year,
        "last_price":                   round(last_price, 2),
        "forecast_year":                forecast_year,
        "forecast_ndvi":                round(ndvi_2025, 4) if ndvi_2025 else None,
        "forecast_yield_vs_avg":        round(yva, 4),
        "forecast_yield_vs_avg_source": yva_source,
        "predicted_price":              round(predicted_price, 2),
        "predicted_pct_change":         round(predicted_pct_chg, 2),
        "price_pressure":               price_pressure,
        "multiyear_forecasts":          multiyear,
    }

    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("PRICE MODEL — Coefficients (trained on full dataset)")
    print(f"{'='*65}")
    print(f"  Intercept:          {model.intercept_:+.2f}")
    print(f"  coef_yield_vs_avg:  {model.coef_[0]:+.2f}   (lower supply → higher price)")
    print(f"  coef_lagged_price:  {model.coef_[1]:+.4f}   (price momentum)")

    print(f"\n{'='*65}")
    print("WALK-FORWARD BACKTEST  (trained on prior years only)")
    print(f"{'='*65}")
    cols = ["year", "actual_price", "predicted_price", "price_error",
            "actual_pct_change", "predicted_pct_change", "direction_correct"]
    print(bt[cols].to_string(index=False))
    print()
    print(f"  R² (backtest):          {r2:.3f}")
    print(f"  MAE (backtest):         ¢{mae:.2f}")
    print(f"  Directional accuracy:   {dir_acc:.0%}  ({int(bt['direction_correct'].sum())}/{len(bt)} years correct)")

    print(f"\n{'='*65}")
    print(f"FORECAST  {forecast_year}")
    print(f"{'='*65}")
    if ndvi_2025:
        print(f"  {forecast_year} growing-season NDVI:  {ndvi_2025:.4f}")
    print(f"  yield_vs_avg ({yva_source}):  {yva:.4f}×")
    print(f"  Lagged price (actual {last_year}):  ¢{last_price:.2f}")
    print(f"  Predicted {forecast_year} price:     ¢{predicted_price:.2f}")
    print(f"  % change vs {last_year}:       {predicted_pct_chg:+.1f}%")
    print(f"  Price signal:           {price_pressure.upper()}")
    print(f"\n{'='*65}")
    print(f"MULTI-YEAR CHAINED FORECAST  ({multiyear[0]['year']} → {multiyear[-1]['year']})")
    print(f"{'='*65}")
    for fc in multiyear:
        print(f"  {fc['year']}  NDVI={fc['forecast_ndvi']} ({fc['ndvi_source']})"
              f"  yield_vs_avg={fc['forecast_yield_vs_avg']:.3f}"
              f"  price=¢{fc['predicted_price']:.2f}"
              f"  ({fc['pct_change']:+.1f}%)  → {fc['price_pressure'].upper()}")

    print(f"\nSaved → {PARAMS_PATH}")
    print(f"Saved → {BACKTEST_PATH}")


if __name__ == "__main__":
    main()
