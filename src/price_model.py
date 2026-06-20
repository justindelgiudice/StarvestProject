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
    Returns (yield_vs_avg, source) where source is 'yield_model' or 'last_known'.
    """
    if YIELD_PARAMS.exists():
        with open(YIELD_PARAMS) as f:
            yp = json.load(f)
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
    print(f"\nSaved → {PARAMS_PATH}")
    print(f"Saved → {BACKTEST_PATH}")


if __name__ == "__main__":
    main()
