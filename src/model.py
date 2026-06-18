"""
Train and evaluate a linear regression model: NDVI → yield.
Saves model coefficients to data/processed/model_params.json.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut

DATASET_PATH = Path(__file__).parent.parent / "data" / "processed" / "dataset.csv"
PARAMS_PATH = Path(__file__).parent.parent / "data" / "processed" / "model_params.json"


def train(df: pd.DataFrame) -> dict:
    X = df[["mean_ndvi"]].values
    y = df["yield_boxes"].values

    model = LinearRegression()
    model.fit(X, y)

    # Leave-one-out CV for honest error estimate on small dataset
    loo = LeaveOneOut()
    preds = []
    for train_idx, test_idx in loo.split(X):
        m = LinearRegression().fit(X[train_idx], y[train_idx])
        preds.append(m.predict(X[test_idx])[0])

    params = {
        "intercept": model.intercept_,
        "coef_ndvi": model.coef_[0],
        "r2_train": r2_score(y, model.predict(X)),
        "mae_loo": mean_absolute_error(y, preds),
        "historical_avg_yield": float(df["yield_boxes"].mean()),
    }
    return params, model


def predict_yield(mean_ndvi: float, params: dict) -> float:
    return params["intercept"] + params["coef_ndvi"] * mean_ndvi


def infer_price_pressure(predicted_yield: float, params: dict) -> str:
    ratio = predicted_yield / params["historical_avg_yield"]
    if ratio < 0.9:
        return "bullish"
    elif ratio > 1.1:
        return "bearish"
    return "neutral"


def main():
    df = pd.read_csv(DATASET_PATH)
    params, _ = train(df)
    PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)
    print(f"R² (train): {params['r2_train']:.3f} | MAE (LOO): {params['mae_loo']:,.0f} boxes")
    print(f"Saved params to {PARAMS_PATH}")


if __name__ == "__main__":
    main()
