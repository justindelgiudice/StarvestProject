"""
FastAPI backend — serves processed Starvest data to the React frontend.
Run with: uvicorn api.main:app --reload
"""

import json
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Starvest API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
RAW = Path(__file__).parent.parent / "data" / "raw"


def load_json(path: Path):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} not found — run the pipeline first")
    with open(path) as f:
        return json.load(f)


def load_csv(path: Path):
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} not found — run the pipeline first")
    return pd.read_csv(path)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/ndvi")
def get_ndvi():
    df = load_csv(RAW / "ndvi_raw.csv")
    return df.to_dict(orient="records")


@app.get("/api/dataset")
def get_dataset():
    df = load_csv(PROCESSED / "dataset.csv")
    return df.to_dict(orient="records")


@app.get("/api/backtest")
def get_backtest():
    df = load_csv(PROCESSED / "backtest_results.csv")
    return df.to_dict(orient="records")


@app.get("/api/model-params")
def get_model_params():
    return load_json(PROCESSED / "model_params.json")


@app.get("/api/forecast/{year}")
def get_forecast(year: int):
    dataset = load_csv(PROCESSED / "dataset.csv")
    params = load_json(PROCESSED / "model_params.json")

    row = dataset[dataset["year"] == year]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"No data for year {year}")

    row = row.iloc[0]
    predicted_yield = (
        params["intercept"]
        + params["coef_ndvi"] * row["mean_ndvi"]
        + params["coef_year"] * year
    )
    ratio = predicted_yield / params["historical_avg_yield"]
    pressure = "bullish" if ratio < 0.9 else ("bearish" if ratio > 1.1 else "neutral")

    return {
        "year": year,
        "mean_ndvi": row["mean_ndvi"],
        "predicted_yield": predicted_yield,
        "actual_yield": row["yield_boxes"],
        "historical_avg_yield": params["historical_avg_yield"],
        "yield_vs_avg": ratio,
        "price_pressure": pressure,
        "avg_oj_price": row["avg_oj_price"],
    }


@app.get("/api/summary")
def get_summary():
    dataset = load_csv(PROCESSED / "dataset.csv")
    backtest = load_csv(PROCESSED / "backtest_results.csv")
    params = load_json(PROCESSED / "model_params.json")

    accuracy = (backtest["actual_pressure"] == backtest["predicted_pressure"]).mean()
    latest = dataset.sort_values("year").iloc[-1]

    return {
        "latest_year": int(latest["year"]),
        "latest_ndvi": latest["mean_ndvi"],
        "latest_yield": latest["yield_boxes"],
        "latest_price_pressure": latest["price_pressure"],
        "latest_oj_price": latest["avg_oj_price"],
        "historical_avg_yield": params["historical_avg_yield"],
        "model_r2": params["r2_train"],
        "model_mae": params["mae_loo"],
        "backtest_accuracy": accuracy,
    }
