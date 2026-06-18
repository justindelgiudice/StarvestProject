"""
Backtest: for each year, train on all prior years and predict on the held-out year.
Outputs data/processed/backtest_results.csv
"""

import json
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

DATASET_PATH = Path(__file__).parent.parent / "data" / "processed" / "dataset.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "backtest_results.csv"


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("year").reset_index(drop=True)
    results = []

    for i in range(2, len(df)):  # need at least 2 years to train
        train = df.iloc[:i]
        test = df.iloc[i]

        model = LinearRegression()
        model.fit(train[["mean_ndvi"]], train["yield_boxes"])

        predicted = model.predict([[test["mean_ndvi"]]])[0]
        actual = test["yield_boxes"]
        hist_avg = train["yield_boxes"].mean()

        results.append({
            "year": test["year"],
            "actual_yield": actual,
            "predicted_yield": predicted,
            "error_boxes": predicted - actual,
            "pct_error": (predicted - actual) / actual * 100,
            "actual_pressure": test["price_pressure"],
            "predicted_pressure": (
                "bullish" if predicted / hist_avg < 0.9
                else "bearish" if predicted / hist_avg > 1.1
                else "neutral"
            ),
        })

    return pd.DataFrame(results)


def main():
    df = pd.read_csv(DATASET_PATH)
    results = run_backtest(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    correct = (results["actual_pressure"] == results["predicted_pressure"]).sum()
    print(f"Price pressure accuracy: {correct}/{len(results)} = {correct/len(results):.0%}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
