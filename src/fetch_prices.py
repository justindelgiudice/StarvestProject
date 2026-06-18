"""
Fetch OJ futures price history via yfinance (ticker: OJ=F).
Outputs data/raw/prices_raw.csv with columns: date, close
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

TICKER = "OJ=F"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "prices_raw.csv"


def fetch_prices(start: str = "2015-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    raw = yf.download(TICKER, start=start, end=end, auto_adjust=True)
    df = raw[["Close"]].rename(columns={"Close": "close"}).reset_index()
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df


def main():
    df = fetch_prices()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} price records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
