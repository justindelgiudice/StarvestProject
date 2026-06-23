"""
Merge satellite, permit, and ZHVI data into a single panel dataset.

For each metro × quarter:
  - Satellite: NDBI change (QoQ), BSI change — construction activity proxy
  - Permits: rolling 12-month permit count, YoY change
  - ZHVI: home value index, YoY % change (target variable)
  - Lead features: satellite/permit signals lead ZHVI by 1–4 quarters

Outputs data/processed/dataset.csv:
  metro | quarter | ndbi_chg | bsi_chg | permits_yoy | zhvi | zhvi_yoy | zhvi_fwd_4q

Run after: fetch_home_prices.py, fetch_satellite.py, fetch_permits.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT     = Path(__file__).parent.parent
RAW      = ROOT / "data" / "raw"
OUT      = ROOT / "data" / "processed" / "dataset.csv"


def load_zhvi() -> pd.DataFrame:
    df = pd.read_csv(RAW / "zhvi_raw.csv", parse_dates=["date"])
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    df = df.groupby(["metro", "quarter"])["zhvi"].mean().reset_index()
    df = df.sort_values(["metro", "quarter"])
    df["zhvi_yoy"] = df.groupby("metro")["zhvi"].pct_change(4) * 100
    # Forward 4-quarter ZHVI change (what we predict)
    df["zhvi_fwd_4q"] = df.groupby("metro")["zhvi"].pct_change(4).shift(-4) * 100
    return df


def load_satellite() -> pd.DataFrame:
    path = RAW / "satellite_raw.csv"
    if not path.exists():
        return pd.DataFrame(columns=["metro", "quarter", "ndbi_mean", "bsi_mean"])
    df = pd.read_csv(path)
    df = df.sort_values(["metro", "quarter"])
    df["ndbi_chg"] = df.groupby("metro")["ndbi_mean"].diff()
    df["bsi_chg"]  = df.groupby("metro")["bsi_mean"].diff()
    return df[["metro", "quarter", "ndbi_mean", "ndbi_chg", "bsi_mean", "bsi_chg"]]


def load_permits() -> pd.DataFrame:
    path = RAW / "permits_raw.csv"
    if not path.exists():
        return pd.DataFrame(columns=["metro", "quarter", "permits_ttm", "permits_yoy"])
    df = pd.read_csv(path, parse_dates=["date"])
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    q = df.groupby(["metro", "quarter"])["permits"].sum().reset_index()
    q = q.sort_values(["metro", "quarter"])
    # Trailing 4-quarter (TTM) permit sum
    q["permits_ttm"] = q.groupby("metro")["permits"].transform(
        lambda x: x.rolling(4, min_periods=1).sum()
    )
    q["permits_yoy"] = q.groupby("metro")["permits_ttm"].pct_change(4) * 100
    return q[["metro", "quarter", "permits_ttm", "permits_yoy"]]


def main():
    zhvi    = load_zhvi()
    sat     = load_satellite()
    permits = load_permits()

    df = zhvi.merge(sat,     on=["metro", "quarter"], how="left")
    df = df.merge(permits,   on=["metro", "quarter"], how="left")
    df = df.sort_values(["metro", "quarter"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Dataset: {len(df)} rows, {df['metro'].nunique()} metros")
    print(f"Columns: {list(df.columns)}")
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
