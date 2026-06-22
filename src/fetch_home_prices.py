"""
Fetch Zillow Home Value Index (ZHVI) for Florida metro areas.

Uses Zillow Research's public CSV: Metro-level, all homes (SFR + condo/co-op),
smoothed seasonally-adjusted middle tier (33rd–67th percentile).

Outputs data/raw/zhvi_raw.csv:
  metro | date | zhvi
"""

import io
import requests
import pandas as pd
from pathlib import Path

ROOT   = Path(__file__).parent.parent
OUTPUT = ROOT / "data" / "raw" / "zhvi_raw.csv"

ZHVI_URL = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)

# Target FL metros: short label → expected Zillow RegionName substring
FL_METRO_KEYS = {
    "Miami":           "Miami",
    "Tampa":           "Tampa",
    "Orlando":         "Orlando",
    "Jacksonville":    "Jacksonville",
    "Fort Lauderdale": "Fort Lauderdale",
}


def fetch_zhvi(url: str = ZHVI_URL) -> tuple[pd.DataFrame, dict[str, str]]:
    """Download and return (long_df, {short_name: full_region_name})."""
    print(f"Downloading ZHVI data...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    wide = pd.read_csv(io.StringIO(resp.text))

    fl_wide = wide[wide["StateName"] == "FL"].copy()

    found: dict[str, str] = {}
    for label, keyword in FL_METRO_KEYS.items():
        matches = fl_wide[fl_wide["RegionName"].str.contains(keyword, case=False, na=False)]
        if not matches.empty:
            found[label] = matches["RegionName"].iloc[0]

    if not found:
        raise ValueError("No Florida metros matched in ZHVI data. Check the URL or column names.")

    df_wide = fl_wide[fl_wide["RegionName"].isin(found.values())].copy()

    date_cols = [c for c in df_wide.columns if c[:4].isdigit()]
    df_long = df_wide.melt(
        id_vars=["RegionName"],
        value_vars=date_cols,
        var_name="date",
        value_name="zhvi",
    )
    df_long["date"] = pd.to_datetime(df_long["date"])
    df_long = df_long.rename(columns={"RegionName": "metro"})

    # Map full region names to short labels for readability
    inv = {v: k for k, v in found.items()}
    df_long["metro"] = df_long["metro"].map(inv)

    df_long = (
        df_long
        .dropna(subset=["zhvi"])
        .sort_values(["metro", "date"])
        .reset_index(drop=True)
    )

    return df_long, found


def main():
    df, found = fetch_zhvi()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"\nMatched {len(found)} of {len(FL_METRO_KEYS)} target metros:\n")
    for label, full_name in found.items():
        sub    = df[df["metro"] == label]
        latest = sub.iloc[-1]
        oldest = sub.iloc[0]
        yoy    = (latest["zhvi"] / sub[sub["date"] >= latest["date"] - pd.DateOffset(years=1)].iloc[0]["zhvi"] - 1) * 100
        print(
            f"  {label:<18} {full_name}\n"
            f"  {'':18} {len(sub)} months  |  "
            f"{oldest['date'].strftime('%Y-%m')} → {latest['date'].strftime('%Y-%m')}  |  "
            f"Latest: ${latest['zhvi']:>9,.0f}  |  YoY: {yoy:+.1f}%\n"
        )

    print(f"Saved {len(df)} rows → {OUTPUT}")


if __name__ == "__main__":
    main()
