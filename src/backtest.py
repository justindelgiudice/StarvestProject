"""
Extended backtest and signal analysis for the real estate model.

Generates per-metro market signal history:
  - "Heating" : predicted forward ZHVI change > +5%
  - "Cooling"  : predicted forward ZHVI change < -5%
  - "Stable"   : within ±5%

Outputs data/processed/signals.csv:
  metro | quarter | predicted_zhvi_fwd | signal | actual_zhvi_fwd | correct

Run after model.py.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT    = Path(__file__).parent.parent
DS_PATH = ROOT / "data" / "processed" / "dataset.csv"
PARAMS  = ROOT / "data" / "processed" / "model_params.json"
BT_PATH = ROOT / "data" / "processed" / "backtest_results.csv"
SIG_PATH = ROOT / "data" / "processed" / "signals.csv"

HEAT_THRESHOLD = 5.0   # % forward ZHVI change → Heating
COOL_THRESHOLD = -5.0  # % forward ZHVI change → Cooling


def label_signal(pct: float) -> str:
    if pct > HEAT_THRESHOLD:
        return "heating"
    elif pct < COOL_THRESHOLD:
        return "cooling"
    return "stable"


def main():
    bt = pd.read_csv(BT_PATH)
    bt["signal"]  = bt["predicted_zhvi_fwd"].apply(label_signal)
    bt["correct"] = (
        bt["signal"].apply(lambda s: 1 if s == "heating" else -1 if s == "cooling" else 0)
        == bt["actual_zhvi_fwd"].apply(lambda v: 1 if v > HEAT_THRESHOLD else -1 if v < COOL_THRESHOLD else 0)
    )

    SIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    bt.to_csv(SIG_PATH, index=False)

    print("Signal accuracy by metro:")
    for metro, grp in bt.groupby("metro"):
        acc = grp["correct"].mean()
        print(f"  {metro:<18} {acc:.0%}  ({len(grp)} quarters)")

    overall = bt["correct"].mean()
    print(f"\nOverall directional accuracy: {overall:.0%}")
    print(f"Saved → {SIG_PATH}")


if __name__ == "__main__":
    main()
