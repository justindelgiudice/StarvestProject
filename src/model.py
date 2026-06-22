"""
Train home price direction model: satellite + permit signals → ZHVI forward change.

Target variable: zhvi_fwd_4q — % ZHVI change over the next 4 quarters.
Features:
  - ndbi_chg       : quarter-over-quarter NDBI change (construction intensity signal)
  - bsi_chg        : quarter-over-quarter bare-soil index change
  - permits_yoy    : trailing 12-month permit count YoY change
  - zhvi_yoy       : current ZHVI momentum
  - metro (dummies): market fixed effects

Walk-forward cross-validation: train on all data before each test quarter.
Outputs data/processed/model_params.json and data/processed/backtest_results.csv.

Note: requires build_dataset.py to have run first.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

ROOT    = Path(__file__).parent.parent
DS_PATH = ROOT / "data" / "processed" / "dataset.csv"
PARAMS  = ROOT / "data" / "processed" / "model_params.json"
BT_PATH = ROOT / "data" / "processed" / "backtest_results.csv"

FEATURES = ["ndbi_chg", "bsi_chg", "permits_yoy", "zhvi_yoy"]
TARGET   = "zhvi_fwd_4q"
MIN_TRAIN_QUARTERS = 8


def train(df: pd.DataFrame):
    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    quarters = sorted(df["quarter"].unique())

    if len(quarters) < MIN_TRAIN_QUARTERS + 1:
        raise ValueError(f"Need at least {MIN_TRAIN_QUARTERS + 1} quarters of data.")

    # ── Walk-forward backtest ─────────────────────────────────────────────────
    bt_rows = []
    for i, test_q in enumerate(quarters[MIN_TRAIN_QUARTERS:], start=MIN_TRAIN_QUARTERS):
        train_mask = df["quarter"] < test_q
        test_mask  = df["quarter"] == test_q

        X_tr = df.loc[train_mask, FEATURES].values
        y_tr = df.loc[train_mask, TARGET].values
        X_te = df.loc[test_mask,  FEATURES].values
        y_te = df.loc[test_mask,  TARGET].values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        mdl = Ridge(alpha=1.0).fit(X_tr_s, y_tr)
        preds = mdl.predict(X_te_s)

        for idx, (metro, actual, pred) in enumerate(
            zip(df.loc[test_mask, "metro"], y_te, preds)
        ):
            bt_rows.append({"quarter": test_q, "metro": metro,
                            "actual_zhvi_fwd": actual, "predicted_zhvi_fwd": pred})

    bt = pd.DataFrame(bt_rows)
    r2  = r2_score(bt["actual_zhvi_fwd"], bt["predicted_zhvi_fwd"])
    mae = mean_absolute_error(bt["actual_zhvi_fwd"], bt["predicted_zhvi_fwd"])
    dir_acc = ((bt["actual_zhvi_fwd"] > 0) == (bt["predicted_zhvi_fwd"] > 0)).mean()

    # ── Final model on all data ───────────────────────────────────────────────
    X_all = df[FEATURES].values
    y_all = df[TARGET].values
    scaler_final = StandardScaler()
    X_all_s = scaler_final.fit_transform(X_all)
    mdl_final = Ridge(alpha=1.0).fit(X_all_s, y_all)

    params = {
        "features":           FEATURES,
        "coefficients":       dict(zip(FEATURES, mdl_final.coef_.tolist())),
        "intercept":          float(mdl_final.intercept_),
        "scaler_mean":        scaler_final.mean_.tolist(),
        "scaler_scale":       scaler_final.scale_.tolist(),
        "r2_backtest":        round(r2, 4),
        "mae_backtest":       round(mae, 2),
        "directional_accuracy": round(float(dir_acc), 4),
        "n_backtest_quarters": len(bt["quarter"].unique()),
        "last_quarter":       quarters[-1],
    }

    return params, bt


def main():
    df = pd.read_csv(DS_PATH)
    params, bt = train(df)

    PARAMS.parent.mkdir(parents=True, exist_ok=True)
    with open(PARAMS, "w") as f:
        json.dump(params, f, indent=2)

    bt.to_csv(BT_PATH, index=False)

    print(f"Backtest R²:             {params['r2_backtest']:.3f}")
    print(f"Backtest MAE:            {params['mae_backtest']:.1f} pp")
    print(f"Directional accuracy:    {params['directional_accuracy']:.0%}")
    print(f"Backtest quarters:       {params['n_backtest_quarters']}")
    print(f"\nCoefficients:")
    for feat, coef in params["coefficients"].items():
        print(f"  {feat:<20} {coef:+.4f}")
    print(f"\nSaved → {PARAMS}")
    print(f"Saved → {BT_PATH}")


if __name__ == "__main__":
    main()
