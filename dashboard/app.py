import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

DATA = Path(__file__).parent.parent / "data" / "processed"
LOGO = Path(__file__).parent.parent / "assets" / "StarvestLogo.png"

st.set_page_config(page_title="Starvest", page_icon="🍊", layout="wide")
st.image(str(LOGO), width=400)
st.caption("NASA MODIS NDVI · USDA Yield Data · OJ Futures")


@st.cache_data
def load_data():
    dataset = pd.read_csv(DATA / "dataset.csv")
    backtest = pd.read_csv(DATA / "backtest_results.csv")
    with open(DATA / "model_params.json") as f:
        params = json.load(f)
    return dataset, backtest, params


@st.cache_data
def load_ndvi_raw():
    path = Path(__file__).parent.parent / "data" / "raw" / "ndvi_raw.csv"
    return pd.read_csv(path, parse_dates=["date"])


try:
    dataset, backtest, params = load_data()
    ndvi_raw = load_ndvi_raw()
except FileNotFoundError:
    st.error("Data not found. Run the pipeline first: `python src/build_dataset.py` then `python src/model.py` and `python src/backtest.py`")
    st.stop()

# ── Sidebar: year selector ──────────────────────────────────────────────────
latest_year = int(dataset["year"].max())
selected_year = st.sidebar.selectbox("Forecast year", sorted(dataset["year"].unique(), reverse=True))
row = dataset[dataset["year"] == selected_year].iloc[0]

# ── KPI row ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Season Avg NDVI", f"{row['mean_ndvi']:.4f}")
col2.metric("Predicted Yield", f"{row['yield_boxes']:,.0f}K boxes")
col3.metric("Hist. Average", f"{params['historical_avg_yield']:,.0f}K boxes",
            delta=f"{(row['yield_boxes'] - params['historical_avg_yield']):+,.0f}")

pressure = row["price_pressure"]
pressure_color = {"bullish": "🟢", "neutral": "🟡", "bearish": "🔴"}
col4.metric("Price Pressure", f"{pressure_color[pressure]} {pressure.capitalize()}")

st.divider()

# ── NDVI trend ───────────────────────────────────────────────────────────────
st.subheader("NDVI Trend — Florida Citrus Belt")
fig_ndvi = px.line(ndvi_raw, x="date", y="mean_ndvi",
                   labels={"mean_ndvi": "Mean NDVI", "date": "Date"},
                   color_discrete_sequence=["#2ecc71"])
fig_ndvi.update_layout(height=300, margin=dict(t=20))
st.plotly_chart(fig_ndvi, use_container_width=True)

# ── Yield vs historical average ──────────────────────────────────────────────
st.subheader("Yield vs Historical Average")
fig_yield = go.Figure()
fig_yield.add_bar(x=dataset["year"], y=dataset["yield_boxes"], name="Actual Yield",
                  marker_color="#f39c12")
fig_yield.add_hline(y=params["historical_avg_yield"], line_dash="dash",
                    line_color="white", annotation_text="Hist. Avg")
fig_yield.update_layout(height=300, margin=dict(t=20),
                        xaxis_title="Year", yaxis_title="1000 Boxes")
st.plotly_chart(fig_yield, use_container_width=True)

st.divider()

# ── Backtest panel ───────────────────────────────────────────────────────────
st.subheader("Backtest — Predicted vs Actual Yield")
fig_bt = go.Figure()
fig_bt.add_scatter(x=backtest["year"], y=backtest["actual_yield"],
                   mode="lines+markers", name="Actual", line=dict(color="#e74c3c"))
fig_bt.add_scatter(x=backtest["year"], y=backtest["predicted_yield"],
                   mode="lines+markers", name="Predicted", line=dict(color="#3498db", dash="dash"))
fig_bt.update_layout(height=300, margin=dict(t=20),
                     xaxis_title="Year", yaxis_title="1000 Boxes")
st.plotly_chart(fig_bt, use_container_width=True)

accuracy = (backtest["actual_pressure"] == backtest["predicted_pressure"]).mean()
st.metric("Price Pressure Accuracy (backtest)", f"{accuracy:.0%}")
st.dataframe(backtest[["year", "actual_yield", "predicted_yield",
                        "pct_error", "actual_pressure", "predicted_pressure"]]\
             .rename(columns={"pct_error": "% Error"})\
             .style.format({"actual_yield": "{:,.0f}", "predicted_yield": "{:,.0f}",
                            "% Error": "{:+.1f}%"}),
             use_container_width=True)

st.caption(f"Model R² (train): {params['r2_train']:.3f} · MAE (LOO): {params['mae_loo']:,.0f}K boxes")
