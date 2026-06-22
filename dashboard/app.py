import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

DATA = Path(__file__).parent.parent / "data" / "processed"
RAW  = Path(__file__).parent.parent / "data" / "raw"

st.set_page_config(page_title="Starvest", page_icon="🏠", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f8fafc; }
  [data-testid="stHeader"]           { background: #f8fafc; }
  .block-container { padding: 1.5rem 2.5rem 2rem; max-width: 1400px; }
  .kpi-card { background:#fff;border:1px solid #e2e8f0;border-radius:12px;
              padding:1.2rem 1.4rem;box-shadow:0 1px 3px rgba(0,0,0,.06); }
  .kpi-label { color:#64748b;font-size:.72rem;text-transform:uppercase;
               letter-spacing:.08em;margin-bottom:.4rem; }
  .kpi-value { color:#0f172a;font-size:1.55rem;font-weight:700;line-height:1.1; }
  .kpi-sub   { color:#94a3b8;font-size:.74rem;margin-top:.3rem; }
  .section-title { color:#64748b;font-size:.78rem;text-transform:uppercase;
                   letter-spacing:.1em;margin-bottom:.5rem;margin-top:.2rem; }
  .panel-card { background:#fff;border:1px solid #e2e8f0;border-radius:12px;
                padding:1.3rem 1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.06); }
  .signal-heating { color:#16a34a;font-weight:700; }
  .signal-cooling  { color:#dc2626;font-weight:700; }
  .signal-stable  { color:#d97706;font-weight:700; }
  #MainMenu { visibility:hidden; } footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

CHART_CFG = {"displayModeBar": "hover", "displaylogo": False}
METRO_COLORS = {
    "Miami":           "#f97316",
    "Tampa":           "#2563eb",
    "Orlando":         "#16a34a",
    "Jacksonville":    "#7c3aed",
    "Fort Lauderdale": "#0891b2",
}
SIGNAL_COLOR = {"heating": "#16a34a", "cooling": "#dc2626", "stable": "#d97706"}

def gap():
    st.markdown('<div style="margin-top:1.6rem;"></div>', unsafe_allow_html=True)

def hint(text: str):
    st.markdown(
        f'<p style="color:#94a3b8;font-size:.73rem;line-height:1.5;margin-top:.35rem;">{text}</p>',
        unsafe_allow_html=True,
    )

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;flex-direction:column;align-items:flex-start;gap:0;line-height:1;margin-bottom:.5rem;">
  <span style="font-size:2rem;font-weight:800;color:#0f172a;letter-spacing:-.03em;">Starvest</span>
  <p style="color:#64748b;font-size:.82rem;margin:3px 0 1px 0;">
    Satellite Construction Signals for Florida Real Estate</p>
  <p style="color:#94a3b8;font-size:.75rem;margin:0;">
    Sentinel-2 NDBI &nbsp;|&nbsp; Census Building Permits &nbsp;|&nbsp; Zillow ZHVI</p>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr style='border-color:#e2e8f0;margin:.8rem 0 1.2rem;'/>", unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_zhvi():
    path = RAW / "zhvi_raw.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])

@st.cache_data(ttl=3600)
def load_dataset():
    path = DATA / "dataset.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data(ttl=3600)
def load_signals():
    path = DATA / "signals.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data(ttl=3600)
def load_model_params():
    path = DATA / "model_params.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

zhvi    = load_zhvi()
dataset = load_dataset()
signals = load_signals()
params  = load_model_params()

# ── HOW IT WORKS ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;
            padding:1.1rem 1.4rem;margin-bottom:1rem;">
  <div style="color:#15803d;font-size:.7rem;font-weight:700;text-transform:uppercase;
              letter-spacing:.1em;margin-bottom:.7rem;">The Informational Edge</div>
  <div style="display:grid;grid-template-columns:1fr 24px 1fr 24px 1fr;gap:.3rem;align-items:center;margin-bottom:.65rem;">
    <div style="background:#fff;border:1px solid #bbf7d0;border-radius:8px;padding:.75rem 1rem;">
      <div style="color:#15803d;font-size:.65rem;font-weight:700;text-transform:uppercase;margin-bottom:.25rem;">① Sentinel-2 Satellite</div>
      <div style="color:#0f172a;font-size:.8rem;font-weight:600;margin-bottom:.2rem;">NDBI · every 5–16 days</div>
      <div style="color:#64748b;font-size:.7rem;line-height:1.4;">Detects built-up area and bare soil expansion across FL metro bounding boxes — a leading proxy for construction activity.</div>
    </div>
    <div style="color:#94a3b8;font-size:1.1rem;text-align:center;">→</div>
    <div style="background:#fff;border:1px solid #bbf7d0;border-radius:8px;padding:.75rem 1rem;">
      <div style="color:#15803d;font-size:.65rem;font-weight:700;text-transform:uppercase;margin-bottom:.25rem;">② Construction Signal</div>
      <div style="color:#0f172a;font-size:.8rem;font-weight:600;margin-bottom:.2rem;">Permits + NDBI combined</div>
      <div style="color:#64748b;font-size:.7rem;line-height:1.4;">Census permit counts confirm satellite signals. Together they quantify new supply entering the market 12–24 months ahead of closing data.</div>
    </div>
    <div style="color:#94a3b8;font-size:1.1rem;text-align:center;">→</div>
    <div style="background:#fff;border:1px solid #bbf7d0;border-radius:8px;padding:.75rem 1rem;">
      <div style="color:#15803d;font-size:.65rem;font-weight:700;text-transform:uppercase;margin-bottom:.25rem;">③ Price Forecast</div>
      <div style="color:#0f172a;font-size:.8rem;font-weight:600;margin-bottom:.2rem;">Heating / Stable / Cooling</div>
      <div style="color:#64748b;font-size:.7rem;line-height:1.4;">Model predicts 4-quarter forward ZHVI change per metro. Signal leads official Case-Shiller / Zillow reports by 1–2 quarters.</div>
    </div>
  </div>
  <p style="color:#166534;font-size:.71rem;margin:0;line-height:1.45;">
    <b>Timing edge:</b> Satellite revisit every 5–16 days vs. closing-based home price indices with 1–2 month publication lags.
    Construction ramp-up visible in NDBI 12–24 months before the supply hits the market and affects prices.
  </p>
</div>
""", unsafe_allow_html=True)

# ── ZHVI TIME SERIES (primary view) ──────────────────────────────────────────
if zhvi is not None:
    st.markdown('<p class="section-title">Home Value Index by Metro (Zillow ZHVI)</p>', unsafe_allow_html=True)

    metros = sorted(zhvi["metro"].unique())

    fig = go.Figure()
    for metro in metros:
        sub = zhvi[zhvi["metro"] == metro].sort_values("date")
        fig.add_scatter(
            x=sub["date"], y=sub["zhvi"],
            mode="lines", name=metro,
            line=dict(color=METRO_COLORS.get(metro, "#94a3b8"), width=2.5),
            hovertemplate=f"<b>{metro}</b><br>%{{x|%b %Y}}<br>ZHVI: $%{{y:,.0f}}<extra></extra>",
        )

    fig.update_layout(
        height=420, paper_bgcolor="#fff", plot_bgcolor="#fff",
        font=dict(color="#475569", size=11),
        margin=dict(t=10, b=30, l=10, r=10),
        hovermode="x unified",
        yaxis=dict(tickprefix="$", gridcolor="#e2e8f0", linecolor="#e2e8f0"),
        xaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155", font=dict(color="#f1f5f9", size=11)),
    )
    st.plotly_chart(fig, use_container_width=True, config=CHART_CFG)
    hint("Zillow Home Value Index (ZHVI): smoothed, seasonally adjusted median home value for all homes (SFR + condo). Middle tier (33rd–67th percentile).")

    gap()

    # ── KPI cards (latest ZHVI values) ───────────────────────────────────────
    cols = st.columns(len(metros), gap="small")
    for col, metro in zip(cols, metros):
        sub    = zhvi[zhvi["metro"] == metro].sort_values("date")
        latest = sub.iloc[-1]
        yoy_df = sub[sub["date"] >= latest["date"] - pd.DateOffset(years=1)]
        yoy    = (latest["zhvi"] / yoy_df.iloc[0]["zhvi"] - 1) * 100 if len(yoy_df) > 1 else float("nan")
        arrow  = "▲" if yoy > 0 else "▼"
        color  = "#16a34a" if yoy > 0 else "#dc2626"
        col.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">{metro}</div>
  <div class="kpi-value">${latest['zhvi']:,.0f}</div>
  <div class="kpi-sub" style="color:{color};">{arrow} {abs(yoy):.1f}% YoY</div>
</div>""", unsafe_allow_html=True)

    gap()

    # ── YoY change chart ─────────────────────────────────────────────────────
    st.markdown('<p class="section-title">Year-over-Year Home Price Change (%)</p>', unsafe_allow_html=True)
    fig2 = go.Figure()
    for metro in metros:
        sub = zhvi[zhvi["metro"] == metro].sort_values("date").copy()
        sub["yoy"] = sub["zhvi"].pct_change(12) * 100
        sub = sub.dropna(subset=["yoy"])
        fig2.add_scatter(
            x=sub["date"], y=sub["yoy"],
            mode="lines", name=metro,
            line=dict(color=METRO_COLORS.get(metro, "#94a3b8"), width=2),
            hovertemplate=f"<b>{metro}</b><br>%{{x|%b %Y}}<br>YoY: %{{y:+.1f}}%<extra></extra>",
        )
    fig2.add_hline(y=0, line_color="#94a3b8", line_dash="dot")
    fig2.update_layout(
        height=360, paper_bgcolor="#fff", plot_bgcolor="#fff",
        font=dict(color="#475569", size=11),
        margin=dict(t=10, b=30, l=10, r=10),
        hovermode="x unified",
        yaxis=dict(ticksuffix="%", gridcolor="#e2e8f0", linecolor="#e2e8f0"),
        xaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155", font=dict(color="#f1f5f9", size=11)),
    )
    st.plotly_chart(fig2, use_container_width=True, config=CHART_CFG)
    hint("Year-over-year % change in Zillow ZHVI. Positive = home values rising vs. same month prior year.")

else:
    st.info("📥 Run `python src/fetch_home_prices.py` to load Zillow ZHVI data.")

gap()

# ── MODEL SIGNALS (once pipeline is trained) ──────────────────────────────────
if signals is not None and params is not None:
    st.markdown('<p class="section-title">Market Signals — Satellite + Permit Model</p>', unsafe_allow_html=True)

    latest_q   = signals["quarter"].max()
    latest_sig = signals[signals["quarter"] == latest_q]
    sig_cols   = st.columns(len(latest_sig), gap="small")

    for col, (_, row) in zip(sig_cols, latest_sig.iterrows()):
        sig   = row["signal"]
        color = SIGNAL_COLOR.get(sig, "#64748b")
        col.markdown(f"""<div class="kpi-card">
  <div class="kpi-label">{row['metro']}</div>
  <div class="kpi-value" style="color:{color};font-size:1.1rem;">{sig.capitalize()}</div>
  <div class="kpi-sub">{row['predicted_zhvi_fwd']:+.1f}% fwd 4Q · {latest_q}</div>
</div>""", unsafe_allow_html=True)

    gap()
    col_l, col_r = st.columns([2, 1], gap="medium")
    with col_l:
        st.markdown('<p class="section-title">Predicted vs Actual Forward ZHVI Change</p>', unsafe_allow_html=True)
        fig3 = go.Figure()
        for metro in signals["metro"].unique():
            sub = signals[signals["metro"] == metro].sort_values("quarter")
            fig3.add_scatter(x=sub["quarter"], y=sub["actual_zhvi_fwd"],
                             mode="lines+markers", name=f"{metro} (actual)",
                             line=dict(color=METRO_COLORS.get(metro, "#94a3b8"), width=2))
            fig3.add_scatter(x=sub["quarter"], y=sub["predicted_zhvi_fwd"],
                             mode="lines+markers", name=f"{metro} (predicted)",
                             line=dict(color=METRO_COLORS.get(metro, "#94a3b8"), width=1.5, dash="dash"))
        fig3.update_layout(height=380, paper_bgcolor="#fff", plot_bgcolor="#fff",
                            font=dict(color="#475569", size=11), margin=dict(t=10, b=30, l=10, r=10),
                            yaxis=dict(ticksuffix="%", gridcolor="#e2e8f0"),
                            xaxis=dict(gridcolor="#e2e8f0"))
        st.plotly_chart(fig3, use_container_width=True, config=CHART_CFG)

    with col_r:
        da  = params.get("directional_accuracy", 0)
        r2  = params.get("r2_backtest", 0)
        mae = params.get("mae_backtest", 0)
        r2_color = "#16a34a" if r2 >= 0.5 else "#d97706" if r2 >= 0.3 else "#dc2626"
        st.markdown(f"""<div class="panel-card">
  <div style="color:#15803d;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.9rem;">Model Performance</div>
  <div style="display:flex;justify-content:space-between;padding:.45rem 0;border-bottom:1px solid #e2e8f0;">
    <span style="color:#64748b;font-size:.8rem;">Directional Accuracy</span>
    <span style="color:#2563eb;font-size:.85rem;font-weight:600;">{da:.0%}</span>
  </div>
  <div style="display:flex;justify-content:space-between;padding:.45rem 0;border-bottom:1px solid #e2e8f0;">
    <span style="color:#64748b;font-size:.8rem;">R² (backtest)</span>
    <span style="color:{r2_color};font-size:.85rem;font-weight:600;">{r2:.2f}</span>
  </div>
  <div style="display:flex;justify-content:space-between;padding:.45rem 0;border-bottom:1px solid #e2e8f0;">
    <span style="color:#64748b;font-size:.8rem;">MAE</span>
    <span style="color:#2563eb;font-size:.85rem;font-weight:600;">{mae:.1f} pp</span>
  </div>
  <div style="display:flex;justify-content:space-between;padding:.45rem 0;">
    <span style="color:#64748b;font-size:.8rem;">Backtest Quarters</span>
    <span style="color:#2563eb;font-size:.85rem;font-weight:600;">{params.get('n_backtest_quarters', '—')}</span>
  </div>
  <p style="color:#94a3b8;font-size:.7rem;line-height:1.5;margin-top:.8rem;border-top:1px solid #e2e8f0;padding-top:.6rem;">
    Directional accuracy = % of quarters model correctly predicted whether ZHVI rose or fell over the following year.
  </p>
</div>""", unsafe_allow_html=True)

else:
    st.info("📊 Run the full pipeline (fetch_satellite → fetch_permits → build_dataset → model → backtest) to see market signals.")

gap()

with st.expander("📖 Methodology & Data Sources"):
    st.markdown("""
**Core Thesis**

Residential construction takes 12–24 months from permit issuance to market impact on home prices.
Satellite imagery (Sentinel-2, 5–16 day revisit) detects built-up area expansion and bare-soil
disturbance *before* permits even appear in Census data — providing the earliest possible
quantitative signal on new housing supply entering each market.

**Pipeline**

| Step | Script | Output |
|------|--------|--------|
| Sentinel-2 NDBI / BSI | `fetch_satellite.py` | `data/raw/satellite_raw.csv` |
| Census building permits | `fetch_permits.py` | `data/raw/permits_raw.csv` |
| Zillow ZHVI | `fetch_home_prices.py` | `data/raw/zhvi_raw.csv` |
| Merge + features | `build_dataset.py` | `data/processed/dataset.csv` |
| Train model | `model.py` | `data/processed/model_params.json` |
| Backtest | `backtest.py` | `data/processed/signals.csv` |

**Satellite Indices**
- **NDBI** (Normalized Difference Built-up Index) = (SWIR − NIR) / (SWIR + NIR). Positive = artificial surfaces.
  Quarter-over-quarter NDBI *change* captures construction ramp-up.
- **BSI** (Bare Soil Index) captures graded/cleared land ahead of construction — a leading indicator even earlier than NDBI.

**Model**
Ridge regression: NDBI change + BSI change + permits YoY + ZHVI momentum → 4-quarter forward ZHVI % change.
Walk-forward backtest: each quarter trained only on prior quarters. Min 8 training quarters before first prediction.

**Limitations**
- Satellite indices are affected by cloud cover, seasonal vegetation, and land use changes unrelated to housing.
- Permits are issued months before construction begins; cancellation rates vary by cycle.
- ZHVI is a model-based estimate, not a transaction price index — it smooths out volatility.
- Florida-specific risks: hurricane seasons, insurance cost shocks, migration flows are not modeled.
""")
