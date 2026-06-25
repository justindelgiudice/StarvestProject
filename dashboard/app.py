"""
Starvest Dashboard
Run from project root: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Starvest · OJ Signal",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Palette ───────────────────────────────────────────────────────────────────
ORANGE = "#F97316"
GREEN  = "#22C55E"
RED    = "#EF4444"
BLUE   = "#60A5FA"
GOLD   = "#FBBF24"
GRAY   = "#94A3B8"
PURPLE = "#A78BFA"

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E2E8F0", family="Inter, sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(30,33,48,0.7)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
)

# ── Data ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "starvest_data.csv", index_col="year")

raw = load_data()


def compute_signals(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Add signal, correct, and trade_ret columns at the given |ndvi_surprise| threshold."""
    d = df.copy()

    def _sig(x):
        if pd.isna(x):
            return None
        if x < -threshold:
            return "LONG"
        if x > threshold:
            return "SHORT"
        return "NEUTRAL"

    d["signal"] = d["ndvi_surprise"].apply(_sig)

    d["correct"] = d.apply(
        lambda r: (
            (r["signal"] == "LONG"  and r["price_direction"] ==  1) or
            (r["signal"] == "SHORT" and r["price_direction"] == -1)
        ) if r["signal"] not in (None, "NEUTRAL") and pd.notna(r["price_direction"]) else None,
        axis=1,
    )

    def _ret(r):
        if r["signal"] == "LONG":
            return (r["sep_close"] - r["apr_close"]) / r["apr_close"] * 100
        if r["signal"] == "SHORT":
            return (r["apr_close"] - r["sep_close"]) / r["apr_close"] * 100
        return None

    d["trade_ret"] = d.apply(_ret, axis=1)
    return d


df = compute_signals(raw, threshold=0.0)

latest_year = df.index.max()
latest      = df.loc[latest_year]
bt0         = df[df["correct"].notna()].copy()
hit0        = bt0["correct"].sum() / len(bt0) if len(bt0) else 0

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.card {
    background: #1a1d2e; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 22px 18px; text-align: center;
}
.card-label { color: #64748b; font-size: 11px; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 8px; }
.card-value { font-size: 28px; font-weight: 800; line-height: 1; }
.card-delta { font-size: 12px; margin-top: 6px; color: #64748b; }
.sig-badge {
    display: inline-block; padding: 4px 14px; border-radius: 999px;
    font-weight: 700; font-size: 13px; letter-spacing: .5px;
}
.section-header { font-size: 1.1rem; font-weight: 700; margin: 18px 0 4px; }
div[data-testid="stTabs"] > div > div > div > button {
    font-size: 14px; padding: 10px 20px; font-weight: 600;
}
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
sig       = latest["signal"]
sig_color = GREEN if sig == "LONG" else (RED if sig == "SHORT" else GRAY)
sig_bg    = "rgba(34,197,94,.15)" if sig == "LONG" else ("rgba(239,68,68,.15)" if sig == "SHORT" else "rgba(148,163,184,.15)")

hcol, scol = st.columns([4, 1])
with hcol:
    st.markdown(
        '<h1 style="font-size:2rem;font-weight:800;margin-bottom:0;letter-spacing:-1px">🍊 Starvest</h1>'
        f'<p style="color:#64748b;margin-top:2px;font-size:14px">'
        f'Florida citrus yield & OJ futures signal · 2005–{latest_year}</p>',
        unsafe_allow_html=True,
    )
with scol:
    st.markdown(
        f'<div style="text-align:right;padding-top:12px">'
        f'<div style="color:#64748b;font-size:11px;letter-spacing:1px;text-transform:uppercase">Current Signal</div>'
        f'<div style="font-size:2.2rem;font-weight:800;color:{sig_color};line-height:1.1">{sig}</div>'
        f'<div style="color:#64748b;font-size:12px">{latest_year} NDVI basis</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown('<hr style="border-color:rgba(255,255,255,0.08);margin:12px 0 18px">', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["  Overview  ", "  NDVI Trend  ", "  Yield vs Price  ", "  Backtest  ", "  Signal  "]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    prod_25  = df.loc[latest_year, "production_boxes"] / 1e6
    prod_05  = df.loc[2005, "production_boxes"] / 1e6
    acres_25 = df.loc[latest_year, "bearing_acres"]
    acres_05 = df.loc[2005, "bearing_acres"]
    apr_25   = df.loc[latest_year, "apr_close"]
    apr_05   = df.loc[2005, "apr_close"]

    k1, k2, k3, k4 = st.columns(4)

    def card(col, label, value, delta_txt, value_color, delta_color):
        col.markdown(
            f'<div class="card">'
            f'<div class="card-label">{label}</div>'
            f'<div class="card-value" style="color:{value_color}">{value}</div>'
            f'<div class="card-delta" style="color:{delta_color}">{delta_txt}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    card(k1, f"{latest_year} Production",
         f"{prod_25:.1f}M boxes",
         f"{(prod_25/prod_05-1)*100:.0f}% vs 2005 peak",
         ORANGE, RED)

    card(k2, "Bearing Acres",
         f"{acres_25/1e3:.0f}K",
         f"{(acres_25/acres_05-1)*100:.0f}% vs 2005",
         BLUE, RED)

    card(k3, f"Apr {latest_year} OJ Price",
         f"{apr_25:.0f}¢/lb",
         f"+{(apr_25/apr_05-1)*100:.0f}% vs 2005",
         GOLD, GREEN)

    card(k4, "Signal Hit Rate",
         f"{hit0:.0%}",
         f"{int(bt0['correct'].sum())}/{len(bt0)} years · threshold=0",
         PURPLE, GRAY)

    st.markdown("<br>", unsafe_allow_html=True)

    # Production + acres collapse chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=df.index, y=df["production_boxes"] / 1e6,
        name="Production (M boxes)",
        marker_color=ORANGE, opacity=0.75,
        hovertemplate="%{x}: %{y:.1f}M boxes<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["bearing_acres"] / 1e3,
        name="Bearing Acres (K)",
        line=dict(color=BLUE, width=2.5),
        mode="lines+markers", marker=dict(size=5),
        hovertemplate="%{x}: %{y:.0f}K acres<extra></extra>",
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["ndvi_x_acres"],
        name="NDVI × Acres (composite)",
        line=dict(color=PURPLE, width=2, dash="dot"),
        mode="lines",
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Florida Citrus: 20-year collapse in production, groves, and grove health",
        height=380,
    )
    fig.update_yaxes(title_text="Production (M boxes)", secondary_y=False,
                     tickfont=dict(color=ORANGE), gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="Bearing Acres (K) / NDVI×Acres", secondary_y=True,
                     tickfont=dict(color=BLUE), showgrid=False)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**How Starvest works:**
Three signals feed a single Jan-Mar directional call on OJ futures (April close → September close):

| Signal | Source | Logic |
|--------|--------|-------|
| NDVI Jan-Mar | MODIS MOD13Q1 250m via GEE | Below 3-yr avg → stress → lower yield → price up |
| NDVI Surprise | NDVI vs rolling 3-yr baseline | Magnitude of deviation → confidence in call |
| NDVI × Acres | NDVI weighted by grove footprint | Structural shrinkage (HLB disease) ↓ 69% since 2005 |

> **Note:** NDVI sees vegetation greenness, not fruit count. HLB-infected trees can appear green while producing near-zero fruit — a fundamental limitation the `ndvi_x_acres` composite partially corrects for.
""")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · NDVI TREND
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.45, 0.28, 0.27],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Jan–Mar NDVI vs 3-Year Baseline", "NDVI Surprise (vs rolling avg)", "NDVI × Grove Footprint"),
    )

    # ── Row 1: NDVI line + 3yr avg ──────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ndvi_jan_mar"],
        name="NDVI (Jan-Mar)", line=dict(color=GREEN, width=2.5),
        mode="lines+markers", marker=dict(size=6),
        hovertemplate="%{x}: NDVI=%{y:.4f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["ndvi_3yr_avg"],
        name="3-yr Rolling Avg", line=dict(color=GRAY, width=1.5, dash="dash"),
        hovertemplate="%{x}: avg=%{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # shade the gap
    fig.add_trace(go.Scatter(
        x=list(df.index) + list(df.index[::-1]),
        y=list(df["ndvi_jan_mar"]) + list(df["ndvi_3yr_avg"][::-1]),
        fill="toself",
        fillcolor="rgba(34,197,94,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # ── Row 2: surprise bars ─────────────────────────────────────────────────
    surprise_colors = [
        GREEN if (pd.notna(v) and v >= 0) else RED
        for v in df["ndvi_surprise"]
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["ndvi_surprise"],
        name="NDVI Surprise",
        marker_color=surprise_colors, opacity=0.85,
        hovertemplate="%{x}: surprise=%{y:.4f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=2, col=1)

    # ── Row 3: ndvi_x_acres ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ndvi_x_acres"],
        name="NDVI × Acres", line=dict(color=PURPLE, width=2.5),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.15)",
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ), row=3, col=1)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=580,
        title="MODIS NDVI · Florida Citrus Belt (26.5–28.5°N, 80–82.5°W)",
    )
    for row in [1, 2, 3]:
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", row=row, col=1)

    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**NDVI interpretation:**
- Range 0→1; healthy dense vegetation ≈ 0.5–0.7
- Jan-Mar captures the FL citrus belt *before* summer rainy season
- Surprise < 0 (red bar) = browner than typical → potential stress signal
- Surprise > 0 (green bar) = greener than typical → healthier canopy
""")
    with col_b:
        st.markdown("""
**NDVI × Acres (composite):**
- NDVI is normalized by grove size relative to 2005 baseline
- 0.54 → 0.17 over 20 years = both fewer trees *and* stressed trees
- This better reflects total OJ supply capacity than NDVI alone
- HLB-infected trees look green but produce almost no fruit
""")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · YIELD vs PRICE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    # Row 1: production (left axis) + OJ price lines (right axis)
    # Row 2: YoY % bars — specs enables secondary_y per row
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.62, 0.38],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Production vs OJ Futures Price", "Year-over-Year Yield Change (%)"),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )

    # ── Row 1: production bars + price lines ─────────────────────────────────
    fig.add_trace(go.Bar(
        x=df.index, y=df["production_boxes"] / 1e6,
        name="Production (M boxes)",
        marker_color=ORANGE, opacity=0.65,
        hovertemplate="%{x}: %{y:.1f}M boxes<extra></extra>",
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["apr_close"],
        name="Apr Close (¢/lb)", line=dict(color=BLUE, width=2),
        mode="lines+markers", marker=dict(size=5),
        hovertemplate="%{x}: Apr=%{y:.1f}¢<extra></extra>",
    ), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["sep_close"],
        name="Sep Close (¢/lb)", line=dict(color=GOLD, width=2, dash="dash"),
        mode="lines+markers", marker=dict(size=5),
        hovertemplate="%{x}: Sep=%{y:.1f}¢<extra></extra>",
    ), row=1, col=1, secondary_y=True)

    # ── Row 2: yield YoY % bars ───────────────────────────────────────────────
    yoy_colors = [
        GREEN if (pd.notna(v) and v >= 0) else RED
        for v in df["yield_yoy_pct"]
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["yield_yoy_pct"],
        name="Yield YoY %", marker_color=yoy_colors, opacity=0.85,
        hovertemplate="%{x}: %{y:+.1f}%<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=2, col=1)

    fig.update_yaxes(title_text="Production (M boxes)", secondary_y=False, row=1, col=1,
                     tickfont=dict(color=ORANGE), gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="OJ Price (¢/lb)", secondary_y=True, row=1, col=1,
                     tickfont=dict(color=BLUE), showgrid=False)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=540,
    )
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Key events annotation
    st.markdown("""
**Key supply events in the data:**

| Year | Event | Production | OJ Price Direction |
|------|--------|-----------|-------------------|
| 2007 | Hard freeze Jan 2007 | 129M → supply shock | ↑ to 172¢ Apr (prior season premium) |
| 2018 | Hurricane Irma (Sep 2017) | 45M boxes — 35% crash | Price ultimately ↑ Apr→Sep |
| 2023 | Hurricane Ian (Oct 2022) + HLB | **15.8M** — record low | ↑ 276¢ → 341¢ (+24%) |
| 2024 | Supply shock ongoing | 18M | ↑ 369¢ → 497¢ (+35%) |
| 2025 | Continued HLB collapse | 12.3M | ↓ 267¢ → 241¢ (demand destruction?) |
""")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("**Signal:** `LONG` when NDVI surprise < −threshold · `SHORT` when NDVI surprise > +threshold")
    threshold = st.slider(
        "NDVI Surprise threshold (±)",
        min_value=0.000, max_value=0.025, value=0.000, step=0.005,
        format="%.3f",
        help="Only trade years where |ndvi_surprise| exceeds this value. Higher = fewer trades, potentially higher hit rate.",
    )

    dft = compute_signals(raw, threshold=threshold)
    bt  = dft[dft["correct"].notna()].copy()

    n_long  = (bt["signal"] == "LONG").sum()
    n_short = (bt["signal"] == "SHORT").sum()
    n_ok    = int(bt["correct"].sum())
    n_tot   = len(bt)
    hit     = n_ok / n_tot if n_tot else 0
    cum_pnl = bt["trade_ret"].sum()
    avg_win = bt.loc[bt["correct"] == True,  "trade_ret"].mean() if n_ok > 0 else 0
    avg_los = bt.loc[bt["correct"] == False, "trade_ret"].mean() if (n_tot - n_ok) > 0 else 0

    m1, m2, m3, m4, m5 = st.columns(5)

    def mini(col, label, val, color=GRAY):
        col.markdown(
            f'<div class="card"><div class="card-label">{label}</div>'
            f'<div class="card-value" style="font-size:22px;color:{color}">{val}</div></div>',
            unsafe_allow_html=True,
        )

    mini(m1, "Hit Rate", f"{hit:.0%}", GREEN if hit >= 0.55 else RED)
    mini(m2, "Years Tested", f"{n_ok}/{n_tot}", GRAY)
    mini(m3, "Cum P&L", f"{cum_pnl:+.1f}%", GREEN if cum_pnl >= 0 else RED)
    mini(m4, "Avg Win", f"{avg_win:+.1f}%", GREEN)
    mini(m5, "Avg Loss", f"{avg_los:+.1f}%", RED)

    st.markdown("<br>", unsafe_allow_html=True)

    # Cumulative P&L chart
    bt_sorted = bt.sort_index()
    bt_sorted["cum_pnl"] = bt_sorted["trade_ret"].cumsum()

    pnl_colors = [GREEN if v >= 0 else RED for v in bt_sorted["cum_pnl"]]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Cumulative P&L (entry price %)", "Signal Distribution"),
        column_widths=[0.65, 0.35],
        specs=[[{"type": "xy"}, {"type": "domain"}]],
    )

    # Cum P&L line
    fig.add_trace(go.Scatter(
        x=bt_sorted.index, y=bt_sorted["cum_pnl"],
        line=dict(color=BLUE, width=2.5),
        fill="tozeroy", fillcolor="rgba(96,165,250,0.12)",
        mode="lines+markers", marker=dict(size=6, color=pnl_colors),
        name="Cum P&L",
        hovertemplate="%{x}: %{y:+.1f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=1, col=1)

    # Signal distribution donut
    counts = [n_ok, n_tot - n_ok, len(dft) - n_tot]
    labels = ["Correct", "Wrong", "No signal"]
    colors = [GREEN, RED, GRAY]
    fig.add_trace(go.Pie(
        values=counts, labels=labels,
        marker_colors=colors,
        hole=0.55,
        textfont_size=12,
        hovertemplate="%{label}: %{value}<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(**PLOTLY_LAYOUT, height=340)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", row=1, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Year-by-year table
    st.markdown("**Year-by-year breakdown**")
    display_df = dft[dft["signal"].notna() & (dft["signal"] != "NEUTRAL")].copy()
    display_df = display_df[[
        "ndvi_surprise", "signal", "apr_close", "sep_close",
        "price_direction", "correct", "trade_ret"
    ]].rename(columns={
        "ndvi_surprise":  "NDVI Surprise",
        "signal":         "Signal",
        "apr_close":      "Apr Close (¢)",
        "sep_close":      "Sep Close (¢)",
        "price_direction":"Price Dir",
        "correct":        "Correct",
        "trade_ret":      "Return (%)",
    })
    display_df["Correct"] = display_df["Correct"].map({True: "✓", False: "✗"})
    display_df["Price Dir"] = display_df["Price Dir"].map({1.0: "↑", -1.0: "↓"})
    display_df["NDVI Surprise"] = display_df["NDVI Surprise"].map("{:+.4f}".format)
    display_df["Return (%)"]    = display_df["Return (%)"].map("{:+.1f}%".format)

    st.dataframe(display_df, use_container_width=True, height=420)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    sig       = latest["signal"]
    surprise  = latest["ndvi_surprise"]
    apr       = latest["apr_close"]
    sep       = latest["sep_close"]
    direction = latest["price_direction"]
    correct   = latest["correct"]
    ndvi      = latest["ndvi_jan_mar"]
    avg3      = latest["ndvi_3yr_avg"]

    sig_color = GREEN if sig == "LONG" else (RED if sig == "SHORT" else GRAY)
    sig_bg    = "rgba(34,197,94,.12)" if sig == "LONG" else "rgba(239,68,68,.12)"
    outcome_color = GREEN if correct else RED
    outcome_txt   = "CORRECT ✓" if correct else "INCORRECT ✗"

    left, right = st.columns([1, 1])

    with left:
        st.markdown(
            f'<div style="background:{sig_bg};border:1px solid {sig_color}33;border-radius:16px;'
            f'padding:32px;text-align:center;margin-bottom:16px">'
            f'<div style="color:{sig_color};font-size:11px;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">'
            f'{latest_year} Signal — Jan-Mar NDVI Basis</div>'
            f'<div style="font-size:4rem;font-weight:800;color:{sig_color};line-height:1">{sig}</div>'
            f'<div style="font-size:14px;color:#94A3B8;margin-top:12px">'
            f'NDVI surprise: <span style="color:{sig_color};font-weight:700">{surprise:+.4f}</span>'
            f' &nbsp;|&nbsp; threshold: 0.000</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # NDVI detail card
        st.markdown(
            f'<div class="card" style="text-align:left;margin-bottom:12px">'
            f'<div class="card-label">NDVI DETAIL · {latest_year}</div>'
            f'<table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:14px">'
            f'<tr><td style="color:#64748b;padding:4px 0">Jan-Mar NDVI</td>'
            f'    <td style="text-align:right;font-weight:600">{ndvi:.4f}</td></tr>'
            f'<tr><td style="color:#64748b;padding:4px 0">3-yr baseline</td>'
            f'    <td style="text-align:right;font-weight:600">{avg3:.4f}</td></tr>'
            f'<tr><td style="color:#64748b;padding:4px 0">Surprise</td>'
            f'    <td style="text-align:right;font-weight:700;color:{sig_color}">{surprise:+.4f}</td></tr>'
            f'<tr><td style="color:#64748b;padding:4px 0">Signal</td>'
            f'    <td style="text-align:right;font-weight:700;color:{sig_color}">{sig}</td></tr>'
            f'</table></div>',
            unsafe_allow_html=True,
        )

        # Outcome card
        actual_move = ((sep - apr) / apr * 100) if pd.notna(apr) and pd.notna(sep) else None
        st.markdown(
            f'<div class="card" style="text-align:left">'
            f'<div class="card-label">OUTCOME · {latest_year}</div>'
            f'<table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:14px">'
            f'<tr><td style="color:#64748b;padding:4px 0">Apr close</td>'
            f'    <td style="text-align:right;font-weight:600">{apr:.1f}¢</td></tr>'
            f'<tr><td style="color:#64748b;padding:4px 0">Sep close</td>'
            f'    <td style="text-align:right;font-weight:600">{sep:.1f}¢</td></tr>'
            f'<tr><td style="color:#64748b;padding:4px 0">Actual move</td>'
            f'    <td style="text-align:right;font-weight:600">'
            f'    {actual_move:+.1f}%</td></tr>'
            f'<tr><td style="color:#64748b;padding:4px 0">Signal outcome</td>'
            f'    <td style="text-align:right;font-weight:700;color:{outcome_color}">{outcome_txt}</td></tr>'
            f'</table></div>',
            unsafe_allow_html=True,
        )

    with right:
        # Last 5 years signal history
        st.markdown("**Signal history — last 5 completed years**")
        hist5 = df[df["signal"].notna() & (df["signal"] != "NEUTRAL")].tail(8).copy()

        for yr, row in hist5.iterrows():
            s     = row["signal"]
            c     = row["correct"]
            sc    = GREEN if s == "LONG" else RED
            cc    = GREEN if c else RED
            cm    = "✓" if c else "✗"
            move  = (row["sep_close"] - row["apr_close"]) / row["apr_close"] * 100
            st.markdown(
                f'<div style="display:flex;align-items:center;background:#1a1d2e;'
                f'border-radius:10px;padding:12px 16px;margin-bottom:8px;'
                f'border-left:3px solid {sc}">'
                f'<div style="font-weight:700;font-size:16px;width:48px">{yr}</div>'
                f'<div style="background:{sc}22;color:{sc};font-weight:700;'
                f'padding:3px 10px;border-radius:999px;font-size:12px;width:60px;text-align:center">{s}</div>'
                f'<div style="flex:1;text-align:center;color:#94A3B8;font-size:13px">'
                f'surprise: <b style="color:{sc}">{row["ndvi_surprise"]:+.4f}</b></div>'
                f'<div style="text-align:center;color:#94A3B8;font-size:13px;width:80px">'
                f'{row["apr_close"]:.0f}¢→{row["sep_close"]:.0f}¢<br>'
                f'<span style="font-size:11px;color:{"#22C55E" if move>=0 else "#EF4444"}">{move:+.1f}%</span>'
                f'</div>'
                f'<div style="font-size:20px;color:{cc};width:32px;text-align:right">{cm}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Forward signal note
        st.markdown(
            f'<div style="background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.3);'
            f'border-radius:12px;padding:18px">'
            f'<div style="color:{GOLD};font-weight:700;font-size:13px;margin-bottom:8px">⚡ 2026 Forward Signal</div>'
            f'<div style="color:#94A3B8;font-size:13px;line-height:1.6">'
            f'The 2026 signal (Apr→Sep 2026 OJ direction) requires <strong>Jan–Mar 2026 NDVI</strong> '
            f'from GEE — data is available now.<br><br>'
            f'<code style="background:#0f1117;padding:4px 8px;border-radius:4px;font-size:12px">'
            f'python data_pipeline.py  # set END_YEAR=2026</code>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Signal gauge (mini)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**NDVI Surprise — signal strength gauge**")

    valid = df[df["ndvi_surprise"].notna()]
    gauge_min = float(valid["ndvi_surprise"].min()) * 1.2
    gauge_max = float(valid["ndvi_surprise"].max()) * 1.2

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=float(surprise),
        delta={"reference": 0, "valueformat": ".4f"},
        number={"valueformat": ".4f", "font": {"size": 36}},
        gauge={
            "axis": {"range": [gauge_min, gauge_max], "tickformat": ".3f",
                     "tickfont": {"size": 11}},
            "bar": {"color": sig_color, "thickness": 0.25},
            "bgcolor": "rgba(30,33,48,0.5)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [gauge_min, -threshold], "color": "rgba(239,68,68,0.15)"},
                {"range": [-threshold, threshold], "color": "rgba(148,163,184,0.1)"},
                {"range": [threshold, gauge_max],  "color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {
                "line": {"color": GOLD, "width": 2},
                "thickness": 0.8,
                "value": 0,
            },
        },
        title={"text": f"NDVI Surprise · {latest_year}<br><span style='font-size:12px;color:#94A3B8'>"
                       f"Red zone = LONG signal &nbsp;|&nbsp; Green zone = SHORT signal</span>"},
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=300,
        margin=dict(l=40, r=40, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
