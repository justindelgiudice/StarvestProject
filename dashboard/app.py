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

# ── Glossary definitions ───────────────────────────────────────────────────────
GLOSSARY: dict[str, str] = {
    "ndvi": (
        "<b>Normalized Difference Vegetation Index</b> — satellite measure of "
        "vegetation greenness (0→1). Starvest pulls Jan–Mar MODIS MOD13Q1 NDVI "
        "at 250 m resolution over the FL citrus belt (26.5–28.5°N, 80–82.5°W) "
        "from Google Earth Engine. Healthy dense citrus groves ≈ 0.50–0.70."
    ),
    "ndvi_surprise": (
        "<b>NDVI Surprise</b> = current Jan–Mar NDVI minus the rolling 3-year "
        "baseline average. Negative → grove canopy below average (stress signal "
        "→ LONG). Positive → above average (healthy canopy → SHORT). "
        "This single number drives every trade signal."
    ),
    "ndvi_x_acres": (
        "<b>NDVI × Acres</b> = Jan–Mar NDVI × (bearing acres / 2005 peak acres). "
        "Weights greenness by the shrinking grove footprint to capture both health "
        "and structural HLB-driven collapse. Fell from ~0.54 in 2005 to ~0.17 by 2025."
    ),
    "bearing_acres": (
        "<b>Bearing acres</b> — citrus grove area old enough to produce fruit, "
        "per USDA NASS annual survey. Florida peaked at 541,800 acres in 2005. "
        "HLB disease has driven a 69% collapse to ~167,400 acres by 2025."
    ),
    "fcoj": (
        "<b>FCOJ futures (OJ=F)</b> — Frozen Concentrated Orange Juice futures "
        "traded on ICE, priced in cents per pound of soluble solids. The most "
        "liquid benchmark for FL orange supply and demand."
    ),
    "apr_sep": (
        "<b>Apr / Sep Close</b> — monthly average OJ futures close price in "
        "April (entry) and September (exit). Price direction is bullish when "
        "Sep > Apr, bearish when Sep < Apr."
    ),
    "hit_rate": (
        "<b>Hit rate</b> — % of signal years where the predicted direction "
        "(LONG → price rises, SHORT → price falls) matched the actual Apr→Sep "
        "move. A coin flip would be ~50%. Only years with a non-neutral signal "
        "are counted."
    ),
    "cum_pnl": (
        "<b>Cumulative P&amp;L</b> — running total of trade returns as % of the "
        "April entry price. LONG return = (Sep−Apr)/Apr. SHORT return = "
        "(Apr−Sep)/Apr. Excludes futures margin requirements, roll costs, "
        "and slippage."
    ),
    "yield_surprise": (
        "<b>Yield surprise</b> — year-over-year % change in Florida orange "
        "production (million 90-lb boxes per USDA NASS). A large negative "
        "surprise typically signals a supply shock and subsequent OJ price rise."
    ),
    "hlb": (
        "<b>HLB (Huanglongbing / citrus greening)</b> — fatal bacterial disease "
        "spread by the Asian citrus psyllid, first confirmed in FL in 2005. "
        "Destroys the phloem; causes small, bitter, misshapen fruit and tree "
        "death within 5–8 years. No cure exists. Primary driver of FL collapse."
    ),
    "long_signal": (
        "<b>LONG signal</b> — issued when NDVI surprise &lt; −threshold. "
        "Grove canopy is below its 3-yr average, signaling vegetation stress and "
        "an expected yield shortfall. Bullish OJ price view: buy April futures, "
        "exit at September expiry."
    ),
    "short_signal": (
        "<b>SHORT signal</b> — issued when NDVI surprise &gt; +threshold. "
        "Grove canopy is above its 3-yr average, signaling healthier trees and "
        "an expected higher yield. Bearish OJ price view: sell April futures, "
        "exit at September expiry."
    ),
}


def tip(key: str, align: str = "left") -> str:
    """Return an inline ℹ icon with a CSS hover tooltip for the given glossary key."""
    text = GLOSSARY.get(key, "")
    align_css = "left:0;transform:none;" if align == "left" else (
        "right:0;left:auto;transform:none;" if align == "right" else
        "left:50%;transform:translateX(-50%);"
    )
    return (
        f'<span class="tipwrap">'
        f'<span class="tipicon">i</span>'
        f'<span class="tipbox" style="{align_css}">{text}</span>'
        f'</span>'
    )


PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E2E8F0", family="Inter, sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(30,33,48,0.7)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="rgba(15,17,30,0.95)",
        bordercolor="rgba(255,255,255,0.12)",
        font=dict(color="#E2E8F0", family="Inter, sans-serif", size=12),
    ),
)

PLOTLY_CONFIG = {"scrollZoom": True, "displayModeBar": False}

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
/* Scroll-zoom is enabled; keep pointer cursor (no crosshair) */
.js-plotly-plot .nsewdrag,
.js-plotly-plot .ewdrag,
.js-plotly-plot .nsdrag { cursor: default !important; }
/* ── Glossary tooltip ── */
.tipwrap {
    position: relative; display: inline-block;
    vertical-align: middle; margin-left: 5px;
}
.tipicon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 15px; height: 15px; border-radius: 50%;
    background: rgba(96,165,250,0.18); border: 1px solid rgba(96,165,250,0.38);
    color: #60A5FA; font-size: 9px; font-weight: 700; font-style: italic;
    cursor: help; line-height: 1; flex-shrink: 0;
}
.tipbox {
    display: none; position: absolute;
    top: calc(100% + 7px);
    background: #1a1d2e; border: 1px solid rgba(255,255,255,0.13);
    border-radius: 10px; padding: 11px 14px; width: 260px;
    font-size: 12px; line-height: 1.65; color: #CBD5E1;
    box-shadow: 0 10px 36px rgba(0,0,0,0.55); z-index: 99999;
    pointer-events: none; white-space: normal;
    font-weight: 400; font-style: normal; text-transform: none; letter-spacing: 0;
}
.tipwrap:hover .tipbox { display: block; }
/* Prevent parent containers from clipping tooltips */
[data-testid="column"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] { overflow: visible !important; }
.card { overflow: visible !important; }
/* Glossary tab term cards */
.gterm {
    background: #1a1d2e; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 18px 20px; margin-bottom: 12px;
    border-left-width: 3px; border-left-style: solid;
}
.gterm-name { font-size: 14px; font-weight: 700; color: #E2E8F0; margin-bottom: 4px; }
.gterm-cat  { font-size: 10px; font-weight: 600; letter-spacing: .6px;
    text-transform: uppercase; padding: 2px 8px; border-radius: 999px;
    display: inline-block; margin-left: 8px; vertical-align: middle; }
.gterm-def  { font-size: 13px; color: #94A3B8; line-height: 1.7; margin-top: 8px; }
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["  Overview  ", "  NDVI Trend  ", "  Yield vs Price  ", "  Backtest  ", "  Signal  ", "  Glossary  "]
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

    card(k1, f"{latest_year} Production {tip('yield_surprise')}",
         f"{prod_25:.1f}M boxes",
         f"{(prod_25/prod_05-1)*100:.0f}% vs 2005 peak",
         ORANGE, RED)

    card(k2, f"Bearing Acres {tip('bearing_acres')}",
         f"{acres_25/1e3:.0f}K",
         f"{(acres_25/acres_05-1)*100:.0f}% vs 2005",
         BLUE, RED)

    card(k3, f"Apr {latest_year} OJ Price {tip('fcoj')}",
         f"{apr_25:.0f}¢/lb",
         f"+{(apr_25/apr_05-1)*100:.0f}% vs 2005",
         GOLD, GREEN)

    card(k4, f"Signal Hit Rate {tip('hit_rate')}",
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
        hovertemplate="%{y:.1f}M boxes<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["bearing_acres"] / 1e3,
        name="Bearing Acres (K)",
        line=dict(color=BLUE, width=2.5),
        mode="lines+markers",
        marker=dict(size=6, symbol="circle", color=BLUE, line=dict(width=1.5, color="rgba(255,255,255,0.6)")),
        hovertemplate="%{y:.0f}K acres<extra></extra>",
    ), secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["ndvi_x_acres"],
        name="NDVI × Acres (composite)",
        line=dict(color=PURPLE, width=2, dash="dot"),
        mode="lines+markers",
        marker=dict(size=5, symbol="circle", color=PURPLE, line=dict(width=1.5, color="rgba(255,255,255,0.5)")),
        hovertemplate="%{y:.3f}<extra></extra>",
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
    # Lock outer x bounds to the full data range so scroll-out stops at start position
    _x0, _x1 = df.index.min() - 0.5, df.index.max() + 0.5
    fig.update_xaxes(range=[_x0, _x1], minallowed=_x0, maxallowed=_x1)

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown(
        f'<p style="font-weight:700;margin-bottom:6px">How Starvest works:</p>'
        f'<p style="color:#94A3B8;font-size:13px;margin-bottom:10px">'
        f'Three signals feed a single Jan–Mar directional call on '
        f'FCOJ futures {tip("fcoj")} (April close → September close {tip("apr_sep")}):</p>'
        f'<table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:14px">'
        f'<thead><tr style="border-bottom:1px solid rgba(255,255,255,0.1)">'
        f'<th style="text-align:left;padding:8px 10px;color:#64748b;font-weight:600">Signal</th>'
        f'<th style="text-align:left;padding:8px 10px;color:#64748b;font-weight:600">Source</th>'
        f'<th style="text-align:left;padding:8px 10px;color:#64748b;font-weight:600">Logic</th>'
        f'</tr></thead><tbody>'
        f'<tr style="border-bottom:1px solid rgba(255,255,255,0.05)">'
        f'<td style="padding:8px 10px;color:#E2E8F0;font-weight:600">'
        f'NDVI Jan-Mar {tip("ndvi")}</td>'
        f'<td style="padding:8px 10px;color:#94A3B8">MODIS MOD13Q1 250m via GEE</td>'
        f'<td style="padding:8px 10px;color:#94A3B8">Below 3-yr avg → stress → lower yield → price up</td>'
        f'</tr>'
        f'<tr style="border-bottom:1px solid rgba(255,255,255,0.05)">'
        f'<td style="padding:8px 10px;color:#E2E8F0;font-weight:600">'
        f'NDVI Surprise {tip("ndvi_surprise")}</td>'
        f'<td style="padding:8px 10px;color:#94A3B8">NDVI vs rolling 3-yr baseline</td>'
        f'<td style="padding:8px 10px;color:#94A3B8">Magnitude of deviation → confidence in call</td>'
        f'</tr>'
        f'<tr>'
        f'<td style="padding:8px 10px;color:#E2E8F0;font-weight:600">'
        f'NDVI × Acres {tip("ndvi_x_acres")}</td>'
        f'<td style="padding:8px 10px;color:#94A3B8">NDVI weighted by grove footprint</td>'
        f'<td style="padding:8px 10px;color:#94A3B8">'
        f'Structural shrinkage (HLB {tip("hlb")}) ↓ 69% since 2005</td>'
        f'</tr>'
        f'</tbody></table>'
        f'<div style="background:rgba(148,163,184,0.07);border-left:3px solid rgba(148,163,184,0.3);'
        f'border-radius:0 8px 8px 0;padding:10px 14px;font-size:12px;color:#94A3B8;line-height:1.6">'
        f'<b style="color:#CBD5E1">Note:</b> NDVI measures vegetation greenness, not fruit count. '
        f'HLB-infected trees can appear green while producing near-zero fruit — '
        f'a fundamental limitation the NDVI × Acres composite partially corrects for.</div>',
        unsafe_allow_html=True,
    )

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
        mode="lines+markers",
        marker=dict(size=7, symbol="circle", color=GREEN, line=dict(width=1.5, color="rgba(255,255,255,0.7)")),
        hovertemplate="NDVI=%{y:.4f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["ndvi_3yr_avg"],
        name="3-yr Rolling Avg", line=dict(color=GRAY, width=1.5, dash="dash"),
        mode="lines+markers",
        marker=dict(size=5, symbol="circle-open", color=GRAY, line=dict(width=1.5, color=GRAY)),
        hovertemplate="avg=%{y:.4f}<extra></extra>",
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
        hovertemplate="surprise=%{y:.4f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=2, col=1)

    # ── Row 3: ndvi_x_acres ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ndvi_x_acres"],
        name="NDVI × Acres", line=dict(color=PURPLE, width=2.5),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.15)",
        mode="lines+markers",
        marker=dict(size=6, symbol="circle", color=PURPLE, line=dict(width=1.5, color="rgba(255,255,255,0.6)")),
        hovertemplate="%{y:.4f}<extra></extra>",
    ), row=3, col=1)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=580,
        title="MODIS NDVI · Florida Citrus Belt (26.5–28.5°N, 80–82.5°W)",
    )
    for row in [1, 2, 3]:
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", row=row, col=1)

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f'<p style="font-weight:700;margin-bottom:6px">NDVI {tip("ndvi")} interpretation:</p>'
            f'<ul style="color:#94A3B8;font-size:13px;line-height:2;margin:0;padding-left:18px">'
            f'<li>Range 0→1; healthy dense citrus groves ≈ 0.50–0.70</li>'
            f'<li>Jan–Mar captures the citrus belt <em>before</em> summer rainy season</li>'
            f'<li>Surprise {tip("ndvi_surprise")} &lt; 0 (red bar) = browner than typical → stress signal</li>'
            f'<li>Surprise &gt; 0 (green bar) = greener than typical → healthier canopy</li>'
            f'</ul>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f'<p style="font-weight:700;margin-bottom:6px">NDVI × Acres {tip("ndvi_x_acres")} composite:</p>'
            f'<ul style="color:#94A3B8;font-size:13px;line-height:2;margin:0;padding-left:18px">'
            f'<li>NDVI weighted by grove size relative to 2005 baseline</li>'
            f'<li>0.54 → 0.17 over 20 years = fewer trees <em>and</em> stressed trees</li>'
            f'<li>Better reflects OJ supply capacity than NDVI alone</li>'
            f'<li>HLB {tip("hlb")} infected trees look green but yield near-zero fruit</li>'
            f'</ul>',
            unsafe_allow_html=True,
        )

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
        hovertemplate="%{y:.1f}M boxes<extra></extra>",
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["apr_close"],
        name="Apr Close (¢/lb)", line=dict(color=BLUE, width=2),
        mode="lines+markers",
        marker=dict(size=6, symbol="circle", color=BLUE, line=dict(width=1.5, color="rgba(255,255,255,0.6)")),
        hovertemplate="Apr=%{y:.1f}¢<extra></extra>",
    ), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["sep_close"],
        name="Sep Close (¢/lb)", line=dict(color=GOLD, width=2, dash="dash"),
        mode="lines+markers",
        marker=dict(size=6, symbol="circle", color=GOLD, line=dict(width=1.5, color="rgba(255,255,255,0.6)")),
        hovertemplate="Sep=%{y:.1f}¢<extra></extra>",
    ), row=1, col=1, secondary_y=True)

    # ── Row 2: yield YoY % bars ───────────────────────────────────────────────
    yoy_colors = [
        GREEN if (pd.notna(v) and v >= 0) else RED
        for v in df["yield_yoy_pct"]
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["yield_yoy_pct"],
        name="Yield YoY %", marker_color=yoy_colors, opacity=0.85,
        hovertemplate="%{y:+.1f}%<extra></extra>",
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

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

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

    mini(m1, f"Hit Rate {tip('hit_rate')}", f"{hit:.0%}", GREEN if hit >= 0.55 else RED)
    mini(m2, "Years Tested", f"{n_ok}/{n_tot}", GRAY)
    mini(m3, f"Cum P&L {tip('cum_pnl')}", f"{cum_pnl:+.1f}%", GREEN if cum_pnl >= 0 else RED)
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
        mode="lines+markers",
        marker=dict(size=7, symbol="circle", color=pnl_colors, line=dict(width=1.5, color="rgba(255,255,255,0.6)")),
        name="Cum P&L",
        hovertemplate="%{y:+.1f}%<extra></extra>",
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
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

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

    sig_tip = tip("long_signal") if sig == "LONG" else (tip("short_signal") if sig == "SHORT" else "")
    with left:
        st.markdown(
            f'<div style="background:{sig_bg};border:1px solid {sig_color}33;border-radius:16px;'
            f'padding:32px;text-align:center;margin-bottom:16px;overflow:visible">'
            f'<div style="color:{sig_color};font-size:11px;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">'
            f'{latest_year} Signal — Jan-Mar NDVI Basis</div>'
            f'<div style="font-size:4rem;font-weight:800;color:{sig_color};line-height:1">'
            f'{sig} {sig_tip}</div>'
            f'<div style="font-size:14px;color:#94A3B8;margin-top:12px">'
            f'NDVI surprise {tip("ndvi_surprise","center")}: '
            f'<span style="color:{sig_color};font-weight:700">{surprise:+.4f}</span>'
            f' &nbsp;|&nbsp; threshold: 0.000</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # NDVI detail card
        st.markdown(
            f'<div class="card" style="text-align:left;margin-bottom:12px">'
            f'<div class="card-label">NDVI {tip("ndvi")} DETAIL · {latest_year}</div>'
            f'<table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:14px">'
            f'<tr><td style="color:#64748b;padding:4px 0">Jan-Mar NDVI</td>'
            f'    <td style="text-align:right;font-weight:600">{ndvi:.4f}</td></tr>'
            f'<tr><td style="color:#64748b;padding:4px 0">3-yr baseline</td>'
            f'    <td style="text-align:right;font-weight:600">{avg3:.4f}</td></tr>'
            f'<tr><td style="color:#64748b;padding:4px 0">Surprise {tip("ndvi_surprise","right")}</td>'
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
    fig.update_layout(**PLOTLY_LAYOUT, height=300)
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 · GLOSSARY
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown(
        '<h2 style="font-size:1.35rem;font-weight:800;margin-bottom:2px">Glossary</h2>'
        '<p style="color:#64748b;font-size:13px;margin-bottom:22px">'
        'Definitions for every signal, metric, and term used in Starvest.</p>',
        unsafe_allow_html=True,
    )

    TERMS = [
        ("NDVI",             "ndvi",           "Satellite signal",  BLUE),
        ("NDVI Surprise",    "ndvi_surprise",  "Core signal",       ORANGE),
        ("NDVI × Acres",     "ndvi_x_acres",   "Composite signal",  PURPLE),
        ("Bearing Acres",    "bearing_acres",  "Supply metric",     GREEN),
        ("FCOJ Futures",     "fcoj",           "Market",            GOLD),
        ("Apr / Sep Close",  "apr_sep",        "Price reference",   BLUE),
        ("Hit Rate",         "hit_rate",       "Backtest metric",   GREEN),
        ("Cumulative P&L",   "cum_pnl",        "Backtest metric",   GREEN),
        ("Yield Surprise",   "yield_surprise", "Supply metric",     ORANGE),
        ("HLB Disease",      "hlb",            "Context",           RED),
        ("LONG Signal",      "long_signal",    "Trade signal",      GREEN),
        ("SHORT Signal",     "short_signal",   "Trade signal",      RED),
    ]

    for i in range(0, len(TERMS), 2):
        pair = TERMS[i:i+2]
        cols = st.columns(len(pair))
        for col, (name, key, category, color) in zip(cols, pair):
            col.markdown(
                f'<div class="gterm" style="border-left-color:{color}">'
                f'<div>'
                f'<span class="gterm-name">{name}</span>'
                f'<span class="gterm-cat" style="background:{color}22;color:{color}">'
                f'{category}</span>'
                f'</div>'
                f'<div class="gterm-def">{GLOSSARY[key]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
