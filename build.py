"""
Build static dashboard → docs/index.html
Run from project root: python build.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Palette ───────────────────────────────────────────────────────────────────
ORANGE = "#F97316"
GREEN  = "#22C55E"
RED    = "#EF4444"
BLUE   = "#60A5FA"
GOLD   = "#FBBF24"
GRAY   = "#94A3B8"
PURPLE = "#A78BFA"

# ── Glossary ──────────────────────────────────────────────────────────────────
GLOSSARY = {
    "ndvi": "<b>Normalized Difference Vegetation Index</b> — satellite measure of vegetation greenness (0→1). Starvest pulls Jan–Mar MODIS MOD13Q1 NDVI at 250 m resolution over the FL citrus belt (26.5–28.5°N, 80–82.5°W) from Google Earth Engine. Healthy dense citrus groves ≈ 0.50–0.70.",
    "ndvi_surprise": "<b>NDVI Surprise</b> = current Jan–Mar NDVI minus the rolling 3-year baseline average. Negative → grove canopy below average (stress signal → LONG). Positive → above average (healthy canopy → SHORT). This single number drives every trade signal.",
    "ndvi_x_acres": "<b>NDVI × Acres</b> = Jan–Mar NDVI × (bearing acres / 2005 peak acres). Weights greenness by the shrinking grove footprint to capture both health and structural HLB-driven collapse. Fell from ~0.54 in 2005 to ~0.17 by 2025.",
    "bearing_acres": "<b>Bearing acres</b> — citrus grove area old enough to produce fruit, per USDA NASS annual survey. Florida peaked at 541,800 acres in 2005. HLB disease has driven a 69% collapse to ~167,400 acres by 2025.",
    "fcoj": "<b>FCOJ futures (OJ=F)</b> — Frozen Concentrated Orange Juice futures traded on ICE, priced in cents per pound of soluble solids. The most liquid benchmark for FL orange supply and demand.",
    "apr_sep": "<b>Apr / Sep Close</b> — monthly average OJ futures close price in April (entry) and September (exit). Price direction is bullish when Sep > Apr, bearish when Sep < Apr.",
    "hit_rate": "<b>Hit rate</b> — % of signal years where the predicted direction (LONG → price rises, SHORT → price falls) matched the actual Apr→Sep move. A coin flip would be ~50%. Only years with a non-neutral signal are counted.",
    "cum_pnl": "<b>Cumulative P&amp;L</b> — running total of trade returns as % of the April entry price. LONG return = (Sep−Apr)/Apr. SHORT return = (Apr−Sep)/Apr. Excludes futures margin requirements, roll costs, and slippage.",
    "yield_surprise": "<b>Yield surprise</b> — year-over-year % change in Florida orange production (million 90-lb boxes per USDA NASS). A large negative surprise typically signals a supply shock and subsequent OJ price rise.",
    "hlb": "<b>HLB (Huanglongbing / citrus greening)</b> — fatal bacterial disease spread by the Asian citrus psyllid, first confirmed in FL in 2005. Destroys the phloem; causes small, bitter, misshapen fruit and tree death within 5–8 years. No cure exists. Primary driver of FL collapse.",
    "long_signal": "<b>LONG signal</b> — issued when NDVI surprise &lt; −threshold. Grove canopy is below its 3-yr average, signaling vegetation stress and an expected yield shortfall. Bullish OJ price view: buy April futures, exit at September expiry.",
    "short_signal": "<b>SHORT signal</b> — issued when NDVI surprise &gt; +threshold. Grove canopy is above its 3-yr average, signaling healthier trees and an expected higher yield. Bearish OJ price view: sell April futures, exit at September expiry.",
}

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

PLOTLY_CONFIG = dict(scrollZoom=False, displayModeBar="hover", displaylogo=False,
                     modeBarButtonsToRemove=["pan2d","select2d","lasso2d","autoScale2d",
                                             "hoverClosestCartesian","hoverCompareCartesian",
                                             "toggleSpikelines","toImage"])
PLOTLY_CONFIG_STATIC = dict(scrollZoom=False, displayModeBar=False)

# ── Data ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
raw = pd.read_csv(ROOT / "starvest_data.csv", index_col="year")

def compute_signals(df, threshold=0.0):
    d = df.copy()
    def _sig(x):
        if pd.isna(x): return None
        if x < -threshold: return "LONG"
        if x > threshold: return "SHORT"
        return "NEUTRAL"
    d["signal"] = d["ndvi_surprise"].apply(_sig)
    d["correct"] = d.apply(
        lambda r: (
            (r["signal"] == "LONG"  and r["price_direction"] ==  1) or
            (r["signal"] == "SHORT" and r["price_direction"] == -1)
        ) if r["signal"] in ("LONG", "SHORT") and pd.notna(r["price_direction"]) else None,
        axis=1,
    )
    def _ret(r):
        if r["signal"] == "LONG":  return (r["sep_close"] - r["apr_close"]) / r["apr_close"] * 100
        if r["signal"] == "SHORT": return (r["apr_close"] - r["sep_close"]) / r["apr_close"] * 100
        return None
    d["trade_ret"] = d.apply(_ret, axis=1)
    return d

df = compute_signals(raw)
latest_year = df.index.max()
latest = df.loc[latest_year]
_rain_mean = raw["fl_rainfall_jan_mar_inches"].mean() if "fl_rainfall_jan_mar_inches" in raw.columns else None
bt0 = df[df["correct"].notna()].copy()
hit0 = bt0["correct"].sum() / len(bt0) if len(bt0) else 0
_X_MIN = float(df.index.min()) - 0.5
_X_MAX = float(df.index.max()) + 0.5

# ── Helpers ───────────────────────────────────────────────────────────────────
def tip(key, align="left"):
    text = GLOSSARY.get(key, "")
    align_css = "left:0;transform:none;" if align == "left" else (
        "right:0;left:auto;transform:none;" if align == "right" else
        "left:50%;transform:translateX(-50%);"
    )
    return (f'<span class="tipwrap"><span class="tipicon">i</span>'
            f'<span class="tipbox" style="{align_css}">{text}</span></span>')

def fig_html(fig, config=None, height=None):
    if height:
        fig.update_layout(height=height)
    cfg = config or PLOTLY_CONFIG
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config=cfg, div_id=None)

# ── Build figures ─────────────────────────────────────────────────────────────

# Tab 1 — Overview chart
fig_overview = make_subplots(specs=[[{"secondary_y": True}]])
fig_overview.add_trace(go.Bar(
    x=df.index, y=df["production_boxes"]/1e6, name="Production (M boxes)",
    marker_color=ORANGE, opacity=0.75,
    hovertemplate="%{y:.1f}M boxes<extra></extra>",
), secondary_y=False)
fig_overview.add_trace(go.Scatter(
    x=df.index, y=df["bearing_acres"]/1e3, name="Bearing Acres (K)",
    line=dict(color=BLUE, width=2.5), mode="lines+markers",
    marker=dict(size=6, color=BLUE, line=dict(width=1.5, color="rgba(255,255,255,0.6)")),
    hovertemplate="%{y:.0f}K acres<extra></extra>",
), secondary_y=True)
fig_overview.add_trace(go.Scatter(
    x=df.index, y=df["ndvi_x_acres"], name="NDVI × Acres (composite)",
    line=dict(color=PURPLE, width=2, dash="dot"), mode="lines+markers",
    marker=dict(size=5, color=PURPLE, line=dict(width=1.5, color="rgba(255,255,255,0.5)")),
    hovertemplate="%{y:.3f}<extra></extra>",
), secondary_y=True)
fig_overview.update_layout(**PLOTLY_LAYOUT,
    title="Florida Citrus: 20-year collapse in production, groves, and grove health", height=380)
fig_overview.update_yaxes(title_text="Production (M boxes)", secondary_y=False,
    tickfont=dict(color=ORANGE), gridcolor="rgba(255,255,255,0.05)")
fig_overview.update_yaxes(title_text="Bearing Acres (K) / NDVI×Acres", secondary_y=True,
    tickfont=dict(color=BLUE), showgrid=False)
fig_overview.update_xaxes(minallowed=_X_MIN, maxallowed=_X_MAX)

# Tab 2 — NDVI Trend
fig_ndvi = make_subplots(rows=3, cols=1, row_heights=[0.45, 0.28, 0.27],
    shared_xaxes=True, vertical_spacing=0.06,
    subplot_titles=("Jan–Mar NDVI vs 3-Year Baseline", "NDVI Surprise (vs rolling avg)", "NDVI × Grove Footprint"))
fig_ndvi.add_trace(go.Scatter(x=df.index, y=df["ndvi_jan_mar"], name="NDVI (Jan-Mar)",
    line=dict(color=GREEN, width=2.5), mode="lines+markers",
    marker=dict(size=7, color=GREEN, line=dict(width=1.5, color="rgba(255,255,255,0.7)")),
    hovertemplate="NDVI=%{y:.4f}<extra></extra>"), row=1, col=1)
fig_ndvi.add_trace(go.Scatter(x=df.index, y=df["ndvi_3yr_avg"], name="3-yr Rolling Avg",
    line=dict(color=GRAY, width=1.5, dash="dash"), mode="lines+markers",
    marker=dict(size=5, symbol="circle-open", color=GRAY, line=dict(width=1.5, color=GRAY)),
    hovertemplate="avg=%{y:.4f}<extra></extra>"), row=1, col=1)
_shade = df[df["ndvi_3yr_avg"].notna()]
fig_ndvi.add_trace(go.Scatter(
    x=list(_shade.index)+list(_shade.index[::-1]),
    y=list(_shade["ndvi_jan_mar"])+list(_shade["ndvi_3yr_avg"][::-1]),
    fill="toself", fillcolor="rgba(34,197,94,0.08)",
    line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"), row=1, col=1)
surprise_colors = [GREEN if (pd.notna(v) and v >= 0) else RED for v in df["ndvi_surprise"]]
fig_ndvi.add_trace(go.Bar(x=df.index, y=df["ndvi_surprise"], name="NDVI Surprise",
    marker_color=surprise_colors, opacity=0.85,
    hovertemplate="surprise=%{y:.4f}<extra></extra>"), row=2, col=1)
fig_ndvi.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=2, col=1)
fig_ndvi.add_trace(go.Scatter(x=df.index, y=df["ndvi_x_acres"], name="NDVI × Acres",
    line=dict(color=PURPLE, width=2.5), fill="tozeroy", fillcolor="rgba(167,139,250,0.15)",
    mode="lines+markers",
    marker=dict(size=6, color=PURPLE, line=dict(width=1.5, color="rgba(255,255,255,0.6)")),
    hovertemplate="%{y:.4f}<extra></extra>"), row=3, col=1)
fig_ndvi.update_layout(**PLOTLY_LAYOUT, height=580,
    title="MODIS NDVI · Florida Citrus Belt (26.5–28.5°N, 80–82.5°W)")
for r in [1,2,3]:
    fig_ndvi.update_yaxes(gridcolor="rgba(255,255,255,0.05)", row=r, col=1)
fig_ndvi.update_xaxes(minallowed=_X_MIN, maxallowed=_X_MAX)

# Tab 3 — Yield vs Price
fig_yp = make_subplots(rows=2, cols=1, row_heights=[0.62, 0.38], shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("Production vs OJ Futures Price", "Year-over-Year Yield Change (%)"),
    specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
fig_yp.add_trace(go.Bar(x=df.index, y=df["production_boxes"]/1e6, name="Production (M boxes)",
    marker_color=ORANGE, opacity=0.65,
    hovertemplate="%{y:.1f}M boxes<extra></extra>"), row=1, col=1, secondary_y=False)
fig_yp.add_trace(go.Scatter(x=df.index, y=df["apr_close"], name="Apr Close (¢/lb)",
    line=dict(color=BLUE, width=2), mode="lines+markers",
    marker=dict(size=6, color=BLUE, line=dict(width=1.5, color="rgba(255,255,255,0.6)")),
    hovertemplate="Apr=%{y:.1f}¢<extra></extra>"), row=1, col=1, secondary_y=True)
fig_yp.add_trace(go.Scatter(x=df.index, y=df["sep_close"], name="Sep Close (¢/lb)",
    line=dict(color=GOLD, width=2, dash="dash"), mode="lines+markers",
    marker=dict(size=6, color=GOLD, line=dict(width=1.5, color="rgba(255,255,255,0.6)")),
    hovertemplate="Sep=%{y:.1f}¢<extra></extra>"), row=1, col=1, secondary_y=True)
yoy_colors = [GREEN if (pd.notna(v) and v >= 0) else RED for v in df["yield_yoy_pct"]]
fig_yp.add_trace(go.Bar(x=df.index, y=df["yield_yoy_pct"], name="Yield YoY %",
    marker_color=yoy_colors, opacity=0.85,
    hovertemplate="%{y:+.1f}%<extra></extra>"), row=2, col=1)
fig_yp.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=2, col=1)
fig_yp.update_yaxes(title_text="Production (M boxes)", secondary_y=False, row=1, col=1,
    tickfont=dict(color=ORANGE), gridcolor="rgba(255,255,255,0.05)")
fig_yp.update_yaxes(title_text="OJ Price (¢/lb)", secondary_y=True, row=1, col=1,
    tickfont=dict(color=BLUE), showgrid=False)
fig_yp.update_layout(**PLOTLY_LAYOUT, height=540)
fig_yp.update_yaxes(gridcolor="rgba(255,255,255,0.05)", row=2, col=1)
fig_yp.update_xaxes(minallowed=_X_MIN, maxallowed=_X_MAX)

# Tab 4 — Backtest
bt_sorted = bt0.sort_index().copy()
bt_sorted["cum_pnl"] = bt_sorted["trade_ret"].cumsum()
pnl_colors = [GREEN if v >= 0 else RED for v in bt_sorted["cum_pnl"]]
n_ok = int(bt0["correct"].sum()); n_tot = len(bt0)
hit = n_ok / n_tot if n_tot else 0
cum_pnl = bt0["trade_ret"].sum()
avg_win = bt0.loc[bt0["correct"]==True, "trade_ret"].mean() if n_ok > 0 else 0
avg_los = bt0.loc[bt0["correct"]==False, "trade_ret"].mean() if (n_tot-n_ok) > 0 else 0

fig_bt = make_subplots(rows=1, cols=2,
    subplot_titles=("Cumulative P&L (entry price %)", "Signal Distribution"),
    column_widths=[0.65, 0.35], specs=[[{"type":"xy"},{"type":"domain"}]])
fig_bt.add_trace(go.Scatter(x=bt_sorted.index, y=bt_sorted["cum_pnl"],
    line=dict(color=BLUE, width=2.5), fill="tozeroy", fillcolor="rgba(96,165,250,0.12)",
    mode="lines+markers",
    marker=dict(size=7, color=pnl_colors, line=dict(width=1.5, color="rgba(255,255,255,0.6)")),
    name="Cum P&L", hovertemplate="%{y:+.1f}%<extra></extra>"), row=1, col=1)
fig_bt.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=1, col=1)
counts = [n_ok, n_tot-n_ok, len(df)-n_tot]
fig_bt.add_trace(go.Pie(values=counts, labels=["Correct","Wrong","No signal"],
    marker_colors=[GREEN, RED, GRAY], hole=0.55, textfont_size=12,
    hovertemplate="%{label}: %{value}<extra></extra>"), row=1, col=2)
fig_bt.update_layout(**PLOTLY_LAYOUT, height=340)
fig_bt.update_yaxes(gridcolor="rgba(255,255,255,0.05)", row=1, col=1)
fig_bt.update_xaxes(minallowed=_X_MIN, maxallowed=_X_MAX, row=1, col=1)

# Confirmation filter tiers
has_brazil = "brazil_yoy_pct" in df.columns and df["brazil_yoy_pct"].notna().any()
has_rainfall = "fl_rainfall_below_avg" in df.columns and df["fl_rainfall_below_avg"].notna().any()
if has_brazil:
    brazil_mask = (
        ((df["signal"]=="LONG") & (df["brazil_yoy_pct"]<0)) |
        ((df["signal"]=="SHORT") & (df["brazil_yoy_pct"]>0))
    )
    bt2 = df[brazil_mask & df["correct"].notna()].copy()
    n_ok2=int(bt2["correct"].sum()); n_tot2=len(bt2)
    hit2=n_ok2/n_tot2 if n_tot2 else 0; pnl2=bt2["trade_ret"].sum()
    if has_rainfall:
        rain_mask = (
            ((df["signal"]=="LONG") & (df["brazil_yoy_pct"]<0) & (df["fl_rainfall_below_avg"]==1)) |
            ((df["signal"]=="SHORT") & (df["brazil_yoy_pct"]>0))
        )
        bt3=df[rain_mask & df["correct"].notna()].copy()
        n_ok3=int(bt3["correct"].sum()); n_tot3=len(bt3)
        hit3=n_ok3/n_tot3 if n_tot3 else 0; pnl3=bt3["trade_ret"].sum()

# Tab 5 — Signal
sig = latest["signal"]; surprise = latest["ndvi_surprise"]
apr = latest["apr_close"]; sep = latest["sep_close"]
correct = latest["correct"]; ndvi = latest["ndvi_jan_mar"]; avg3 = latest["ndvi_3yr_avg"]
sig_color = GREEN if sig=="LONG" else (RED if sig=="SHORT" else GRAY)
sig_bg = "rgba(34,197,94,.12)" if sig=="LONG" else ("rgba(239,68,68,.12)" if sig=="SHORT" else "rgba(148,163,184,.12)")
outcome_color = GREEN if correct is True else (RED if correct is False else GRAY)
outcome_txt = "CORRECT ✓" if correct is True else ("INCORRECT ✗" if correct is False else "PENDING —")

valid = df[df["ndvi_surprise"].notna()]
gauge_min = float(valid["ndvi_surprise"].min())*1.2
gauge_max = float(valid["ndvi_surprise"].max())*1.2
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta", value=float(surprise),
    delta={"reference":0,"valueformat":".4f"},
    number={"valueformat":".4f","font":{"size":36}},
    gauge={
        "axis":{"range":[gauge_min,gauge_max],"tickformat":".3f","tickfont":{"size":11}},
        "bar":{"color":sig_color,"thickness":0.25},
        "bgcolor":"rgba(30,33,48,0.5)", "bordercolor":"rgba(255,255,255,0.1)",
        "steps":[
            {"range":[gauge_min,0],"color":"rgba(239,68,68,0.15)"},
            {"range":[0,gauge_max],"color":"rgba(34,197,94,0.15)"},
        ],
        "threshold":{"line":{"color":GOLD,"width":2},"thickness":0.8,"value":0},
    },
    title={"text":f"NDVI Surprise · {latest_year}<br><span style='font-size:12px;color:#94A3B8'>Red zone = LONG signal &nbsp;|&nbsp; Green zone = SHORT signal</span>"},
))
fig_gauge.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k!="margin"}, height=300, margin=dict(l=40,r=40,t=60,b=20))

# ── Backtest table ─────────────────────────────────────────────────────────────
display_df = df[df["signal"].notna() & (df["signal"]!="NEUTRAL")].copy()
if has_brazil:
    display_df["Brazil"] = display_df.apply(
        lambda r: "✓" if (
            (r["signal"]=="LONG" and pd.notna(r["brazil_yoy_pct"]) and r["brazil_yoy_pct"]<0) or
            (r["signal"]=="SHORT" and pd.notna(r["brazil_yoy_pct"]) and r["brazil_yoy_pct"]>0)
        ) else ("—" if pd.isna(r["brazil_yoy_pct"]) else "✗"), axis=1)
if has_rainfall:
    display_df["Dry Jan-Mar"] = display_df.apply(
        lambda r: (
            "✓" if r["signal"]=="LONG" and r["fl_rainfall_below_avg"]==1
            else ("✗" if r["signal"]=="LONG" and r["fl_rainfall_below_avg"]==0 else "—")
        ) if pd.notna(r.get("fl_rainfall_below_avg")) else "—", axis=1)
cols_show = ["ndvi_surprise","signal"]
if has_brazil: cols_show.append("Brazil")
if has_rainfall: cols_show.append("Dry Jan-Mar")
cols_show += ["apr_close","sep_close","price_direction","correct","trade_ret"]
display_df = display_df[cols_show].rename(columns={
    "ndvi_surprise":"NDVI Surprise","signal":"Signal",
    "apr_close":"Apr Close (¢)","sep_close":"Sep Close (¢)",
    "price_direction":"Price Dir","correct":"Correct","trade_ret":"Return (%)"
})
display_df["Correct"] = display_df["Correct"].map({True:"✓",False:"✗"})
display_df["Price Dir"] = display_df["Price Dir"].map({1.0:"↑",-1.0:"↓"})
display_df["NDVI Surprise"] = display_df["NDVI Surprise"].map("{:+.4f}".format)
display_df["Return (%)"] = display_df["Return (%)"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")

def df_to_html(d):
    rows = ""
    for yr, row in d.iterrows():
        cells = f"<td>{yr}</td>" + "".join(f"<td>{v}</td>" for v in row)
        rows += f"<tr>{cells}</tr>"
    headers = "<th>Year</th>" + "".join(f"<th>{c}</th>" for c in d.columns)
    return f'<table class="bt-table"><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>'

# Signal history
hist5 = df[df["signal"].notna() & (df["signal"]!="NEUTRAL") & df["correct"].notna()].tail(8).copy()
sig_history_html = ""
for yr, row in hist5.iterrows():
    s=row["signal"]; c=row["correct"]
    sc=GREEN if s=="LONG" else RED; cc=GREEN if c is True else RED
    cm="✓" if c is True else "✗"
    move=(row["sep_close"]-row["apr_close"])/row["apr_close"]*100
    sig_history_html += f'''
    <div style="display:flex;align-items:center;background:#1a1d2e;border-radius:10px;padding:12px 16px;margin-bottom:8px;border-left:3px solid {sc}">
      <div style="font-weight:700;font-size:16px;width:48px">{yr}</div>
      <div style="background:{sc}22;color:{sc};font-weight:700;padding:3px 10px;border-radius:999px;font-size:12px;width:60px;text-align:center">{s}</div>
      <div style="flex:1;text-align:center;color:#94A3B8;font-size:13px">surprise: <b style="color:{sc}">{row["ndvi_surprise"]:+.4f}</b></div>
      <div style="text-align:center;color:#94A3B8;font-size:13px;width:80px">{row["apr_close"]:.0f}¢→{row["sep_close"]:.0f}¢<br><span style="font-size:11px;color:{"#22C55E" if move>=0 else "#EF4444"}">{move:+.1f}%</span></div>
      <div style="font-size:20px;color:{cc};width:32px;text-align:right">{cm}</div>
    </div>'''

fwd = df[df["correct"].isna() & df["signal"].isin(["LONG","SHORT"])]
fwd_html = ""
if not fwd.empty:
    fwd_yr=int(fwd.index[-1]); fwd_row=fwd.iloc[-1]
    fwd_sig=fwd_row["signal"]; fwd_sc=GREEN if fwd_sig=="LONG" else RED
    fwd_apr=fwd_row["apr_close"]; fwd_surp=fwd_row["ndvi_surprise"]
    apr_txt=f"{fwd_apr:.1f}¢ entry" if pd.notna(fwd_apr) else "entry TBD"
    dir_txt="above Apr close → ✓" if fwd_sig=="LONG" else "below Apr close → ✓"
    fwd_html = f'''
    <div style="background:{fwd_sc}11;border:1px solid {fwd_sc}44;border-radius:12px;padding:18px;margin-top:16px">
      <div style="color:{GOLD};font-weight:700;font-size:13px;margin-bottom:10px">⚡ {fwd_yr} Forward Signal</div>
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
        <div style="background:{fwd_sc}22;color:{fwd_sc};font-weight:800;padding:6px 18px;border-radius:999px;font-size:18px">{fwd_sig}</div>
        <div style="color:#94A3B8;font-size:13px">NDVI surprise <b style="color:{fwd_sc}">{fwd_surp:+.4f}</b></div>
      </div>
      <div style="color:#94A3B8;font-size:12px;line-height:1.7">
        Entry: <strong style="color:#E2E8F0">{apr_txt}</strong><br>
        Prediction: Sep {fwd_yr} OJ {dir_txt}<br>
        Outcome: <strong style="color:{GOLD}">Pending — resolves Sep {fwd_yr}</strong>
      </div>
    </div>'''

# ── Card helpers ───────────────────────────────────────────────────────────────
prod_raw = df.loc[latest_year, "production_boxes"]
prod_25 = prod_raw/1e6 if pd.notna(prod_raw) else None
prod_05 = df.loc[2005,"production_boxes"]/1e6
acres_25 = df.loc[latest_year,"bearing_acres"]
acres_05 = df.loc[2005,"bearing_acres"]
apr_25 = df.loc[latest_year,"apr_close"]
apr_05 = df.loc[2005,"apr_close"]

def card(label, value, delta, val_color, delta_color):
    return f'''<div class="card">
      <div class="card-label">{label}</div>
      <div class="card-value" style="color:{val_color}">{value}</div>
      <div class="card-delta" style="color:{delta_color}">{delta}</div>
    </div>'''

card1 = card(f"{latest_year} Production {tip('yield_surprise')}",
    f"{prod_25:.1f}M boxes" if prod_25 else "Pending",
    f"{(prod_25/prod_05-1)*100:.0f}% vs 2005 peak" if prod_25 else "NASS not yet released",
    ORANGE, RED)
card2 = card(f"Bearing Acres {tip('bearing_acres')}",
    f"{acres_25/1e3:.0f}K" if pd.notna(acres_25) else "Pending",
    f"{(acres_25/acres_05-1)*100:.0f}% vs 2005" if pd.notna(acres_25) else "NASS not yet released",
    BLUE, RED)
card3 = card(f"Apr {latest_year} OJ Price {tip('fcoj')}",
    f"{apr_25:.0f}¢/lb",
    f"+{(apr_25/apr_05-1)*100:.0f}% vs 2005",
    GOLD, GREEN)
card4 = card(f"Signal Hit Rate {tip('hit_rate')}",
    f"{hit0:.0%}",
    f"{int(bt0['correct'].sum())}/{len(bt0)} years · threshold=0",
    PURPLE, GRAY)

# Tier cards
def tier_card(title, subtitle, hit_val, ok, tot, pnl_val, tier_color):
    dv = hit_val - hit
    dc = GREEN if dv>=0 else RED
    return f'''<div class="card" style="border-top:3px solid {tier_color};padding-top:14px">
      <div class="card-label" style="color:{tier_color}">{title}</div>
      <div style="font-size:11px;color:#64748b;margin-bottom:8px">{subtitle}</div>
      <div style="font-size:2rem;font-weight:800;color:{GREEN if hit_val>=0.55 else RED};line-height:1;margin-bottom:4px">{hit_val:.0%}</div>
      <div style="font-size:11px;color:#64748b">{ok}/{tot} years correct <span style="color:{dc};margin-left:6px">({dv:+.0%} vs baseline)</span></div>
      <div style="font-size:12px;color:{GREEN if pnl_val>=0 else RED};margin-top:6px">Cum P&L: {pnl_val:+.1f}%</div>
    </div>'''

# Glossary terms
TERMS = [
    ("NDVI","ndvi","Satellite signal",BLUE),
    ("NDVI Surprise","ndvi_surprise","Core signal",ORANGE),
    ("NDVI × Acres","ndvi_x_acres","Composite signal",PURPLE),
    ("Bearing Acres","bearing_acres","Supply metric",GREEN),
    ("FCOJ Futures","fcoj","Market",GOLD),
    ("Apr / Sep Close","apr_sep","Price reference",BLUE),
    ("Hit Rate","hit_rate","Backtest metric",GREEN),
    ("Cumulative P&L","cum_pnl","Backtest metric",GREEN),
    ("Yield Surprise","yield_surprise","Supply metric",ORANGE),
    ("HLB Disease","hlb","Context",RED),
    ("LONG Signal","long_signal","Trade signal",GREEN),
    ("SHORT Signal","short_signal","Trade signal",RED),
]
glossary_html = ""
for i in range(0, len(TERMS), 2):
    pair = TERMS[i:i+2]
    glossary_html += '<div class="g2col">'
    for name, key, cat, color in pair:
        glossary_html += f'''<div class="gterm" style="border-left-color:{color}">
          <div><span class="gterm-name">{name}</span>
          <span class="gterm-cat" style="background:{color}22;color:{color}">{cat}</span></div>
          <div class="gterm-def">{GLOSSARY[key]}</div>
        </div>'''
    glossary_html += '</div>'

is_forward = correct is None and sig in ("LONG","SHORT")
fwd_banner = ""
if is_forward:
    fwd_banner = f'''<div style="background:rgba(251,191,36,.10);border:1px solid rgba(251,191,36,.4);border-radius:10px;padding:10px 16px;margin-bottom:12px;display:flex;align-items:center;gap:10px">
      <span style="font-size:16px">⚡</span>
      <div><span style="color:{GOLD};font-weight:700;font-size:13px">FORWARD PREDICTION — OUTCOME PENDING</span><br>
      <span style="color:#94A3B8;font-size:12px">Apr→Sep {latest_year} OJ direction not yet known · resolves September {latest_year}</span></div>
    </div>'''

actual_move = ((sep-apr)/apr*100) if pd.notna(apr) and pd.notna(sep) else None
sep_txt = f"{sep:.1f}¢" if pd.notna(sep) else "Pending"

tier_section = ""
if has_brazil:
    t1 = tier_card("Tier 1 — NDVI Signal","All signal years, no extra filters",hit,n_ok,n_tot,cum_pnl,GRAY)
    t2 = tier_card("Tier 2 — + Brazil","LONG: Brazil ↓ YoY · SHORT: Brazil ↑ YoY",hit2,n_ok2,n_tot2,pnl2,GOLD)
    t3 = tier_card("Tier 3 — + Brazil + Dry Jan-Mar","Tier 2 + LONG requires FL rainfall < avg",hit3,n_ok3,n_tot3,pnl3,BLUE) if has_rainfall else ""
    tier_section = f'''<hr style="border-color:rgba(255,255,255,0.07);margin:28px 0">
    <div style="font-size:15px;font-weight:700;margin-bottom:4px">Signal Confirmation Filters</div>
    <div style="font-size:12px;color:#64748b;margin-bottom:18px;line-height:1.6">Each tier adds a macro confirmation layer on top of the NDVI signal.</div>
    <div class="three-col">{t1}{t2}{t3}</div>'''

# ── Assemble HTML ──────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Starvest · OJ Signal</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0f111e;color:#E2E8F0;font-family:'Inter',sans-serif;min-height:100vh}}
.container{{max-width:1400px;margin:0 auto;padding:24px 20px}}
hr{{border:none;border-top:1px solid rgba(255,255,255,0.08);margin:12px 0 18px}}
.card{{background:#1a1d2e;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:22px 18px;text-align:center;overflow:visible}}
.card-label{{color:#64748b;font-size:11px;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px}}
.card-value{{font-size:28px;font-weight:800;line-height:1}}
.card-delta{{font-size:12px;margin-top:6px;color:#64748b}}
.four-col{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
.three-col{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:16px}}
.five-col{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:16px}}
.tabs{{display:flex;gap:4px;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:24px;flex-wrap:wrap}}
.tab-btn{{background:none;border:none;color:#64748b;font-size:14px;font-weight:600;padding:10px 20px;cursor:pointer;border-bottom:2px solid transparent;transition:all 0.15s;font-family:'Inter',sans-serif}}
.tab-btn:hover{{color:#E2E8F0}}
.tab-btn.active{{color:#E2E8F0;border-bottom-color:#60A5FA}}
.tab-panel{{display:none}}.tab-panel.active{{display:block}}
.tipwrap{{position:relative;display:inline-block;vertical-align:middle;margin-left:5px}}
.tipicon{{display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;border-radius:50%;background:rgba(96,165,250,0.18);border:1px solid rgba(96,165,250,0.38);color:#60A5FA;font-size:9px;font-weight:700;font-style:italic;cursor:help;line-height:1;flex-shrink:0}}
.tipbox{{display:none;position:absolute;top:calc(100% + 7px);background:#1a1d2e;border:1px solid rgba(255,255,255,0.13);border-radius:10px;padding:11px 14px;width:260px;font-size:12px;line-height:1.65;color:#CBD5E1;box-shadow:0 10px 36px rgba(0,0,0,0.55);z-index:99999;pointer-events:none;white-space:normal;font-weight:400}}
.tipwrap:hover .tipbox{{display:block}}
.gterm{{background:#1a1d2e;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:18px 20px;margin-bottom:12px;border-left-width:3px;border-left-style:solid}}
.gterm-name{{font-size:14px;font-weight:700;color:#E2E8F0;margin-bottom:4px}}
.gterm-cat{{font-size:10px;font-weight:600;letter-spacing:.6px;text-transform:uppercase;padding:2px 8px;border-radius:999px;display:inline-block;margin-left:8px;vertical-align:middle}}
.gterm-def{{font-size:13px;color:#94A3B8;line-height:1.7;margin-top:8px}}
.g2col{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:0}}
.bt-table{{width:100%;border-collapse:collapse;font-size:13px}}
.bt-table th{{color:#64748b;font-weight:600;padding:8px 10px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.1)}}
.bt-table td{{padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.05);color:#E2E8F0}}
.bt-table tr:hover td{{background:rgba(255,255,255,0.03)}}
.table-wrap{{overflow-x:auto;border-radius:10px;max-height:420px;overflow-y:auto}}
.mini-card{{background:#1a1d2e;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:16px;text-align:center}}
@media(max-width:768px){{.four-col,.three-col,.two-col,.g2col{{grid-template-columns:1fr 1fr}}.five-col{{grid-template-columns:1fr 1fr}}}}
@media(max-width:480px){{.four-col,.three-col,.two-col,.g2col,.five-col{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="container">

<!-- Header -->
<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
  <div>
    <h1 style="font-size:2rem;font-weight:800;letter-spacing:-1px">🍊 Starvest</h1>
    <p style="color:#64748b;margin-top:2px;font-size:14px">Florida citrus yield & OJ futures signal · 2005–{latest_year}</p>
  </div>
  <div style="text-align:right;padding-top:12px">
    <div style="color:#64748b;font-size:11px;letter-spacing:1px;text-transform:uppercase">Current Signal</div>
    <div style="font-size:2.2rem;font-weight:800;color:{sig_color};line-height:1.1">{sig}</div>
    <div style="color:#64748b;font-size:12px">{latest_year} NDVI basis</div>
  </div>
</div>
<hr>

<!-- Tabs -->
<div class="tabs">
  <button class="tab-btn active" onclick="showTab('overview')">Overview</button>
  <button class="tab-btn" onclick="showTab('ndvi')">NDVI Trend</button>
  <button class="tab-btn" onclick="showTab('yield')">Yield vs Price</button>
  <button class="tab-btn" onclick="showTab('backtest')">Backtest</button>
  <button class="tab-btn" onclick="showTab('signal')">Signal</button>
  <button class="tab-btn" onclick="showTab('glossary')">Glossary</button>
</div>

<!-- Tab 1: Overview -->
<div id="tab-overview" class="tab-panel active">
  <div class="four-col">
    {card1}{card2}{card3}{card4}
  </div>
  {fig_html(fig_overview)}
  <div style="margin-top:20px">
    <p style="font-weight:700;margin-bottom:6px">How Starvest works:</p>
    <p style="color:#94A3B8;font-size:13px;margin-bottom:10px">Three signals feed a single Jan–Mar directional call on FCOJ futures {tip('fcoj')} (April close → September close {tip('apr_sep')}):</p>
    <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:14px">
      <thead><tr style="border-bottom:1px solid rgba(255,255,255,0.1)">
        <th style="text-align:left;padding:8px 10px;color:#64748b;font-weight:600">Signal</th>
        <th style="text-align:left;padding:8px 10px;color:#64748b;font-weight:600">Source</th>
        <th style="text-align:left;padding:8px 10px;color:#64748b;font-weight:600">Logic</th>
      </tr></thead><tbody>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
        <td style="padding:8px 10px;color:#E2E8F0;font-weight:600">NDVI Jan-Mar {tip('ndvi')}</td>
        <td style="padding:8px 10px;color:#94A3B8">MODIS MOD13Q1 250m via GEE</td>
        <td style="padding:8px 10px;color:#94A3B8">Below 3-yr avg → stress → lower yield → price up</td>
      </tr>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.05)">
        <td style="padding:8px 10px;color:#E2E8F0;font-weight:600">NDVI Surprise {tip('ndvi_surprise')}</td>
        <td style="padding:8px 10px;color:#94A3B8">NDVI vs rolling 3-yr baseline</td>
        <td style="padding:8px 10px;color:#94A3B8">Magnitude of deviation → confidence in call</td>
      </tr>
      <tr>
        <td style="padding:8px 10px;color:#E2E8F0;font-weight:600">NDVI × Acres {tip('ndvi_x_acres')}</td>
        <td style="padding:8px 10px;color:#94A3B8">NDVI weighted by grove footprint</td>
        <td style="padding:8px 10px;color:#94A3B8">Structural shrinkage (HLB {tip('hlb')}) ↓ 69% since 2005</td>
      </tr>
      </tbody>
    </table>
    <div style="background:rgba(148,163,184,0.07);border-left:3px solid rgba(148,163,184,0.3);border-radius:0 8px 8px 0;padding:10px 14px;font-size:12px;color:#94A3B8;line-height:1.6">
      <b style="color:#CBD5E1">Note:</b> NDVI measures vegetation greenness, not fruit count. HLB-infected trees can appear green while producing near-zero fruit — a fundamental limitation the NDVI × Acres composite partially corrects for.
    </div>
  </div>
</div>

<!-- Tab 2: NDVI -->
<div id="tab-ndvi" class="tab-panel">
  {fig_html(fig_ndvi)}
  <div class="two-col" style="margin-top:16px">
    <div>
      <p style="font-weight:700;margin-bottom:6px">NDVI {tip('ndvi')} interpretation:</p>
      <ul style="color:#94A3B8;font-size:13px;line-height:2;margin:0;padding-left:18px">
        <li>Range 0→1; healthy dense citrus groves ≈ 0.50–0.70</li>
        <li>Jan–Mar captures the citrus belt <em>before</em> summer rainy season</li>
        <li>Surprise {tip('ndvi_surprise')} &lt; 0 (red bar) = browner than typical → stress signal</li>
        <li>Surprise &gt; 0 (green bar) = greener than typical → healthier canopy</li>
      </ul>
    </div>
    <div>
      <p style="font-weight:700;margin-bottom:6px">NDVI × Acres {tip('ndvi_x_acres')} composite:</p>
      <ul style="color:#94A3B8;font-size:13px;line-height:2;margin:0;padding-left:18px">
        <li>NDVI weighted by grove size relative to 2005 baseline</li>
        <li>0.54 → 0.17 over 20 years = fewer trees <em>and</em> stressed trees</li>
        <li>Better reflects OJ supply capacity than NDVI alone</li>
        <li>HLB {tip('hlb')} infected trees look green but yield near-zero fruit</li>
      </ul>
    </div>
  </div>
</div>

<!-- Tab 3: Yield vs Price -->
<div id="tab-yield" class="tab-panel">
  {fig_html(fig_yp)}
  <div style="margin-top:16px;font-size:13px;color:#94A3B8">
    <p style="font-weight:700;color:#E2E8F0;margin-bottom:8px">Key supply events in the data:</p>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <thead><tr style="border-bottom:1px solid rgba(255,255,255,0.1)">
        <th style="padding:8px 10px;color:#64748b;font-weight:600;text-align:left">Year</th>
        <th style="padding:8px 10px;color:#64748b;font-weight:600;text-align:left">Event</th>
        <th style="padding:8px 10px;color:#64748b;font-weight:600;text-align:left">Production</th>
        <th style="padding:8px 10px;color:#64748b;font-weight:600;text-align:left">OJ Price Direction</th>
      </tr></thead><tbody>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:8px 10px">2007</td><td style="padding:8px 10px">Hard freeze Jan 2007</td><td style="padding:8px 10px">129M → supply shock</td><td style="padding:8px 10px">↑ to 172¢ Apr</td></tr>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:8px 10px">2018</td><td style="padding:8px 10px">Hurricane Irma (Sep 2017)</td><td style="padding:8px 10px">45M boxes — 35% crash</td><td style="padding:8px 10px">Price ultimately ↑ Apr→Sep</td></tr>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:8px 10px">2023</td><td style="padding:8px 10px">Hurricane Ian + HLB</td><td style="padding:8px 10px"><b>15.8M</b> — record low</td><td style="padding:8px 10px">↑ 276¢ → 341¢ (+24%)</td></tr>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:8px 10px">2024</td><td style="padding:8px 10px">Supply shock ongoing</td><td style="padding:8px 10px">18M</td><td style="padding:8px 10px">↑ 369¢ → 497¢ (+35%)</td></tr>
      <tr style="border-bottom:1px solid rgba(255,255,255,0.05)"><td style="padding:8px 10px">2025</td><td style="padding:8px 10px">Continued HLB collapse</td><td style="padding:8px 10px">12.3M</td><td style="padding:8px 10px">↓ 267¢ → 241¢</td></tr>
      <tr><td style="padding:8px 10px">2026</td><td style="padding:8px 10px">Hard freeze + drought</td><td style="padding:8px 10px">12.2M prelim</td><td style="padding:8px 10px">Apr 179¢ · Sep pending → <b>LONG signal</b></td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- Tab 4: Backtest -->
<div id="tab-backtest" class="tab-panel">
  <p style="font-size:13px;color:#64748b;margin-bottom:16px"><b style="color:#E2E8F0">Signal:</b> LONG when NDVI surprise &lt; 0 · SHORT when NDVI surprise &gt; 0 (threshold = 0)</p>
  <div class="five-col">
    <div class="mini-card"><div class="card-label">Hit Rate {tip('hit_rate')}</div><div class="card-value" style="font-size:22px;color:{GREEN if hit>=0.55 else RED}">{hit:.0%}</div></div>
    <div class="mini-card"><div class="card-label">Years Tested</div><div class="card-value" style="font-size:22px;color:{GRAY}">{n_ok}/{n_tot}</div></div>
    <div class="mini-card"><div class="card-label">Cum P&L {tip('cum_pnl')}</div><div class="card-value" style="font-size:22px;color:{GREEN if cum_pnl>=0 else RED}">{cum_pnl:+.1f}%</div><div style="font-size:10px;color:#64748b;margin-top:5px">total over {n_tot} years</div></div>
    <div class="mini-card"><div class="card-label">Avg Win</div><div class="card-value" style="font-size:22px;color:{GREEN}">{avg_win:+.1f}%</div></div>
    <div class="mini-card"><div class="card-label">Avg Loss</div><div class="card-value" style="font-size:22px;color:{RED}">{avg_los:+.1f}%</div></div>
  </div>
  <div style="color:#64748b;font-size:12px;margin-bottom:16px">{hit:.0%} directional accuracy &nbsp;·&nbsp; 52–54% is typical for professional quant commodity signals</div>
  {fig_html(fig_bt)}
  {tier_section}
  <div style="margin-top:24px">
    <p style="font-weight:700;margin-bottom:8px">Year-by-year breakdown</p>
    <div class="table-wrap">{df_to_html(display_df)}</div>
  </div>
</div>

<!-- Tab 5: Signal -->
<div id="tab-signal" class="tab-panel">
  <div class="two-col">
    <div>
      {fwd_banner}
      <div style="background:{sig_bg};border:1px solid {sig_color}33;border-radius:16px;padding:32px;text-align:center;margin-bottom:16px">
        <div style="color:{sig_color};font-size:11px;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">{latest_year} Signal — Jan-Mar NDVI Basis</div>
        <div style="font-size:4rem;font-weight:800;color:{sig_color};line-height:1">{sig} {tip('long_signal' if sig=='LONG' else ('short_signal' if sig=='SHORT' else 'ndvi_surprise'))}</div>
        <div style="font-size:14px;color:#94A3B8;margin-top:12px">NDVI surprise {tip('ndvi_surprise','center')}: <span style="color:{sig_color};font-weight:700">{surprise:+.4f}</span> &nbsp;|&nbsp; threshold: 0.000</div>
      </div>
      <div class="card" style="text-align:left;margin-bottom:12px">
        <div class="card-label">NDVI {tip('ndvi')} DETAIL · {latest_year}</div>
        <table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:14px">
          <tr><td style="color:#64748b;padding:4px 0">Jan-Mar NDVI</td><td style="text-align:right;font-weight:600">{ndvi:.4f}</td></tr>
          <tr><td style="color:#64748b;padding:4px 0">3-yr baseline</td><td style="text-align:right;font-weight:600">{avg3:.4f}</td></tr>
          <tr><td style="color:#64748b;padding:4px 0">Surprise {tip('ndvi_surprise','right')}</td><td style="text-align:right;font-weight:700;color:{sig_color}">{surprise:+.4f}</td></tr>
          <tr><td style="color:#64748b;padding:4px 0">Signal</td><td style="text-align:right;font-weight:700;color:{sig_color}">{sig}</td></tr>
        </table>
      </div>
      <div class="card" style="text-align:left">
        <div class="card-label">OUTCOME · {latest_year}</div>
        <table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:14px">
          <tr><td style="color:#64748b;padding:4px 0">Apr close</td><td style="text-align:right;font-weight:600">{apr:.1f}¢</td></tr>
          <tr><td style="color:#64748b;padding:4px 0">Sep close</td><td style="text-align:right;font-weight:600">{sep_txt}</td></tr>
          <tr><td style="color:#64748b;padding:4px 0">Actual move</td><td style="text-align:right;font-weight:600">{f"{actual_move:+.1f}%" if actual_move is not None else "N/A"}</td></tr>
          <tr><td style="color:#64748b;padding:4px 0">Signal outcome</td><td style="text-align:right;font-weight:700;color:{outcome_color}">{outcome_txt}</td></tr>
        </table>
      </div>
    </div>
    <div>
      <p style="font-weight:700;margin-bottom:12px">Signal history — last 8 completed years</p>
      {sig_history_html}
      {fwd_html}
    </div>
  </div>
  <div style="margin-top:24px">
    <p style="font-weight:700;margin-bottom:8px">NDVI Surprise — signal strength gauge</p>
    {fig_html(fig_gauge, config=PLOTLY_CONFIG_STATIC)}
  </div>
</div>

<!-- Tab 6: Glossary -->
<div id="tab-glossary" class="tab-panel">
  <h2 style="font-size:1.35rem;font-weight:800;margin-bottom:2px">Glossary</h2>
  <p style="color:#64748b;font-size:13px;margin-bottom:22px">Definitions for every signal, metric, and term used in Starvest.</p>
  {glossary_html}
</div>

</div><!-- /container -->

<script>
function showTab(name) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
  window.dispatchEvent(new Event('resize'));
}}
</script>
</body>
</html>"""

# ── Write output ───────────────────────────────────────────────────────────────
out = ROOT / "docs" / "index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html, encoding="utf-8")
print(f"Built → {out}  ({out.stat().st_size // 1024} KB)")
