---
title: Starvest
emoji: 🍊
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: "1.58.0"
app_file: dashboard/app.py
pinned: false
---

# 🍊 Starvest

**A satellite-driven trading signal for OJ futures — built on NASA MODIS data.**

Starvest uses 20 years of NASA satellite imagery to measure the health of Florida's orange groves every January–March, then issues a directional prediction on Frozen Concentrated Orange Juice (FCOJ) futures before the April expiry. The 2026 signal: **LONG** (hard freeze + drought stress, NDVI surprise −0.0423).

> Built for the [NASA × Hack Club Stardance Challenge](https://www.nasa.gov/stem-content/hack-club-stardance-challenge/) · Summer 2026

---

## How it works

| Signal layer | NASA / public data source | Logic |
|---|---|---|
| **NDVI Jan–Mar** | MODIS MOD13Q1 250 m via Google Earth Engine | Below 3-yr baseline → grove stress → lower yield → OJ price up |
| **NDVI Surprise** | Computed from MODIS | Magnitude of deviation sets signal strength |
| **NDVI × Acres** | MODIS × USDA NASS bearing acreage | Weights greenness by the shrinking grove footprint (−69% since 2005 due to HLB disease) |

Confirmation filters layer on top:
- **Brazil production** (USDA FAS PSD) — global supply tightening check
- **FL Jan–Mar rainfall** (NOAA GHCN-Daily) — drought amplifies NDVI stress

---

## NASA data used

- **MODIS MOD13Q1** (16-day NDVI composite, 250 m) — NASA Terra satellite, accessed via [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1)  
  Region: Florida citrus belt (26.5–28.5°N, 80–82.5°W), Jan 1 – Mar 31 each year, 2002–2026

---

## Dashboard tabs

| Tab | What it shows |
|---|---|
| **Overview** | Key stats (production, acreage, price, hit rate) + 20-year collapse chart |
| **NDVI Trend** | Raw NDVI, 3-yr rolling baseline, surprise bars, and NDVI × Acres composite |
| **Yield vs Price** | Production (M boxes) vs OJ futures price, YoY change |
| **Backtest** | Adjustable threshold slider, cumulative P&L, 3-tier signal comparison |
| **Signal** | Current year signal card, NDVI detail, outcome tracker, signal history |
| **Glossary** | Hover-tooltip definitions for every term |

---

## Run locally

```bash
# 1. Clone
git clone https://github.com/justindelgiudice/Starvest.git
cd Starvest

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch dashboard (uses the pre-built CSV — no API keys needed)
streamlit run dashboard/app.py
```

The pre-built `starvest_data.csv` is included — you can run the dashboard without any API keys.

### Rebuild the data (optional)

To regenerate `starvest_data.csv` from source APIs:

```bash
# Copy the example env file and fill in your keys
cp .env.example .env

# Then run the pipeline
python data_pipeline.py
```

You'll need:
- `GEE_PROJECT` — Google Earth Engine cloud project ID ([sign up free](https://earthengine.google.com/))
- `NASS_API_KEY` — USDA NASS QuickStats API key ([request free](https://quickstats.nass.usda.gov/api))

NOAA GHCN-Daily and USDA FAS data require no API key.

---

## Data sources

| Dataset | Source | Use |
|---|---|---|
| MODIS MOD13Q1 NDVI | NASA / Google Earth Engine | Grove health signal |
| FL orange production + bearing acres | USDA NASS QuickStats API | Yield trend |
| FCOJ futures (OJ=F) | ICE via yfinance | Entry/exit prices |
| FL hard freeze days | NOAA GHCN-Daily (Tampa Intl, Orlando McCoy, Avon Park 2W) | Freeze confirmation |
| FL Jan–Mar rainfall | NOAA GHCN-Daily (Avon Park 2W, Sebring) | Drought confirmation |
| Brazil orange production | USDA FAS PSD bulk download | Global supply check |

---

## Project structure

```
Starvest/
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── data/
│   ├── raw/                # Source datasets (GeoJSON, FEMA NRI, etc.)
│   └── processed/          # Pre-computed outputs
├── data_pipeline.py        # Pulls all 6 datasets → starvest_data.csv
├── starvest_data.csv       # Pre-built dataset (2005–2026)
└── requirements.txt
```

---

## Backtest results (threshold = 0, 2005–2025)

| Metric | Value |
|---|---|
| Signal hit rate | 62% directional accuracy (13/21 years correct) |
| Years with signal | 22 / 22 (all years produced a non-neutral signal) |
| Baseline for comparison | ~52–54% (typical quant commodity signal) |

*Returns are Apr→Sep price moves; excludes futures margin, roll costs, and slippage.*

---

## AI usage

This project was built with assistance from [Claude Code](https://claude.ai/code) (Anthropic). AI was used to help write and debug code, structure the dashboard, and draft documentation. All data, analysis logic, and signal design are my own work.
