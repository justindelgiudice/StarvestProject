# Starvest

NASA/Hack Club Stardance Challenge project. Uses Sentinel-2 satellite imagery
to detect construction activity across Florida metros and correlates with Zillow
home price changes to predict which markets are heating up or cooling down.

**Financial angle:** Satellite construction signals (NDBI/BSI) predict home price
movement 12–24 months ahead of official market data. Sentinel-2 revisits every
5–16 days vs. closing-based indices with 1–2 month publication lags.

## Architecture

```
src/fetch_satellite.py   — Sentinel-2 NDBI/BSI quarterly composites via GEE
src/fetch_home_prices.py — Zillow ZHVI for FL metros (public CSV download)
src/fetch_permits.py     — Census Building Permits Survey via API
src/build_dataset.py     — Merge sources; compute lag features; create panel
src/model.py             — Ridge regression: construction signals → fwd ZHVI change
src/backtest.py          — Walk-forward backtest; per-metro Heating/Cooling signals
dashboard/app.py         — Streamlit dashboard

archive/                 — Original citrus/OJ pipeline (preserved, not active)
```

## FL Metro Coverage

Miami, Tampa, Orlando, Jacksonville, Fort Lauderdale

## Data Pipeline

Run in order:

```bash
python src/fetch_home_prices.py   # no auth required — Zillow public CSV
python src/fetch_satellite.py     # requires GEE auth: earthengine authenticate
python src/fetch_permits.py       # requires CENSUS_API_KEY in .env
python src/build_dataset.py
python src/model.py
python src/backtest.py
```

## Running the Dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Dashboard shows Zillow ZHVI time series immediately after `fetch_home_prices.py`.
Full model signals appear after the complete pipeline runs.

## Environment Variables

Create a `.env` file (never commit):

```
CENSUS_API_KEY=your_key_here
```

Get a free Census API key at https://api.census.gov/data/key_signup.html

## Google Earth Engine Auth

```bash
earthengine authenticate
```

Follow the browser prompt. Credentials are stored locally by the GEE SDK.

## Key Concepts

- **NDBI** (Normalized Difference Built-up Index): (SWIR − NIR) / (SWIR + NIR).
  Positive values = artificial surfaces. QoQ change = construction ramp-up signal.
- **BSI** (Bare Soil Index): detects graded/cleared land ahead of construction —
  leading indicator earlier than NDBI.
- **ZHVI** (Zillow Home Value Index): smoothed, seasonally adjusted median home value;
  middle tier (33rd–67th percentile); published monthly.
- **Lead time**: permits → construction start ~3–6 months. NDBI detects clearing
  before permits. Full cycle to price impact: 12–24 months.
- **Market signal**: Heating (>+5% fwd ZHVI), Cooling (<−5%), Stable (±5%).
- **Backtest**: walk-forward — each quarter trained only on prior quarters.
