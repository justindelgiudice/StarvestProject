# Starvest

NASA/Hack Club Stardance Challenge project. Analyzes Florida citrus crop health
using NASA MODIS NDVI satellite data to predict orange harvest yield and infer
OJ futures price pressure.

## Architecture

```
src/fetch_ndvi.py      — Pull MODIS NDVI from Google Earth Engine for the Florida citrus belt
src/fetch_yield.py     — Pull annual Florida orange production from USDA NASS API
src/fetch_prices.py    — Pull OJ futures (OJ=F) history from yfinance
src/build_dataset.py   — Merge all sources; compute growing-season NDVI, yield vs avg, price pressure label
src/model.py           — Linear regression: NDVI + year → yield; saves coefficients to data/processed/model_params.json
src/backtest.py        — Walk-forward backtest; outputs data/processed/backtest_results.csv
api/main.py            — FastAPI backend serving processed data to the React frontend
dashboard/app.py       — Streamlit dashboard (legacy; React frontend via Lovable is the primary UI)
```

## Data Pipeline

Run in order:

```bash
python src/fetch_ndvi.py       # requires GEE auth: earthengine authenticate
python src/fetch_yield.py      # requires NASS_API_KEY in .env
python src/fetch_prices.py
python src/build_dataset.py
python src/model.py
python src/backtest.py
```

## Running the API

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

API runs at http://localhost:8000. Endpoints:
- `GET /api/summary` — latest year KPIs + model stats
- `GET /api/ndvi` — full NDVI time series
- `GET /api/dataset` — yearly dataset (yield, NDVI, price pressure)
- `GET /api/backtest` — walk-forward backtest results
- `GET /api/model-params` — model coefficients
- `GET /api/forecast/{year}` — forecast for a specific year

## React Frontend (Lovable)

Connect the GitHub repo to Lovable. The frontend calls the FastAPI backend at
`http://localhost:8000` (update to deployed URL in production).

## Environment Variables

Create a `.env` file (never commit it):

```
NASS_API_KEY=your_key_here
```

Get a free USDA NASS API key at https://quickstats.nass.usda.gov/api

## Google Earth Engine Auth

```bash
earthengine authenticate
```

Follow the browser prompt. Credentials are stored locally by the GEE SDK.

## Key Concepts

- **NDVI** (Normalized Difference Vegetation Index): values closer to 1.0 = healthier crops
- **Growing season**: October–May for Florida citrus
- **Price pressure**: if predicted yield is >10% below historical average → bullish (supply shock); >10% above → bearish
- **Backtest**: walk-forward — each year trained only on prior years, never future data
