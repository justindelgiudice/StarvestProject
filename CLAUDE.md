# TerraRisk

TerraRisk is a Florida climate risk intelligence platform designed to help insurance underwriters and real estate investors assess coastal hazard exposure.

This project combines historical hurricane tracks, FEMA flood zone data, and NASA sea level rise projections to generate a per-county composite risk score for Florida.

Key components:
- `src/fetch_hurricanes.py`: download NOAA HURDAT2 Atlantic hurricane track data and identify storms that impacted Florida since 1950.
- `src/fetch_flood_zones.py`: ingest FEMA flood zone data by Florida county.
- `src/fetch_sealevel.py`: ingest NASA sea level rise projections for the Florida coastline.
- `src/fetch_tornadoes.py`: download NOAA SPC tornado touchdown data for Florida.
- `src/fetch_sinkholes.py`: ingest Florida Geological Survey sinkhole report data.
- `src/fetch_storm_surge.py`: build county-level storm surge depth data (Cat 1–5) from NOAA NHC SLOSH MEow v4.
- `src/build_risk_score.py`: combine all 7 hazard datasets into a county-level composite risk score.
  - Weights: Hurricane 25% · Storm Surge 20% · Flood 20% · Sea Level 15% · Tornado 10% · Sinkhole 5% · Wildfire 5%
- `src/generate_risk_grid.py`: pre-compute 34K land-point IDW risk grid at 0.02° spacing.
- `src/generate_risk_images.py`: generate 13 smooth RGBA PNG overlays (risk layers + surge Cat 1–5).
- `dashboard/app.py`: Streamlit dashboard with 8 layer toggles (including Storm Surge with Cat 1–5 sub-toggle).

Data directories:
- `data/raw/`: raw source datasets
- `data/processed/`: derived and analytics-ready outputs (county_risk_scores.csv, risk_grid.csv, *.png)

Dependencies are managed in `requirements.txt`.

## Running the pipeline

```bash
python src/fetch_hurricanes.py
python src/fetch_flood_zones.py
python src/fetch_sealevel.py
python src/fetch_tornadoes.py
python src/fetch_sinkholes.py
python src/fetch_storm_surge.py
python src/build_risk_score.py
python src/generate_risk_grid.py
python src/generate_risk_images.py
streamlit run dashboard/app.py
```
