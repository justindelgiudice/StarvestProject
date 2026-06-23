# TerraRisk

TerraRisk is a Florida climate risk intelligence platform designed to help insurance underwriters and real estate investors assess coastal hazard exposure.

This project combines historical hurricane tracks, FEMA flood zone data, and NASA sea level rise projections to generate a per-county composite risk score for Florida.

Key components:
- `src/fetch_hurricanes.py`: download NOAA HURDAT2 Atlantic hurricane track data and identify storms that impacted Florida since 1950.
- `src/fetch_flood_zones.py`: ingest FEMA flood zone data by Florida county.
- `src/fetch_sealevel.py`: ingest NASA sea level rise projections for the Florida coastline.
- `src/build_risk_score.py`: combine hurricane, flood, and sea level datasets into a single county-level risk score.
- `dashboard/app.py`: Streamlit dashboard that visualizes Florida counties colored by composite risk.

Data directories:
- `data/raw/`: raw source datasets
- `data/processed/`: derived and analytics-ready outputs

Dependencies are managed in `requirements.txt`.
