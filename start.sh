#!/usr/bin/env bash
# Run from repo root on Hack Club Nest
set -e
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run dashboard/app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true
