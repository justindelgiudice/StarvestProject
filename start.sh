#!/usr/bin/env bash
# Run from repo root on Hack Club Nest
set -e
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run dashboard/app.py \
  --server.port 8501 \
  --server.address 127.0.0.1 \
  --server.headless true
