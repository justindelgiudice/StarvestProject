#!/usr/bin/env python3
"""
src/generate_risk_grid.py

Pre-compute a dense IDW risk grid covering all of Florida's land area.
Grid: 0.02° spacing (~2km), clipped to Florida county polygons via PIP.

All layers are county-centroid IDW from FEMA NRI scores in county_risk_scores.csv.

Run once (or after fetch_fema_nri.py + build_risk_score.py):
  python src/generate_risk_grid.py

Output: data/processed/risk_grid.csv (~34K rows)
"""

import csv
import json
import os
import time

import numpy as np
import requests

ROOT          = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RAW           = os.path.join(ROOT, "data", "raw")
PROC          = os.path.join(ROOT, "data", "processed")
OUTPUT        = os.path.join(PROC, "risk_grid.csv")
GEOJSON_CACHE = os.path.join(RAW, "fl_counties.geojson")
RISK_CSV      = os.path.join(PROC, "county_risk_scores.csv")
SURGE_CSV     = os.path.join(RAW,  "storm_surge.csv")

FLORIDA_FIPS = {
    "12001","12003","12005","12007","12009","12011","12013","12015",
    "12017","12019","12021","12023","12027","12029","12031","12033",
    "12035","12037","12039","12041","12043","12045","12047","12049",
    "12051","12053","12055","12057","12059","12061","12063","12065",
    "12067","12069","12071","12073","12075","12077","12079","12081",
    "12083","12085","12086","12087","12089","12091","12093","12095",
    "12097","12099","12101","12103","12105","12107","12109","12111",
    "12113","12115","12117","12119","12121","12123","12125","12127",
    "12129","12131","12133",
}

COUNTY_CENTROIDS = {
    "Alachua": (29.67, -82.33), "Baker": (30.33, -82.30), "Bay": (30.22, -85.65),
    "Bradford": (29.94, -82.17), "Brevard": (28.26, -80.72), "Broward": (26.07, -80.25),
    "Calhoun": (30.41, -85.20), "Charlotte": (26.95, -82.03), "Citrus": (28.84, -82.50),
    "Clay": (30.00, -81.87), "Collier": (25.90, -81.30), "Columbia": (30.23, -82.62),
    "DeSoto": (27.18, -81.80), "Dixie": (29.58, -83.17), "Duval": (30.37, -81.65),
    "Escambia": (30.61, -87.34), "Flagler": (29.47, -81.27), "Franklin": (29.84, -84.83),
    "Gadsden": (30.58, -84.62), "Gilchrist": (29.72, -82.79), "Glades": (26.96, -81.19),
    "Gulf": (29.92, -85.18), "Hamilton": (30.49, -82.98), "Hardee": (27.49, -81.79),
    "Hendry": (26.50, -81.31), "Hernando": (28.56, -82.46), "Highlands": (27.34, -81.34),
    "Hillsborough": (27.90, -82.35), "Holmes": (30.87, -85.81), "Indian River": (27.70, -80.57),
    "Jackson": (30.72, -85.20), "Jefferson": (30.42, -83.90), "Lafayette": (29.98, -83.20),
    "Lake": (28.77, -81.71), "Lee": (26.54, -81.76), "Leon": (30.46, -84.29),
    "Levy": (29.28, -82.78), "Liberty": (30.24, -84.88), "Madison": (30.47, -83.47),
    "Manatee": (27.47, -82.35), "Marion": (29.21, -82.06), "Martin": (27.07, -80.41),
    "Miami-Dade": (25.55, -80.63), "Monroe": (24.56, -81.36), "Nassau": (30.61, -81.77),
    "Okaloosa": (30.65, -86.51), "Okeechobee": (27.39, -80.90), "Orange": (28.49, -81.26),
    "Osceola": (27.84, -81.11), "Palm Beach": (26.65, -80.30), "Pasco": (28.30, -82.44),
    "Pinellas": (27.88, -82.73), "Polk": (27.94, -81.68), "Putnam": (29.62, -81.74),
    "St. Johns": (29.95, -81.44), "St. Lucie": (27.38, -80.43), "Santa Rosa": (30.68, -86.98),
    "Sarasota": (27.19, -82.37), "Seminole": (28.71, -81.22), "Sumter": (28.71, -82.08),
    "Suwannee": (30.19, -83.00), "Taylor": (30.06, -83.61), "Union": (30.04, -82.37),
    "Volusia": (29.03, -81.18), "Wakulla": (30.10, -84.37), "Walton": (30.58, -86.13),
    "Washington": (30.60, -85.67),
}

LAT_MIN, LAT_MAX = 24.5, 31.0
LON_MIN, LON_MAX = -87.6, -80.0
STEP = 0.02


# ── GeoJSON / PIP ──────────────────────────────────────────────────────────────

def fetch_fl_geojson():
    if os.path.exists(GEOJSON_CACHE):
        with open(GEOJSON_CACHE) as f:
            return json.load(f)
    print("  Downloading county GeoJSON...")
    r = requests.get(
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        timeout=30,
    )
    r.raise_for_status()
    fl = {
        "type": "FeatureCollection",
        "features": [feat for feat in r.json()["features"] if feat["id"] in FLORIDA_FIPS],
    }
    os.makedirs(RAW, exist_ok=True)
    with open(GEOJSON_CACHE, "w") as f:
        json.dump(fl, f)
    return fl


def prep_polys(geojson):
    out = []
    for feat in geojson["features"]:
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            all_rings = [geom["coordinates"][0]]
        else:
            all_rings = [poly[0] for poly in geom["coordinates"]]
        all_lons = [c[0] for r in all_rings for c in r]
        all_lats = [c[1] for r in all_rings for c in r]
        out.append({
            "bbox": (min(all_lats), max(all_lats), min(all_lons), max(all_lons)),
            "rings": [np.array(r, dtype=np.float32) for r in all_rings],
        })
    return out


def pip_batch(test_lats, test_lons, ring):
    result = np.zeros(len(test_lats), dtype=bool)
    rlon = ring[:, 0]
    rlat = ring[:, 1]
    M = len(ring)
    j = M - 1
    for i in range(M):
        lat_i, lat_j = rlat[i], rlat[j]
        lon_i, lon_j = rlon[i], rlon[j]
        cross = (lat_i > test_lats) != (lat_j > test_lats)
        denom = lat_j - lat_i
        if abs(float(denom)) < 1e-12:
            denom = np.float32(1e-12)
        x_int = lon_i + (lon_j - lon_i) * (test_lats - lat_i) / denom
        result ^= cross & (test_lons < x_int)
        j = i
    return result


def build_land_mask(all_lats, all_lons, polys):
    land = np.zeros(len(all_lats), dtype=bool)
    for poly in polys:
        mn_lat, mx_lat, mn_lon, mx_lon = poly["bbox"]
        bbox = (
            (all_lats >= mn_lat) & (all_lats <= mx_lat) &
            (all_lons >= mn_lon) & (all_lons <= mx_lon)
        )
        if not np.any(bbox):
            continue
        cand_lats = all_lats[bbox]
        cand_lons = all_lons[bbox]
        hit = np.zeros(int(bbox.sum()), dtype=bool)
        for ring in poly["rings"]:
            hit |= pip_batch(cand_lats, cand_lons, ring)
        land[bbox] |= hit
    return land


# ── IDW ────────────────────────────────────────────────────────────────────────

def idw_chunked(glat, glon, slat, slon, sval, power=2, chunk=800):
    slat = np.asarray(slat, np.float32)
    slon = np.asarray(slon, np.float32)
    sval = np.asarray(sval, np.float32)
    N = len(glat)
    out = np.zeros(N, np.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        dl = (glat[s:e, None] - slat) ** 2
        do = (glon[s:e, None] - slon) ** 2
        d2 = np.maximum(dl + do, 1e-9)
        w = 1.0 / (d2 ** power)
        out[s:e] = (w * sval).sum(1) / w.sum(1)
    return out


# ── Score columns to generate as grid layers ───────────────────────────────────

GRID_LAYERS = [
    "eal_score",           # EAL (primary — pure physical hazard)
    "risk_score",          # Composite risk (includes social factors)
    "hurricane_score",     # HRCN_EALS
    "coastal_flood_score", # CFLD_EALS
    "inland_flood_score",  # IFLD_EALS
    "tornado_score",       # TRND_EALS
    "wildfire_score",      # WFIR_EALS
    "wind_score",          # SWND_EALS
    "surge",               # Cat 4 surge depth (from storm_surge.csv)
]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PROC, exist_ok=True)
    t0 = time.time()

    print("Loading Florida county polygons...")
    gj = fetch_fl_geojson()
    polys = prep_polys(gj)
    print(f"  {len(polys)} county polygons ({time.time()-t0:.1f}s)")

    lats_arr = np.arange(LAT_MIN, LAT_MAX + STEP / 2, STEP, dtype=np.float32)
    lons_arr = np.arange(LON_MIN, LON_MAX + STEP / 2, STEP, dtype=np.float32)
    mesh_lats, mesh_lons = np.meshgrid(lats_arr, lons_arr, indexing="ij")
    all_lats = mesh_lats.ravel()
    all_lons = mesh_lons.ravel()
    print(f"Testing {len(all_lats):,} candidates ({len(lats_arr)}×{len(lons_arr)})...")

    land = build_land_mask(all_lats, all_lons, polys)
    glat = all_lats[land]
    glon = all_lons[land]
    print(f"  {len(glat):,} land points retained ({time.time()-t0:.1f}s)")

    # ── Load NRI county scores ─────────────────────────────────────────────────
    print("Loading NRI county scores...")
    nri_scores: dict[str, dict] = {}
    with open(RISK_CSV) as f:
        for row in csv.DictReader(f):
            nri_scores[row["county"]] = row

    surge_scores: dict[str, int] = {}
    if os.path.exists(SURGE_CSV):
        with open(SURGE_CSV) as f:
            for row in csv.DictReader(f):
                surge_scores[row["county"]] = int(row.get("cat4_ft") or 0)

    # Build centroid arrays used for all IDW layers
    counties = [c for c in COUNTY_CENTROIDS if c in nri_scores]
    c_lats = np.array([COUNTY_CENTROIDS[c][0] for c in counties], dtype=np.float32)
    c_lons = np.array([COUNTY_CENTROIDS[c][1] for c in counties], dtype=np.float32)

    # ── Compute per-layer IDW ──────────────────────────────────────────────────
    layer_arrays = {}
    for layer in GRID_LAYERS:
        t1 = time.time()
        if layer == "surge":
            vals = np.array([float(surge_scores.get(c, 0)) for c in counties], dtype=np.float32)
            # Only use counties with actual surge exposure as IDW source points
            mask = vals > 0
            if mask.sum() > 0:
                arr = idw_chunked(glat, glon, c_lats[mask], c_lons[mask], vals[mask], power=2)
            else:
                arr = np.zeros(len(glat), dtype=np.float32)
        else:
            vals = np.array([float(nri_scores[c].get(layer, 0) or 0) for c in counties], dtype=np.float32)
            arr = idw_chunked(glat, glon, c_lats, c_lons, vals, power=2)
        layer_arrays[layer] = arr
        print(f"  [{layer}] {arr.min():.1f}–{arr.max():.1f}  ({time.time()-t1:.1f}s)")

    # ── Save ──────────────────────────────────────────────────────────────────
    n = len(glat)
    print(f"\nSaving {n:,} grid points → {OUTPUT}")
    col_order = ["lat", "lon"] + GRID_LAYERS
    with open(OUTPUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(col_order)
        for i in range(n):
            row = [round(float(glat[i]), 4), round(float(glon[i]), 4)]
            row += [round(float(layer_arrays[col][i]), 3) for col in GRID_LAYERS]
            w.writerow(row)

    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
