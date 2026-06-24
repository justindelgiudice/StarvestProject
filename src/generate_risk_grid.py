#!/usr/bin/env python3
"""
src/generate_risk_grid.py

Pre-compute a dense IDW risk grid covering all of Florida's land area.
Grid: 0.05° spacing (~5km), clipped to Florida county polygons via PIP.
Each point receives a 0-1 normalized risk score per layer:
  overall, hurricane, tornado, sinkhole, sealevel

Run once before launching the dashboard:
  python src/generate_risk_grid.py

Output: data/processed/risk_grid.csv (~10-12k rows)
Runtime: 30-90 seconds depending on machine.
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
STEP = 0.02   # ~2km spacing — dense enough that no individual dots visible at any zoom


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
    """Extract (bbox, list-of-rings) per county feature for efficient PIP."""
    out = []
    for feat in geojson["features"]:
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            all_rings = [geom["coordinates"][0]]
        else:  # MultiPolygon
            all_rings = [poly[0] for poly in geom["coordinates"]]
        all_lons = [c[0] for r in all_rings for c in r]
        all_lats = [c[1] for r in all_rings for c in r]
        out.append({
            "bbox": (min(all_lats), max(all_lats), min(all_lons), max(all_lons)),
            "rings": [np.array(r, dtype=np.float32) for r in all_rings],
        })
    return out


def pip_batch(test_lats, test_lons, ring):
    """
    Vectorized ray-casting PIP for N test points against one ring.
    ring: float32 array (M, 2) with columns [lon, lat].
    Returns bool array (N,).
    """
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
    """Return bool array: True where (lat, lon) falls on Florida land."""
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

def idw_chunked(glat, glon, slat, slon, sval, power=3, chunk=800):
    """Chunked inverse-distance weighting. Returns float32 array (N,)."""
    slat = np.asarray(slat, np.float32)
    slon = np.asarray(slon, np.float32)
    sval = np.asarray(sval, np.float32)
    N = len(glat)
    out = np.zeros(N, np.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        dl = (glat[s:e, None] - slat) ** 2  # (chunk, M)
        do = (glon[s:e, None] - slon) ** 2
        d2 = np.maximum(dl + do, 1e-9)
        w = 1.0 / (d2 ** power)
        out[s:e] = (w * sval).sum(1) / w.sum(1)
    return out


def norm01(arr):
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PROC, exist_ok=True)
    t0 = time.time()

    print("Loading Florida county polygons...")
    gj = fetch_fl_geojson()
    polys = prep_polys(gj)
    print(f"  {len(polys)} county polygons ({time.time()-t0:.1f}s)")

    # Generate all candidate grid points
    lats_arr = np.arange(LAT_MIN, LAT_MAX + STEP / 2, STEP, dtype=np.float32)
    lons_arr = np.arange(LON_MIN, LON_MAX + STEP / 2, STEP, dtype=np.float32)
    mesh_lats, mesh_lons = np.meshgrid(lats_arr, lons_arr, indexing="ij")
    all_lats = mesh_lats.ravel()
    all_lons = mesh_lons.ravel()
    print(f"Testing {len(all_lats):,} candidates ({len(lats_arr)}×{len(lons_arr)})...")

    # Vectorized PIP — clip to Florida land
    land = build_land_mask(all_lats, all_lons, polys)
    glat = all_lats[land]
    glon = all_lons[land]
    print(f"  {len(glat):,} land points retained ({time.time()-t0:.1f}s)")

    # ── Layer 1: Overall risk from county composite scores ────────────────────
    print("Computing overall risk...")
    risk_by_county = {}
    with open(os.path.join(PROC, "county_risk_scores.csv")) as f:
        for row in csv.DictReader(f):
            risk_by_county[row["county"]] = float(row["composite_risk_score"])

    ctr_lats, ctr_lons, ctr_vals = zip(*[
        (clat, clon, risk_by_county.get(c, 5.0) / 10.0)
        for c, (clat, clon) in COUNTY_CENTROIDS.items()
    ])
    overall = norm01(idw_chunked(glat, glon, ctr_lats, ctr_lons, ctr_vals, power=2))
    print(f"  {overall.min():.3f}–{overall.max():.3f} ({time.time()-t0:.1f}s)")

    # ── Layer 2: Hurricane (track points, weight = wind_knots / 165) ─────────
    print("Computing hurricane risk...")
    h_lats, h_lons, h_vals = [], [], []
    with open(os.path.join(RAW, "hurricanes.csv")) as f:
        for row in csv.DictReader(f):
            try:
                w = float(row["wind_knots"])
                if w > 0:
                    h_lats.append(float(row["latitude"]))
                    h_lons.append(float(row["longitude"]))
                    h_vals.append(min(1.0, w / 165.0))
            except (ValueError, KeyError):
                pass
    hurricane = norm01(idw_chunked(glat, glon, h_lats, h_lons, h_vals, power=3))
    print(f"  {hurricane.min():.3f}–{hurricane.max():.3f} ({time.time()-t0:.1f}s)")

    # ── Layer 3: Tornado (touchdown, weight = (EF+0.5) / 5.5) ────────────────
    print("Computing tornado risk...")
    t_lats, t_lons, t_vals = [], [], []
    with open(os.path.join(RAW, "tornadoes.csv")) as f:
        for row in csv.DictReader(f):
            try:
                ef = max(0, int(row["ef_scale"]))
                t_lats.append(float(row["latitude"]))
                t_lons.append(float(row["longitude"]))
                t_vals.append((ef + 0.5) / 5.5)
            except (ValueError, KeyError):
                pass
    tornado = norm01(idw_chunked(glat, glon, t_lats, t_lons, t_vals, power=3))
    print(f"  {tornado.min():.3f}–{tornado.max():.3f} ({time.time()-t0:.1f}s)")

    # ── Layer 4: Sinkhole (county centroid IDW weighted by sinkhole count) ────
    # Uniform-weight IDW always returns 1.0, so use county-level counts instead.
    print("Computing sinkhole risk...")
    count_by_county: dict[str, int] = {}
    with open(os.path.join(RAW, "sinkholes.csv")) as f:
        for row in csv.DictReader(f):
            c = row.get("county", "")
            if c:
                count_by_county[c] = count_by_county.get(c, 0) + 1
    max_count = max(count_by_county.values()) if count_by_county else 1
    s_lats, s_lons, s_vals = [], [], []
    for county, (clat, clon) in COUNTY_CENTROIDS.items():
        cnt = count_by_county.get(county, 0)
        if cnt > 0:
            s_lats.append(clat)
            s_lons.append(clon)
            s_vals.append(cnt / max_count)
    sinkhole = norm01(idw_chunked(glat, glon, s_lats, s_lons, s_vals, power=2))
    print(f"  {sinkhole.min():.3f}–{sinkhole.max():.3f} ({time.time()-t0:.1f}s)")

    # ── Layer 5: Sea level (county centroids, weight = SLR_2100 / 0.85) ──────
    print("Computing sea level risk...")
    slr_by_county = {}
    with open(os.path.join(RAW, "sealevel.csv")) as f:
        for row in csv.DictReader(f):
            try:
                slr_by_county[row["county"]] = float(row["projected_slr_2100_m"])
            except (ValueError, KeyError):
                pass

    sl_lats, sl_lons, sl_vals = zip(*[
        (clat, clon, min(1.0, slr_by_county.get(c, 0.65) / 0.85))
        for c, (clat, clon) in COUNTY_CENTROIDS.items()
    ])
    sealevel = norm01(idw_chunked(glat, glon, sl_lats, sl_lons, sl_vals, power=2))
    print(f"  {sealevel.min():.3f}–{sealevel.max():.3f} ({time.time()-t0:.1f}s)")

    # ── Layer 6: Wildfire (fire points, weight = FRP / max_FRP) ──────────────
    print("Computing wildfire risk...")
    wf_lats, wf_lons, wf_vals = [], [], []
    wf_path = os.path.join(RAW, "wildfires.csv")
    if os.path.exists(wf_path):
        with open(wf_path) as f:
            for row in csv.DictReader(f):
                try:
                    wf_lats.append(float(row["latitude"]))
                    wf_lons.append(float(row["longitude"]))
                    wf_vals.append(float(row["frp"]))
                except (ValueError, KeyError):
                    pass
    max_frp = max(wf_vals) if wf_vals else 1.0
    wf_vals_n = [v / max_frp for v in wf_vals]
    wildfire = norm01(idw_chunked(glat, glon, wf_lats, wf_lons, wf_vals_n, power=3, chunk=200))
    print(f"  {wildfire.min():.3f}–{wildfire.max():.3f} ({time.time()-t0:.1f}s)")

    # ── Save ──────────────────────────────────────────────────────────────────
    n = len(glat)
    print(f"\nSaving {n:,} grid points → {OUTPUT}")
    with open(OUTPUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "overall", "hurricane", "tornado", "sinkhole", "sealevel", "wildfire"])
        for i in range(n):
            w.writerow([
                round(float(glat[i]), 4),
                round(float(glon[i]), 4),
                round(float(overall[i]), 4),
                round(float(hurricane[i]), 4),
                round(float(tornado[i]), 4),
                round(float(sinkhole[i]), 4),
                round(float(sealevel[i]), 4),
                round(float(wildfire[i]), 4),
            ])

    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
