#!/usr/bin/env python3
"""
src/generate_risk_images.py

Generate smooth risk raster PNG images for each TerraRisk layer.

For grid-based layers (overall, hurricane, tornado, sinkhole, sealevel,
wildfire) the 34K land-point CSV is interpolated to a 500×600 raster with
scipy.interpolate.griddata + Gaussian smoothing.

For the flood layer, 67 county centroid values are interpolated the same way
using the sfha_pct / flood_score column from county_risk_scores.csv.

Each output is a RGBA PNG with full alpha over land and transparent ocean,
sized to Florida's bounding box — ready for Folium ImageOverlay.

Run once (or whenever source data changes):
    python src/generate_risk_images.py
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import requests
from matplotlib.path import Path as MplPath
from PIL import Image
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RAW  = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")

GEOJSON_CACHE = os.path.join(RAW,  "fl_counties.geojson")
GRID_CSV      = os.path.join(PROC, "risk_grid.csv")
RISK_CSV      = os.path.join(PROC, "county_risk_scores.csv")
SURGE_CSV     = os.path.join(ROOT, "data", "raw", "storm_surge.csv")

# Florida bounding box — matches generate_risk_grid.py land mask
LAT_MIN, LAT_MAX = 24.4, 31.2
LON_MIN, LON_MAX = -87.7, -79.9

# Raster dimensions: height=lat axis (north→south), width=lon axis (west→east)
IMG_H, IMG_W = 500, 600

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

# Risk colormap: blue (minimal) → cyan → green → yellow → orange → dark red (extreme)
RISK_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "risk",
    list(zip(
        [0.0,      0.2,      0.4,      0.6,      0.8,      1.0],
        ["#0044CC","#00BBFF","#00FF88","#FFFF00","#FF8800","#AA0000"],
    )),
)

# Surge colormap: light blue (1-3 ft) → medium blue (3-6 ft) → dark blue (6-9 ft) → navy (9+ ft)
SURGE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "surge",
    list(zip(
        [0.0,      0.10,     0.25,     0.45,     0.70,     1.0],
        ["#C8E8FF","#6DB8F0","#2070D0","#0038A0","#001A6E","#00052A"],
    )),
)

# Gaussian smoothing sigma (pixels). All NRI layers use county centroids → similar smoothing.
SIGMA = {
    "eal":           6,
    "risk":          6,
    "hurricane":     7,
    "coastal_flood": 8,
    "inland_flood":  7,
    "tornado":       6,
    "wildfire":      6,
    "wind":          6,
    "surge":         8,
}

MAX_SURGE_FT = 20.0  # Gulf County Cat 5 ceiling — used for fixed normalization


# ── GeoJSON / land mask ────────────────────────────────────────────────────────

def fetch_fl_geojson() -> dict:
    if os.path.exists(GEOJSON_CACHE):
        with open(GEOJSON_CACHE) as f:
            return json.load(f)
    print("  Downloading Florida county GeoJSON...")
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


def build_land_mask(lats_2d: np.ndarray, lons_2d: np.ndarray, geojson: dict) -> np.ndarray:
    """Return bool (H, W) array — True where the pixel falls on Florida land."""
    H, W = lats_2d.shape
    # Stack as (lon, lat) pairs — MplPath expects (x, y) = (lon, lat)
    pts = np.column_stack([lons_2d.ravel(), lats_2d.ravel()])
    mask = np.zeros(H * W, dtype=bool)

    for feat in geojson["features"]:
        geom = feat["geometry"]
        rings = (
            [geom["coordinates"][0]]
            if geom["type"] == "Polygon"
            else [poly[0] for poly in geom["coordinates"]]
        )
        for ring in rings:
            ring_arr = np.array(ring)            # (M, 2) as [lon, lat]
            # bbox prefilter to skip rings that can't contain any target point
            lon0, lon1 = ring_arr[:, 0].min(), ring_arr[:, 0].max()
            lat0, lat1 = ring_arr[:, 1].min(), ring_arr[:, 1].max()
            cand = (
                (pts[:, 0] >= lon0) & (pts[:, 0] <= lon1) &
                (pts[:, 1] >= lat0) & (pts[:, 1] <= lat1)
            )
            if not cand.any():
                continue
            path = MplPath(ring_arr)
            inside = np.zeros(len(pts), dtype=bool)
            inside[cand] = path.contains_points(pts[cand])
            mask |= inside

    return mask.reshape(H, W)


# ── Raster generation ──────────────────────────────────────────────────────────

def make_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return (lats_2d, lons_2d) meshgrid of shape (H, W).
    Rows run north→south so row 0 = LAT_MAX."""
    lat_vec = np.linspace(LAT_MAX, LAT_MIN, IMG_H)
    lon_vec = np.linspace(LON_MIN, LON_MAX, IMG_W)
    lons_2d, lats_2d = np.meshgrid(lon_vec, lat_vec)
    return lats_2d.astype(np.float32), lons_2d.astype(np.float32)


def interpolate_smooth(
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    src_vals: np.ndarray,
    tgt_lats: np.ndarray,
    tgt_lons: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Linearly interpolate source scatter to target raster, then Gaussian smooth."""
    src_pts = np.column_stack([src_lons.astype(np.float64),
                               src_lats.astype(np.float64)])
    tgt_pts = np.column_stack([tgt_lons.ravel().astype(np.float64),
                               tgt_lats.ravel().astype(np.float64)])
    vals = src_vals.astype(np.float64)

    interp = griddata(src_pts, vals, tgt_pts, method="linear", fill_value=0.0)
    raster = interp.reshape(IMG_H, IMG_W).astype(np.float32)
    raster = np.nan_to_num(raster, nan=0.0)
    return gaussian_filter(raster, sigma=sigma).astype(np.float32)


def raster_to_png(
    raster: np.ndarray,
    land_mask: np.ndarray,
    out_path: str,
    cmap=None,
    fixed_vmax: float | None = None,
    raw_threshold: float = 0.0,
) -> None:
    """Normalise → colormap → alpha mask → save RGBA PNG.

    cmap: colormap to use (defaults to RISK_CMAP)
    fixed_vmax: if set, normalise to [0, fixed_vmax] instead of data range
    raw_threshold: pixels with raster value <= threshold get alpha=0 even on land
    """
    if cmap is None:
        cmap = RISK_CMAP

    # Build visible mask: land pixels above optional threshold
    visible = land_mask.copy()
    if raw_threshold > 0.0:
        visible = visible & (raster > raw_threshold)

    if fixed_vmax is not None:
        vmin, vmax = 0.0, fixed_vmax
    else:
        vis_vals = raster[visible]
        vmin = float(vis_vals.min()) if vis_vals.size else 0.0
        vmax = float(vis_vals.max()) if vis_vals.size else 1.0
    if vmax == vmin:
        vmax = vmin + 1e-9

    norm_arr = np.clip((raster - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = cmap(norm_arr)                           # (H, W, 4) float64
    rgba[:, :, 3] = np.where(visible, 1.0, 0.0)

    img = (rgba * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img, "RGBA").save(out_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(PROC, exist_ok=True)
    t0 = time.time()

    print("Loading Florida county GeoJSON...")
    gj = fetch_fl_geojson()

    print(f"Building raster grid ({IMG_H}×{IMG_W})...")
    lats_2d, lons_2d = make_grid()

    print("Building Florida land mask...")
    land_mask = build_land_mask(lats_2d, lons_2d, gj)
    n_land = int(land_mask.sum())
    print(f"  {n_land:,} / {IMG_H * IMG_W:,} pixels on land "
          f"({n_land / (IMG_H * IMG_W) * 100:.1f}%)")

    # ── NRI grid-based layers ──────────────────────────────────────────────────
    print("Loading risk grid...")
    grid = pd.read_csv(GRID_CSV)
    g_lats = grid["lat"].values.astype(np.float32)
    g_lons = grid["lon"].values.astype(np.float32)

    # name → grid CSV column; output → risk_{name}.png
    grid_layers = {
        "eal":           "eal_score",
        "risk":          "risk_score",
        "hurricane":     "hurricane_score",
        "coastal_flood": "coastal_flood_score",
        "inland_flood":  "inland_flood_score",
        "tornado":       "tornado_score",
        "wildfire":      "wildfire_score",
        "wind":          "wind_score",
        "surge":         "surge",
    }

    n_generated = 0
    for name, col in grid_layers.items():
        if col not in grid.columns:
            print(f"  [SKIP] {name}: column '{col}' missing from grid CSV")
            continue
        t1 = time.time()
        print(f"  [{name}] interpolating {len(g_lats):,} pts → {IMG_H}×{IMG_W}...")
        vals = grid[col].values.astype(np.float32)
        raster = interpolate_smooth(g_lats, g_lons, vals, lats_2d, lons_2d, SIGMA[name])
        out = os.path.join(PROC, f"risk_{name}.png")
        raster_to_png(raster, land_mask, out)
        print(f"    → {out}  ({time.time()-t1:.1f}s)")
        n_generated += 1

    # ── Storm surge — category-specific PNGs (Cat 1–5) ─────────────────────────
    if os.path.exists(SURGE_CSV):
        print("  [surge] generating Cat 1–5 surge images...")
        surge_df = pd.read_csv(SURGE_CSV)
        for cat_n in range(1, 6):
            col = f"cat{cat_n}_ft"
            t1 = time.time()
            s_lats, s_lons, s_vals = [], [], []
            for _, row in surge_df.iterrows():
                county = row["county"]
                depth = float(row.get(col, 0) or 0)
                if depth > 0 and county in COUNTY_CENTROIDS:
                    clat, clon = COUNTY_CENTROIDS[county]
                    s_lats.append(clat)
                    s_lons.append(clon)
                    s_vals.append(depth)
            if s_lats:
                s_lats_arr = np.array(s_lats, dtype=np.float32)
                s_lons_arr = np.array(s_lons, dtype=np.float32)
                s_vals_arr = np.array(s_vals, dtype=np.float32)
                surge_raster = interpolate_smooth(
                    s_lats_arr, s_lons_arr, s_vals_arr, lats_2d, lons_2d, SIGMA["surge"]
                )
                out = os.path.join(PROC, f"surge_cat{cat_n}.png")
                raster_to_png(
                    surge_raster, land_mask, out,
                    cmap=SURGE_CMAP,
                    fixed_vmax=MAX_SURGE_FT,
                    raw_threshold=0.4,
                )
                print(f"    → {out}  ({time.time()-t1:.1f}s)")
                n_generated += 1
    else:
        print("  [surge] storm_surge.csv not found — skipping surge images")

    print(f"\nGenerated {n_generated} PNG images in {PROC}/  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
