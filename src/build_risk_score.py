"""
Combine hurricane, flood zone, sea level rise, tornado, and wildfire data
into a composite risk score (0-10) per Florida county.

Weights:  Hurricane 35% · Flood 30% · Sea Level 20% · Tornado 10% · Wildfire 5%

Saves to data/processed/county_risk_scores.csv.
"""

import csv
import math
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RAW  = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")

PATHS = {
    "hurricanes": os.path.join(RAW, "hurricanes.csv"),
    "flood":      os.path.join(RAW, "flood_zones.csv"),
    "sealevel":   os.path.join(RAW, "sealevel.csv"),
    "tornadoes":  os.path.join(RAW, "tornadoes.csv"),
    "wildfires":  os.path.join(RAW, "wildfires.csv"),
    "output":     os.path.join(PROC, "county_risk_scores.csv"),
}

FLORIDA_COUNTIES = [
    "Alachua", "Baker", "Bay", "Bradford", "Brevard", "Broward", "Calhoun",
    "Charlotte", "Citrus", "Clay", "Collier", "Columbia", "DeSoto", "Dixie",
    "Duval", "Escambia", "Flagler", "Franklin", "Gadsden", "Gilchrist",
    "Glades", "Gulf", "Hamilton", "Hardee", "Hendry", "Hernando", "Highlands",
    "Hillsborough", "Holmes", "Indian River", "Jackson", "Jefferson", "Lafayette",
    "Lake", "Lee", "Leon", "Levy", "Liberty", "Madison", "Manatee", "Marion",
    "Martin", "Miami-Dade", "Monroe", "Nassau", "Okaloosa", "Okeechobee",
    "Orange", "Osceola", "Palm Beach", "Pasco", "Pinellas", "Polk", "Putnam",
    "St. Johns", "St. Lucie", "Santa Rosa", "Sarasota", "Seminole", "Sumter",
    "Suwannee", "Taylor", "Union", "Volusia", "Wakulla", "Walton", "Washington",
]

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

WEIGHTS = {
    "hurricane": 0.35,
    "flood":     0.30,
    "sealevel":  0.20,
    "tornado":   0.10,
    "wildfire":  0.05,
}

COUNTY_RADIUS_MILES = 75.0
MAX_STORM_COUNT  = 60.0
MAX_MAX_WIND     = 175.0
MAX_SFHA_PCT     = 60.0
MAX_SLR_2100     = 0.85
MAX_TORNADO_CNT  = 200.0    # FL county tornado count, historical high ≈ 180
MAX_FIRE_SCORE   = 200.0    # count × avg_frp normalization ceiling


def haversine(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    dphi = math.radians(lat2 - lat1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 3958.8 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def norm(value: float, max_val: float) -> float:
    return min(10.0, round(10.0 * value / max_val, 3)) if max_val > 0 else 0.0


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_hurricane_stats() -> dict:
    county_storms: dict[str, set] = {c: set() for c in FLORIDA_COUNTIES}
    county_wind: dict[str, float]  = {c: 0.0   for c in FLORIDA_COUNTIES}
    with open(PATHS["hurricanes"], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rlat = float(row["latitude"])
                rlon = float(row["longitude"])
                wind = float(row["wind_knots"] or 0)
            except (ValueError, KeyError):
                continue
            for county, (clat, clon) in COUNTY_CENTROIDS.items():
                if haversine(clat, clon, rlat, rlon) <= COUNTY_RADIUS_MILES:
                    county_storms[county].add(row["storm_id"])
                    county_wind[county] = max(county_wind[county], wind)
    return {c: {"storm_count": len(s), "max_wind_knots": county_wind[c]}
            for c, s in county_storms.items()}


def load_flood_stats() -> dict:
    result = {}
    with open(PATHS["flood"], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pct = row.get("sfha_pct")
            result[row["county"]] = float(pct) if pct not in (None, "", "None") else 0.0
    return result


def load_sealevel_stats() -> dict:
    result = {}
    with open(PATHS["sealevel"], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            slr = row.get("projected_slr_2100_m")
            result[row["county"]] = float(slr) if slr not in (None, "", "None") else 0.0
    return result


def load_tornado_stats() -> dict:
    """Count tornado touchdowns within 75 miles of each county centroid."""
    counts: dict[str, int] = {c: 0 for c in FLORIDA_COUNTIES}
    if not os.path.exists(PATHS["tornadoes"]):
        return counts
    with open(PATHS["tornadoes"], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                tlat = float(row["latitude"])
                tlon = float(row["longitude"])
            except (ValueError, KeyError):
                continue
            for county, (clat, clon) in COUNTY_CENTROIDS.items():
                if haversine(clat, clon, tlat, tlon) <= COUNTY_RADIUS_MILES:
                    counts[county] += 1
    return counts


def load_wildfire_stats() -> dict:
    """
    Compute per-county wildfire intensity score = count × avg_FRP
    using detections within 75 miles of each county centroid.
    """
    counts: dict[str, int]   = {c: 0   for c in FLORIDA_COUNTIES}
    frp_sum: dict[str, float] = {c: 0.0 for c in FLORIDA_COUNTIES}
    if not os.path.exists(PATHS["wildfires"]):
        return {c: {"fire_count": 0, "avg_frp": 0.0} for c in FLORIDA_COUNTIES}
    with open(PATHS["wildfires"], newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                flat = float(row["latitude"])
                flon = float(row["longitude"])
                frp  = float(row["frp"])
            except (ValueError, KeyError):
                continue
            for county, (clat, clon) in COUNTY_CENTROIDS.items():
                if haversine(clat, clon, flat, flon) <= COUNTY_RADIUS_MILES:
                    counts[county] += 1
                    frp_sum[county] += frp
    return {
        c: {
            "fire_count": counts[c],
            "avg_frp": round(frp_sum[c] / counts[c], 2) if counts[c] > 0 else 0.0,
        }
        for c in FLORIDA_COUNTIES
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Building composite risk scores (5-factor model)...")
    h_stats = load_hurricane_stats()
    f_stats = load_flood_stats()
    s_stats = load_sealevel_stats()
    t_stats = load_tornado_stats()
    w_stats = load_wildfire_stats()

    os.makedirs(PROC, exist_ok=True)
    rows = []

    for county in FLORIDA_COUNTIES:
        h = h_stats.get(county, {"storm_count": 0, "max_wind_knots": 0})
        storm_score = norm(h["storm_count"],    MAX_STORM_COUNT)
        wind_score  = norm(h["max_wind_knots"], MAX_MAX_WIND)
        hurricane_score = round(0.6 * storm_score + 0.4 * wind_score, 3)

        flood_score    = norm(f_stats.get(county, 0.0), MAX_SFHA_PCT)
        sealevel_score = norm(s_stats.get(county, 0.0), MAX_SLR_2100)
        tornado_score  = norm(t_stats.get(county, 0),   MAX_TORNADO_CNT)

        wf = w_stats.get(county, {"fire_count": 0, "avg_frp": 0.0})
        fire_intensity  = wf["fire_count"] * wf["avg_frp"] / 1000.0  # scale: count×MW/1000
        wildfire_score  = norm(fire_intensity, MAX_FIRE_SCORE / 1000.0)

        composite = round(
            WEIGHTS["hurricane"] * hurricane_score
            + WEIGHTS["flood"]    * flood_score
            + WEIGHTS["sealevel"] * sealevel_score
            + WEIGHTS["tornado"]  * tornado_score
            + WEIGHTS["wildfire"] * wildfire_score,
            3,
        )

        rows.append({
            "county":           county,
            "storm_count":      h["storm_count"],
            "max_wind_knots":   h["max_wind_knots"],
            "hurricane_score":  hurricane_score,
            "sfha_pct":         f_stats.get(county, 0.0),
            "flood_score":      flood_score,
            "slr_2100_m":       s_stats.get(county, 0.0),
            "sealevel_score":   sealevel_score,
            "tornado_count":    t_stats.get(county, 0),
            "tornado_score":    tornado_score,
            "fire_count":       wf["fire_count"],
            "avg_frp":          wf["avg_frp"],
            "wildfire_score":   wildfire_score,
            "composite_risk_score": composite,
        })

    rows.sort(key=lambda r: r["composite_risk_score"], reverse=True)

    fieldnames = [
        "county", "storm_count", "max_wind_knots", "hurricane_score",
        "sfha_pct", "flood_score", "slr_2100_m", "sealevel_score",
        "tornado_count", "tornado_score", "fire_count", "avg_frp", "wildfire_score",
        "composite_risk_score",
    ]
    with open(PATHS["output"], "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)

    print(f"Saved {len(rows)} counties → {PATHS['output']}")
    print("\nTop 10 highest-risk counties:")
    for r in rows[:10]:
        print(f"  {r['county']:20s}  composite={r['composite_risk_score']:.2f}  "
              f"storms={r['storm_count']}  flood={r['sfha_pct']:.0f}%  "
              f"tornado={r['tornado_count']}  fires={r['fire_count']}")


if __name__ == "__main__":
    main()
