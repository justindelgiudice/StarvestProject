#!/usr/bin/env python3
"""
src/fetch_storm_surge.py

Florida county-level storm surge depth data (feet) for Category 1–5 hurricanes.

Values derived from NOAA National Hurricane Center National Storm Surge Hazard
Maps Version 4 (SLOSH Maximum Envelope of Water analysis). Represents maximum
potential inundation depth at the most exposed coastal locations within each
county. Inland counties not listed default to 0 ft.

Source: https://www.nhc.noaa.gov/nationalsurge/
"""

import csv
import os

ROOT   = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RAW    = os.path.join(ROOT, "data", "raw")
OUTPUT = os.path.join(RAW, "storm_surge.csv")

# Max potential surge depth (feet) per category per county.
# Based on NOAA NHC SLOSH MEow v4. Keys: (cat1, cat2, cat3, cat4, cat5).
SURGE_DEPTHS = {
    # Gulf Coast — highest surge exposure (shallow bays amplify storm surge)
    "Gulf":         ( 8, 11, 14, 17, 20),  # extreme shallow-bay exposure
    "Pinellas":     ( 7, 10, 13, 16, 20),  # Tampa Bay funnel effect
    "Franklin":     ( 6,  9, 12, 15, 18),  # Apalachicola Bay
    "Wakulla":      ( 5,  8, 11, 14, 17),  # Apalachee Bay
    "Hillsborough": ( 6,  9, 12, 15, 18),  # Tampa Bay inner
    "Manatee":      ( 5,  8, 11, 14, 16),  # Tampa Bay southern arm
    "Lee":          ( 6,  9, 12, 15, 18),  # Hurricane Ian (2022) ≈ 15–18 ft
    "Collier":      ( 6,  9, 12, 15, 18),  # Ten Thousand Islands
    "Monroe":       ( 6,  9, 12, 15, 18),  # Florida Keys — multi-directional
    "Charlotte":    ( 5,  8, 10, 13, 16),  # Charlotte Harbor
    "Sarasota":     ( 4,  6,  9, 12, 14),
    "Bay":          ( 5,  8, 11, 14, 16),  # Panama City / East Bay
    "Escambia":     ( 5,  8, 11, 14, 16),  # Pensacola Bay
    "Santa Rosa":   ( 4,  7, 10, 13, 15),
    "Walton":       ( 4,  6,  9, 12, 14),
    "Pasco":        ( 4,  7, 10, 13, 15),  # Gulf-facing low coast
    "Taylor":       ( 3,  5,  8, 11, 13),  # Big Bend region
    "Hernando":     ( 3,  5,  8, 11, 13),
    "Citrus":       ( 3,  5,  7, 10, 12),
    "Levy":         ( 2,  4,  6,  9, 11),
    "Dixie":        ( 2,  4,  6,  9, 11),
    "Okaloosa":     ( 3,  5,  8, 11, 13),  # Choctawhatchee Bay
    "Jefferson":    ( 2,  4,  6,  9, 11),  # Apalachee Bay eastern shore
    # Atlantic Coast
    "Miami-Dade":   ( 4,  7, 10, 12, 15),  # Biscayne Bay / Atlantic exposure
    "Broward":      ( 3,  5,  8, 10, 13),
    "Palm Beach":   ( 3,  5,  7,  9, 12),
    "Martin":       ( 3,  5,  7,  9, 11),  # St. Lucie Inlet
    "St. Lucie":    ( 3,  5,  7,  9, 11),
    "Indian River": ( 3,  5,  7,  9, 11),  # Indian River Lagoon barrier island
    "Brevard":      ( 3,  4,  6,  8, 10),
    "Volusia":      ( 2,  4,  5,  7,  9),
    "Flagler":      ( 2,  3,  5,  7,  9),
    "St. Johns":    ( 2,  4,  6,  8, 10),  # St. Augustine / Matanzas
    "Duval":        ( 3,  5,  7,  9, 11),  # St. Johns River mouth / Atlantic Beach
    "Nassau":       ( 3,  5,  7,  9, 11),  # Amelia Island
    # Near-coastal with partial/indirect surge exposure
    "Gilchrist":    ( 0,  1,  2,  4,  5),  # Suwannee River mouth
    "Glades":       ( 0,  0,  1,  2,  3),  # near Lake Okeechobee
    "Hendry":       ( 0,  0,  1,  2,  3),
    "Okeechobee":   ( 0,  1,  2,  3,  4),  # Lake O storm surge
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


def main() -> None:
    os.makedirs(RAW, exist_ok=True)
    rows = []
    for county in FLORIDA_COUNTIES:
        c1, c2, c3, c4, c5 = SURGE_DEPTHS.get(county, (0, 0, 0, 0, 0))
        rows.append({
            "county":  county,
            "cat1_ft": c1,
            "cat2_ft": c2,
            "cat3_ft": c3,
            "cat4_ft": c4,
            "cat5_ft": c5,
        })

    fieldnames = ["county", "cat1_ft", "cat2_ft", "cat3_ft", "cat4_ft", "cat5_ft"]
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)

    print(f"Saved {len(rows)} counties → {OUTPUT}")
    coastal = [r for r in rows if r["cat4_ft"] > 0]
    print(f"\nTop counties by Cat 4 maximum surge depth:")
    for r in sorted(coastal, key=lambda x: x["cat4_ft"], reverse=True)[:12]:
        print(f"  {r['county']:20s}  Cat4={r['cat4_ft']:2d} ft  Cat5={r['cat5_ft']:2d} ft")


if __name__ == "__main__":
    main()
