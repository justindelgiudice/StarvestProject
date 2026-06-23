"""
Download NOAA SPC historical tornado records for Florida (1950-2023).
Source: https://www.spc.noaa.gov/wcm/data/1950-2023_actual_tornadoes.csv
Each record is a touchdown point with EF scale.
Saves to data/raw/tornadoes.csv.
"""

import csv
import io
import os
import requests

URL = "https://www.spc.noaa.gov/wcm/data/1950-2023_actual_tornadoes.csv"
OUTPUT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "raw", "tornadoes.csv")
)

# Florida bounding box (generous, covers panhandle through Keys)
LAT_MIN, LAT_MAX = 24.3, 31.2
LON_MIN, LON_MAX = -87.9, -79.6


def main():
    print("Downloading NOAA SPC tornado data (1950-2023)...")
    resp = requests.get(URL, timeout=120)
    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    rows = []
    for rec in reader:
        if rec.get("st", "").strip() != "FL":
            continue
        try:
            lat = float(rec["slat"])
            lon = float(rec["slon"])
            mag = int(rec["mag"])
            yr = int(rec["yr"])
        except (ValueError, KeyError):
            continue

        # slon in NOAA SPC data is already negative for W longitudes
        if lon > 0:
            lon = -lon

        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            continue

        ef = max(0, mag)   # -9 (unknown/pre-EF) → 0
        rows.append({
            "latitude": lat,
            "longitude": lon,
            "ef_scale": ef,
            "year": yr,
        })

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["latitude", "longitude", "ef_scale", "year"])
        writer.writeheader()
        writer.writerows(rows)

    ef_counts = {}
    for r in rows:
        ef_counts[r["ef_scale"]] = ef_counts.get(r["ef_scale"], 0) + 1

    print(f"Found {len(rows)} Florida tornado touchdowns (1950-2023).")
    for ef in sorted(ef_counts):
        print(f"  EF{ef}: {ef_counts[ef]}")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
