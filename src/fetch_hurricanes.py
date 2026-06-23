"""
Download NOAA HURDAT2 Atlantic hurricane database.
Filter for storms that passed within 100 miles of Florida since 1950.
Save all track records for matching storms to data/raw/hurricanes.csv.
"""

import csv
import math
import os
import requests
from datetime import datetime

HURDAT2_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-02272026.txt"
OUTPUT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "raw", "hurricanes.csv")
)

START_YEAR = 1950
MAX_DISTANCE_MILES = 100.0

# Bounding box + reference points spread across Florida to capture storms
# that hit any part of the state (not just the geographic center)
FLORIDA_REFERENCE_POINTS = [
    (30.33, -81.66),  # Jacksonville (NE)
    (28.54, -81.38),  # Orlando (Central)
    (27.77, -82.64),  # Tampa (Gulf Coast)
    (25.77, -80.19),  # Miami (SE)
    (24.56, -81.78),  # Key West (Keys)
    (30.45, -87.22),  # Pensacola (NW Panhandle)
    (29.65, -85.35),  # Panama City area (Panhandle Gulf)
]


def haversine_miles(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 3958.8 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_hurdat2(text):
    storms = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        parts = line.split(",")
        # Header line: AL092004, IAN, 60,
        if len(parts) < 3 or len(parts[0].strip()) < 8:
            i += 1
            continue

        storm_id = parts[0].strip()
        storm_name = parts[1].strip()
        try:
            record_count = int(parts[2].strip())
            year = int(storm_id[4:8])
        except (ValueError, IndexError):
            i += 1
            continue

        records = []
        for j in range(i + 1, i + 1 + record_count):
            if j >= len(lines):
                break
            rparts = lines[j].split(",")
            if len(rparts) < 7:
                continue
            try:
                date_str = rparts[0].strip()
                time_str = rparts[1].strip().zfill(4)
                status = rparts[3].strip()
                lat_s = rparts[4].strip()
                lon_s = rparts[5].strip()
                wind_s = rparts[6].strip()

                lat = float(lat_s[:-1]) * (1 if lat_s[-1] == "N" else -1)
                lon = float(lon_s[:-1]) * (1 if lon_s[-1] == "E" else -1)
                dt = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M")
                wind = int(wind_s) if wind_s.lstrip("-").isdigit() and int(wind_s) > 0 else None
            except (ValueError, IndexError):
                continue

            records.append({
                "storm_id": storm_id,
                "storm_name": storm_name,
                "year": year,
                "datetime": dt.isoformat(),
                "status": status,
                "latitude": lat,
                "longitude": lon,
                "wind_knots": wind,
            })

        storms.append({"year": year, "records": records, "storm_id": storm_id, "storm_name": storm_name})
        i += 1 + record_count

    return storms


def impacted_florida(storm):
    if storm["year"] < START_YEAR:
        return False
    for rec in storm["records"]:
        for ref_lat, ref_lon in FLORIDA_REFERENCE_POINTS:
            if haversine_miles(ref_lat, ref_lon, rec["latitude"], rec["longitude"]) <= MAX_DISTANCE_MILES:
                return True
    return False


def main():
    print("Downloading NOAA HURDAT2 Atlantic hurricane database...")
    resp = requests.get(HURDAT2_URL, timeout=60)
    resp.raise_for_status()

    storms = parse_hurdat2(resp.text)
    florida_storms = [s for s in storms if impacted_florida(s)]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fieldnames = ["storm_id", "storm_name", "year", "datetime", "status", "latitude", "longitude", "wind_knots"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for storm in florida_storms:
            writer.writerows(storm["records"])

    print(f"Found {len(florida_storms)} storms that impacted Florida since {START_YEAR}.")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
