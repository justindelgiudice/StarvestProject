import csv
import math
import os
import requests
from datetime import datetime

HURDAT2_URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-050524.txt"
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_PATH = os.path.normpath(os.path.join(RAW_DIR, "hurricanes.csv"))

FLORIDA_LAT_MIN = 24.0
FLORIDA_LAT_MAX = 31.0
FLORIDA_LON_MIN = -88.0
FLORIDA_LON_MAX = -79.0
MAX_DISTANCE_MILES = 100.0
START_YEAR = 1950

FLORIDA_CENTERS = [
    (27.9944024, -81.7602544),  # Florida center
]


def haversine_miles(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return 3958.8 * c


def parse_hurdat2(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    lines = response.text.splitlines()
    storms = []
    i = 0

    while i < len(lines):
        header = lines[i].strip()
        if not header:
            i += 1
            continue

        header_parts = header.split(",")
        storm_id = header_parts[0].strip()
        storm_name = header_parts[1].strip()
        record_count = int(header_parts[2].strip())
        storm_year = int(storm_id[:4])
        storm_records = []

        for j in range(i + 1, i + 1 + record_count):
            parts = lines[j].split(",")
            if len(parts) < 7:
                continue

            date_str = parts[0].strip()
            time_str = parts[1].strip()
            record_type = parts[2].strip()
            latitude = parts[4].strip()
            longitude = parts[5].strip()
            wind = parts[6].strip()

            try:
                lat = float(latitude[:-1]) * (1 if latitude[-1] == "N" else -1)
                lon = float(longitude[:-1]) * (1 if longitude[-1] == "E" else -1)
            except (ValueError, IndexError):
                continue

            record_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y%m%d %H%M")
            storm_records.append({
                "storm_id": storm_id,
                "storm_name": storm_name,
                "year": storm_year,
                "record_datetime": record_datetime,
                "record_type": record_type,
                "latitude": lat,
                "longitude": lon,
                "wind_knots": int(wind) if wind.isdigit() else None,
            })

        storms.append({
            "storm_id": storm_id,
            "storm_name": storm_name,
            "year": storm_year,
            "records": storm_records,
        })
        i += 1 + record_count

    return storms


def storm_impacted_florida(storm):
    if storm["year"] < START_YEAR:
        return False

    for record in storm["records"]:
        for center_lat, center_lon in FLORIDA_CENTERS:
            distance = haversine_miles(center_lat, center_lon, record["latitude"], record["longitude"])
            if distance <= MAX_DISTANCE_MILES:
                return True
    return False


def save_storms(storms, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "storm_id",
        "storm_name",
        "year",
        "record_datetime",
        "record_type",
        "latitude",
        "longitude",
        "wind_knots",
    ]

    with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for storm in storms:
            for record in storm["records"]:
                row = {
                    "storm_id": storm["storm_id"],
                    "storm_name": storm["storm_name"],
                    "year": storm["year"],
                    "record_datetime": record["record_datetime"].isoformat(),
                    "record_type": record["record_type"],
                    "latitude": record["latitude"],
                    "longitude": record["longitude"],
                    "wind_knots": record["wind_knots"],
                }
                writer.writerow(row)


def main():
    print("Downloading NOAA HURDAT2 Atlantic hurricane database...")
    storms = parse_hurdat2(HURDAT2_URL)
    matching_storms = [storm for storm in storms if storm_impacted_florida(storm)]
    save_storms(matching_storms, OUTPUT_PATH)
    print(f"Found {len(matching_storms)} storms that impacted Florida since {START_YEAR}.")
    print(f"Saved hurricane records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
