import os

def fetch_flood_zones():
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data", "raw"), exist_ok=True)
    # TODO: implement FEMA flood zone download and county-level processing
    return None


if __name__ == "__main__":
    fetch_flood_zones()
