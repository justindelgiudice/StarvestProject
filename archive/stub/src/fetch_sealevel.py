import os

def fetch_sealevel():
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data", "raw"), exist_ok=True)
    # TODO: implement NASA sea level rise projection download for Florida coastline
    return None


if __name__ == "__main__":
    fetch_sealevel()
