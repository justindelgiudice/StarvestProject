import os

def build_risk_score():
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data", "processed"), exist_ok=True)
    # TODO: combine hurricane, flood, and sea level data into a composite risk score per Florida county
    return None


if __name__ == "__main__":
    build_risk_score()
