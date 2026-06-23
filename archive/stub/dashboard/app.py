import streamlit as st

st.set_page_config(page_title="TerraRisk", layout="wide")
st.title("TerraRisk — Florida Climate Risk Intelligence")

st.markdown(
    "This dashboard will visualize a Florida county map colored by the composite risk score for hurricane, flood, and sea level exposure."
)

st.info("Run `src/build_risk_score.py` to generate processed county risk scores before visualizing them here.")
