import streamlit as st

st.set_page_config(
    page_title="Rain Prediction — MI vs PSO vs Baseline",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

baseline_page = st.Page("pages/1_baseline.py", title="Baseline · All 16 Features", icon="📡")
mi_page       = st.Page("pages/2_mi.py",       title="Mutual Information · 8 Features", icon="📊")
pso_page      = st.Page("pages/3_pso.py",      title="Particle Swarm · 8 Features", icon="🌀")

pg = st.navigation([baseline_page, mi_page, pso_page])
pg.run()
