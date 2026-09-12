import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from utils import (
    load_feature_info, load_artifact, predict, models_available,
    RESULTS, FEATURE_META
)

# ─── Instrument-panel theme ──────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0E1116; }
    [data-testid="stSidebar"] { background-color: #14181F; }
    .panel-header {
        font-family: 'Courier New', monospace;
        color: #F5A623;
        letter-spacing: 0.08em;
        border-bottom: 1px solid #2A2F3A;
        padding-bottom: 14px;
        margin-bottom: 6px;
    }
    .panel-sub {
        font-family: 'Courier New', monospace;
        color: #6B7280;
        font-size: 0.85rem;
        letter-spacing: 0.04em;
    }
    .sensor-count {
        font-family: 'Courier New', monospace;
        color: #F5A623;
        font-size: 2.6rem;
        font-weight: 700;
        line-height: 1;
    }
    .sensor-label {
        font-family: 'Courier New', monospace;
        color: #6B7280;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .metric-box {
        background-color: #161B22;
        border: 1px solid #2A2F3A;
        border-left: 3px solid #F5A623;
        padding: 14px 18px;
        border-radius: 2px;
    }
    .metric-val { font-family: 'Courier New', monospace; color: #E8EAED; font-size: 1.4rem; font-weight: 700; }
    .metric-key { font-family: 'Courier New', monospace; color: #6B7280; font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase; }
    .result-rain {
        background: linear-gradient(135deg, #1A2540, #0E1116);
        border: 1px solid #3B82F6;
        border-radius: 4px; padding: 28px; text-align: center;
    }
    .result-dry {
        background: linear-gradient(135deg, #2A2410, #0E1116);
        border: 1px solid #F5A623;
        border-radius: 4px; padding: 28px; text-align: center;
    }
    .stButton button {
        background-color: #F5A623; color: #0E1116; font-family: 'Courier New', monospace;
        font-weight: 700; letter-spacing: 0.08em; border: none; border-radius: 2px;
        width: 100%; padding: 0.7rem;
    }
    .stButton button:hover { background-color: #FFC04D; }
    div[data-testid="stMetricValue"] { font-family: 'Courier New', monospace; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="panel-sub">WEATHER STATION · CONSOLE 01</p>', unsafe_allow_html=True)
st.markdown('<h1 class="panel-header">📡 BASELINE FORECAST PANEL</h1>', unsafe_allow_html=True)
st.markdown('<p class="panel-sub">Full sensor array — no feature selection applied. Every available reading feeds the Random Forest classifier.</p>', unsafe_allow_html=True)
st.write("")

if not models_available():
    st.error(
        "**Model files not found.** Place `model_mi.pkl`, `model_pso.pkl`, "
        "`scaler.pkl`, and `feature_info.json` inside the `models/` folder "
        "next to this app, then restart Streamlit.",
        icon="⚠️"
    )
    st.stop()

feature_info = load_feature_info()
scaler = load_artifact("scaler.pkl")
# Baseline uses MI's RF architecture is irrelevant here — baseline was trained
# separately on all features; if you saved it, point to it. Otherwise we
# reconstruct on the fly is not possible without the file, so we require it.
baseline_model = load_artifact("model_baseline.pkl")

all_features = feature_info["all_features"]
r = RESULTS["baseline"]

# ─── Top stats row ──────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="sensor-count">{16}</div><div class="sensor-label">Active Sensors</div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{r["test_accuracy"]*100:.2f}%</div><div class="metric-key">Test Accuracy</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{r["auc_roc"]:.4f}</div><div class="metric-key">AUC-ROC</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{r["test_f1"]*100:.2f}%</div><div class="metric-key">F1-Score</div></div>', unsafe_allow_html=True)

st.write("")
st.markdown("---")

left, right = st.columns([3, 2])

with left:
    st.markdown('<p class="panel-sub">SENSOR INPUT GRID</p>', unsafe_allow_html=True)
    input_values = {}
    grid = st.columns(2)
    for i, feat in enumerate(all_features):
        meta = FEATURE_META[feat]
        with grid[i % 2]:
            input_values[feat] = st.number_input(
                f"{meta['label']} ({meta['unit']})",
                min_value=float(meta["min"]), max_value=float(meta["max"]),
                value=float(meta["default"]), step=float(meta["step"]),
                key=f"baseline_{feat}"
            )

with right:
    st.markdown('<p class="panel-sub">MODEL SPECIFICATION</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-box" style="margin-bottom:10px;">
        <div class="metric-key">Algorithm</div>
        <div class="metric-val" style="font-size:1rem;">Random Forest (300 trees)</div>
    </div>
    <div class="metric-box" style="margin-bottom:10px;">
        <div class="metric-key">Feature Selection</div>
        <div class="metric-val" style="font-size:1rem;">None — all 16 retained</div>
    </div>
    <div class="metric-box" style="margin-bottom:10px;">
        <div class="metric-key">Class Balancing</div>
        <div class="metric-val" style="font-size:1rem;">SMOTE (in-fold)</div>
    </div>
    <div class="metric-box">
        <div class="metric-key">CV Accuracy</div>
        <div class="metric-val" style="font-size:1rem;">{r['cv_accuracy']*100:.2f}% ± {r['cv_accuracy_std']*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    run = st.button("▸ RUN FORECAST")

st.write("")

if run:
    if baseline_model is None:
        st.warning(
            "No `model_baseline.pkl` found in `models/`. The baseline model "
            "was not saved during your Colab session — re-run the baseline "
            "cell and add `joblib.dump(baseline['model'], 'model_baseline.pkl')` "
            "to your save step, then re-download.",
            icon="📦"
        )
    else:
        pred, proba = predict(baseline_model, scaler, all_features, all_features, input_values)
        rain_prob = proba[1] * 100
        st.markdown("---")
        if pred == 1:
            st.markdown(f"""
            <div class="result-rain">
                <div style="font-family:'Courier New',monospace; color:#93C5FD; font-size:0.85rem; letter-spacing:0.1em;">FORECAST OUTPUT</div>
                <div style="font-family:'Courier New',monospace; color:#E8EAED; font-size:2.2rem; font-weight:700; margin:8px 0;">🌧️ RAIN EXPECTED</div>
                <div style="font-family:'Courier New',monospace; color:#93C5FD; font-size:1rem;">Confidence: {rain_prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-dry">
                <div style="font-family:'Courier New',monospace; color:#FCD34D; font-size:0.85rem; letter-spacing:0.1em;">FORECAST OUTPUT</div>
                <div style="font-family:'Courier New',monospace; color:#E8EAED; font-size:2.2rem; font-weight:700; margin:8px 0;">☀️ NO RAIN EXPECTED</div>
                <div style="font-family:'Courier New',monospace; color:#FCD34D; font-size:1rem;">Confidence: {100-rain_prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
