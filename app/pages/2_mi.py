import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from utils import (
    load_feature_info, load_artifact, predict, models_available,
    RESULTS, FEATURE_META
)

st.markdown("""
<style>
    .stApp { background-color: #FAFAF8; color: #1C1C1A; }
    [data-testid="stSidebar"] { background-color: #F2F1EC; }
    label, p, div, span { color: #1C1C1A !important; }
    .stNumberInput label { color: #1C1C1A !important; font-family: Georgia, serif; }
    .stCaption { color: #57534E !important; }
    .mi-eyebrow {
        font-family: Georgia, 'Times New Roman', serif;
        color: #0F766E !important; font-style: italic; font-size: 0.95rem;
        letter-spacing: 0.02em;
    }
    .mi-title {
        font-family: Georgia, 'Times New Roman', serif;
        color: #1C1C1A !important; font-weight: 700;
        border-bottom: 2px solid #0F766E;
        padding-bottom: 12px; margin-bottom: 4px;
    }
    .mi-desc { font-family: Georgia, serif; color: #57534E !important; font-size: 0.98rem; }
    .stat-card {
        background-color: #FFFFFF; border: 1px solid #E7E5E0;
        border-top: 3px solid #0F766E; border-radius: 4px; padding: 16px 18px;
    }
    .stat-val { font-family: Georgia, serif; color: #1C1C1A !important; font-size: 1.6rem; font-weight: 700; }
    .stat-key { font-family: -apple-system, sans-serif; color: #78716C !important; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; }
    .result-rain {
        background-color: #ECFEFF; border: 1px solid #0F766E;
        border-radius: 6px; padding: 30px; text-align: center;
    }
    .result-dry {
        background-color: #FFFBEB; border: 1px solid #D97706;
        border-radius: 6px; padding: 30px; text-align: center;
    }
    .stButton button {
        background-color: #0F766E; color: #FFFFFF !important;
        font-family: Georgia, serif; font-weight: 700;
        border: none; border-radius: 4px; width: 100%; padding: 0.7rem;
    }
    .stButton button:hover { background-color: #0D9488; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="mi-eyebrow">Filter Method &nbsp;·&nbsp; Information-Theoretic Ranking</p>', unsafe_allow_html=True)
st.markdown('<h1 class="mi-title">📊 Mutual Information Forecast</h1>', unsafe_allow_html=True)
st.markdown('<p class="mi-desc">Eight features, ranked independently by statistical dependency on rainfall outcome, selected in a single deterministic pass.</p>', unsafe_allow_html=True)
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
mi_model = load_artifact("model_mi.pkl")
all_features = feature_info["all_features"]
mi_features = feature_info["mi_features"]
r = RESULTS["mi"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="stat-card"><div class="stat-val">8 / 16</div><div class="stat-key">Features Used</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-card"><div class="stat-val">{r["test_accuracy"]*100:.2f}%</div><div class="stat-key">Test Accuracy</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-card"><div class="stat-val">{r["auc_roc"]:.4f}</div><div class="stat-key">AUC-ROC</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="stat-card"><div class="stat-val">~25s</div><div class="stat-key">Selection Time</div></div>', unsafe_allow_html=True)

st.write("")
st.markdown("---")

left, right = st.columns([3, 2])

with left:
    st.markdown('<p class="mi-eyebrow">Ranked Feature Inputs</p>', unsafe_allow_html=True)
    st.caption("Ordered by Mutual Information score (highest dependency on RainTomorrow first).")
    input_values = {}
    col_a, col_b = st.columns(2)
    for i, feat in enumerate(mi_features, start=1):
        meta = FEATURE_META[feat]
        col = col_a if i % 2 != 0 else col_b
        with col:
            input_values[feat] = st.number_input(
                f"#{i:02d} · {meta['label']} ({meta['unit']})",
                min_value=float(meta["min"]), max_value=float(meta["max"]),
                value=float(meta["default"]), step=float(meta["step"]),
                key=f"mi_{feat}"
            )

    for feat in all_features:
        if feat not in input_values:
            input_values[feat] = FEATURE_META[feat]["default"]

with right:
    st.markdown('<p class="mi-eyebrow">Method Notes</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-card" style="margin-bottom:10px;">
        <div class="stat-key">Selection Criterion</div>
        <div class="stat-val" style="font-size:1rem;">Shannon entropy / KNN-MI estimator</div>
    </div>
    <div class="stat-card" style="margin-bottom:10px;">
        <div class="stat-key">CV Accuracy</div>
        <div class="stat-val" style="font-size:1rem;">{r['cv_accuracy']*100:.2f}% ± {r['cv_accuracy_std']*100:.2f}%</div>
    </div>
    <div class="stat-card">
        <div class="stat-key">Known Limitation</div>
        <div class="stat-val" style="font-size:0.92rem; font-weight:400;">Evaluates each feature independently — may retain correlated pairs (e.g. Humidity9am &amp; Humidity3pm).</div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    run = st.button("Run Forecast →")

st.write("")

if run:
    pred, proba = predict(mi_model, scaler, all_features, mi_features, input_values)
    rain_prob = proba[1] * 100
    st.markdown("---")
    if pred == 1:
        st.markdown(f"""
        <div class="result-rain">
            <div style="font-family:-apple-system,sans-serif; color:#0F766E; font-size:0.8rem; letter-spacing:0.08em; text-transform:uppercase;">Forecast</div>
            <div style="font-family:Georgia,serif; color:#1C1C1A; font-size:2.1rem; font-weight:700; margin:8px 0;">🌧️ Rain Expected Tomorrow</div>
            <div style="font-family:Georgia,serif; color:#0F766E; font-size:1.05rem;">Model confidence: {rain_prob:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-dry">
            <div style="font-family:-apple-system,sans-serif; color:#B45309; font-size:0.8rem; letter-spacing:0.08em; text-transform:uppercase;">Forecast</div>
            <div style="font-family:Georgia,serif; color:#1C1C1A; font-size:2.1rem; font-weight:700; margin:8px 0;">☀️ No Rain Expected Tomorrow</div>
            <div style="font-family:Georgia,serif; color:#B45309; font-size:1.05rem;">Model confidence: {100-rain_prob:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)