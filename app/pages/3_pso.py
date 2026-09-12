import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from utils import (
    load_feature_info, load_artifact, predict, models_available,
    RESULTS, FEATURE_META
)

# ─── Swarm theme: deep indigo/violet, organic, in-motion ────────────────
st.markdown("""
<style>
    .stApp { background-color: #1A1530; }
    [data-testid="stSidebar"] { background-color: #211B3D; }
    .pso-eyebrow {
        font-family: 'Trebuchet MS', sans-serif; color: #C084FC;
        font-size: 0.9rem; letter-spacing: 0.06em;
    }
    .pso-title {
        font-family: 'Trebuchet MS', sans-serif; color: #F5F3FF; font-weight: 700;
        margin-bottom: 4px;
    }
    .pso-desc { font-family: 'Trebuchet MS', sans-serif; color: #B8B0D8; }
    .swarm-card {
        background: radial-gradient(circle at top left, #2D2456, #1A1530);
        border: 1px solid #4C3A8C; border-radius: 14px; padding: 16px 18px;
    }
    .swarm-val { font-family: 'Trebuchet MS', sans-serif; color: #F5F3FF; font-size: 1.6rem; font-weight: 700; }
    .swarm-key { font-family: 'Trebuchet MS', sans-serif; color: #A78BDA; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; }
    .particle-dot {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background: #C084FC; margin-right: 6px;
        box-shadow: 0 0 8px #C084FC;
    }
    .feature-pill {
        background-color: #2D2456; border: 1px solid #4C3A8C; border-radius: 20px;
        padding: 6px 14px; margin: 4px; display: inline-block;
        font-family: 'Trebuchet MS', sans-serif; color: #E9D5FF; font-size: 0.85rem;
    }
    .result-rain {
        background: radial-gradient(circle at top, #2A2160, #1A1530);
        border: 1px solid #818CF8; border-radius: 14px; padding: 30px; text-align: center;
    }
    .result-dry {
        background: radial-gradient(circle at top, #3D2A60, #1A1530);
        border: 1px solid #C084FC; border-radius: 14px; padding: 30px; text-align: center;
    }
    .stButton button {
        background: linear-gradient(135deg, #7C3AED, #C084FC); color: #FFFFFF;
        font-family: 'Trebuchet MS', sans-serif; font-weight: 700; border: none;
        border-radius: 24px; width: 100%; padding: 0.7rem;
    }
    .stButton button:hover { background: linear-gradient(135deg, #8B5CF6, #D8B4FE); }
    label, .stMarkdown, p { color: #E9D5FF !important; }
    .stNumberInput label { color: #D8CFF0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="pso-eyebrow">✺ Wrapper Method &nbsp;·&nbsp; Swarm Intelligence Search</p>', unsafe_allow_html=True)
st.markdown('<h1 class="pso-title">🌀 PSO Forecast</h1>', unsafe_allow_html=True)
st.markdown('<p class="pso-desc">Eight features, converged on by a swarm of 10 particles evaluating subsets holistically across 15 search iterations.</p>', unsafe_allow_html=True)
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
pso_model = load_artifact("model_pso.pkl")
all_features = feature_info["all_features"]
pso_features = feature_info["pso_features"]
r = RESULTS["pso"]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="swarm-card"><div class="swarm-val">8 / 16</div><div class="swarm-key">Features Converged</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="swarm-card"><div class="swarm-val">{r["test_accuracy"]*100:.2f}%</div><div class="swarm-key">Test Accuracy</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="swarm-card"><div class="swarm-val">{r["auc_roc"]:.4f}</div><div class="swarm-key">AUC-ROC</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="swarm-card"><div class="swarm-val">~29 min</div><div class="swarm-key">Search Time</div></div>', unsafe_allow_html=True)

st.write("")
st.markdown("---")

st.markdown('<p class="pso-eyebrow">Particles Selected This Subset</p>', unsafe_allow_html=True)
pills_html = "".join(
    f'<span class="feature-pill"><span class="particle-dot"></span>{FEATURE_META[f]["label"]}</span>'
    for f in pso_features
)
st.markdown(pills_html, unsafe_allow_html=True)
st.write("")

left, right = st.columns([3, 2])

with left:
    st.markdown('<p class="pso-eyebrow">Swarm-Selected Inputs</p>', unsafe_allow_html=True)
    input_values = {}
    grid = st.columns(2)
    for i, feat in enumerate(pso_features):
        meta = FEATURE_META[feat]
        with grid[i % 2]:
            input_values[feat] = st.number_input(
                f"{meta['label']} ({meta['unit']})",
                min_value=float(meta["min"]), max_value=float(meta["max"]),
                value=float(meta["default"]), step=float(meta["step"]),
                key=f"pso_{feat}"
            )

    for feat in all_features:
        if feat not in input_values:
            input_values[feat] = FEATURE_META[feat]["default"]

with right:
    st.markdown('<p class="pso-eyebrow">Method Notes</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="swarm-card" style="margin-bottom:10px;">
        <div class="swarm-key">Fitness Function</div>
        <div class="swarm-val" style="font-size:1rem;">1 − F1-weighted (+ size penalty)</div>
    </div>
    <div class="swarm-card" style="margin-bottom:10px;">
        <div class="swarm-key">CV Accuracy</div>
        <div class="swarm-val" style="font-size:1rem;">{r['cv_accuracy']*100:.2f}% ± {r['cv_accuracy_std']*100:.2f}%</div>
    </div>
    <div class="swarm-card">
        <div class="swarm-key">Advantage Over MI</div>
        <div class="swarm-val" style="font-size:0.92rem; font-weight:400;">Evaluates subsets holistically — captures complementary feature interactions MI misses.</div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    run = st.button("✺ Run Swarm Forecast")

st.write("")

if run:
    pred, proba = predict(pso_model, scaler, all_features, pso_features, input_values)
    rain_prob = proba[1] * 100
    st.markdown("---")
    if pred == 1:
        st.markdown(f"""
        <div class="result-rain">
            <div style="font-family:'Trebuchet MS',sans-serif; color:#A5B4FC; font-size:0.8rem; letter-spacing:0.08em; text-transform:uppercase;">Swarm Consensus</div>
            <div style="font-family:'Trebuchet MS',sans-serif; color:#F5F3FF; font-size:2.1rem; font-weight:700; margin:8px 0;">🌧️ Rain Expected Tomorrow</div>
            <div style="font-family:'Trebuchet MS',sans-serif; color:#A5B4FC; font-size:1.05rem;">Model confidence: {rain_prob:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-dry">
            <div style="font-family:'Trebuchet MS',sans-serif; color:#E9D5FF; font-size:0.8rem; letter-spacing:0.08em; text-transform:uppercase;">Swarm Consensus</div>
            <div style="font-family:'Trebuchet MS',sans-serif; color:#F5F3FF; font-size:2.1rem; font-weight:700; margin:8px 0;">☀️ No Rain Expected Tomorrow</div>
            <div style="font-family:'Trebuchet MS',sans-serif; color:#E9D5FF; font-size:1.05rem;">Model confidence: {100-rain_prob:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
