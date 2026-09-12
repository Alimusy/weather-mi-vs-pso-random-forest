"""
Shared utilities for the Rain Prediction multi-model app.
Loads trained models, scaler, and feature metadata produced by the
KWASU FYP notebook (MI vs PSO vs Baseline Random Forest).
"""
import json
import os
import joblib
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Real metric results from the actual experiment (final_results.csv)
RESULTS = {
    "baseline": {
        "label": "Baseline", "n_features": 16,
        "cv_accuracy": 0.8408, "cv_accuracy_std": 0.0022,
        "cv_f1": 0.8402, "cv_f1_std": 0.0023,
        "test_accuracy": 0.8445, "test_precision": 0.8435,
        "test_recall": 0.8445, "test_f1": 0.8440, "auc_roc": 0.8778,
    },
    "mi": {
        "label": "Mutual Information", "n_features": 8,
        "cv_accuracy": 0.8217, "cv_accuracy_std": 0.0012,
        "cv_f1": 0.8220, "cv_f1_std": 0.0011,
        "test_accuracy": 0.8230, "test_precision": 0.8251,
        "test_recall": 0.8230, "test_f1": 0.8240, "auc_roc": 0.8493,
    },
    "pso": {
        "label": "Particle Swarm Optimization", "n_features": 8,
        "cv_accuracy": 0.8347, "cv_accuracy_std": 0.0013,
        "cv_f1": 0.8326, "cv_f1_std": 0.0011,
        "test_accuracy": 0.8370, "test_precision": 0.8338,
        "test_recall": 0.8370, "test_f1": 0.8352, "auc_roc": 0.8577,
    },
}

# Realistic value ranges for the Australian Weather Dataset, used for
# input widget bounds + sensible defaults (medians).
FEATURE_META = {
    "MinTemp":       dict(label="Min Temperature",        unit="°C",   min=-8.0,  max=34.0,  default=12.0,  step=0.5),
    "MaxTemp":       dict(label="Max Temperature",        unit="°C",   min=-4.8,  max=48.1,  default=23.0,  step=0.5),
    "Rainfall":      dict(label="Rainfall (today)",       unit="mm",   min=0.0,   max=150.0, default=0.0,   step=0.2),
    "Evaporation":   dict(label="Evaporation",             unit="mm",   min=0.0,   max=80.0,  default=5.0,   step=0.2),
    "Sunshine":      dict(label="Sunshine",                unit="hrs",  min=0.0,   max=14.5,  default=7.5,   step=0.1),
    "WindGustSpeed": dict(label="Wind Gust Speed",         unit="km/h", min=6.0,   max=135.0, default=39.0,  step=1.0),
    "WindSpeed9am":  dict(label="Wind Speed (9am)",        unit="km/h", min=0.0,   max=87.0,  default=14.0,  step=1.0),
    "WindSpeed3pm":  dict(label="Wind Speed (3pm)",        unit="km/h", min=0.0,   max=87.0,  default=19.0,  step=1.0),
    "Humidity9am":   dict(label="Humidity (9am)",          unit="%",    min=0.0,   max=100.0, default=69.0,  step=1.0),
    "Humidity3pm":   dict(label="Humidity (3pm)",          unit="%",    min=0.0,   max=100.0, default=51.0,  step=1.0),
    "Pressure9am":   dict(label="Pressure (9am)",          unit="hPa",  min=980.0, max=1041.0, default=1017.6, step=0.5),
    "Pressure3pm":   dict(label="Pressure (3pm)",          unit="hPa",  min=977.0, max=1040.0, default=1015.2, step=0.5),
    "Cloud9am":      dict(label="Cloud Cover (9am)",       unit="oktas",min=0.0,   max=9.0,   default=4.0,   step=1.0),
    "Cloud3pm":      dict(label="Cloud Cover (3pm)",       unit="oktas",min=0.0,   max=9.0,   default=4.0,   step=1.0),
    "Temp9am":       dict(label="Temperature (9am)",       unit="°C",   min=-7.0,  max=40.0,  default=16.5,  step=0.5),
    "Temp3pm":       dict(label="Temperature (3pm)",       unit="°C",   min=-5.4,  max=46.7,  default=21.5,  step=0.5),
}


def load_feature_info():
    path = os.path.join(MODELS_DIR, "feature_info.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_artifact(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def models_available():
    """Returns True only if all 4 required files are present."""
    required = ["model_mi.pkl", "model_pso.pkl", "scaler.pkl", "feature_info.json"]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)


def predict(model, scaler, all_features, selected_features, input_values):
    """
    input_values: dict of {feature_name: value} for ALL 16 base features
    (scaler was fit on all 16 columns). We scale the full vector, then
    slice down to the selected subset before calling the model.
    """
    full_vector = np.array([[input_values[f] for f in all_features]])
    scaled = scaler.transform(full_vector)

    selected_idx = [all_features.index(f) for f in selected_features]
    scaled_subset = scaled[:, selected_idx]

    pred = model.predict(scaled_subset)[0]
    proba = model.predict_proba(scaled_subset)[0]
    return pred, proba
