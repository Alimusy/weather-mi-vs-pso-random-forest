# Mutual Information vs PSO Feature Selection on Random Forest

Rainfall prediction on the Australian weather dataset, used to compare a
filter-based feature selector (Mutual Information) against a wrapper-based one
(Particle Swarm Optimization) on the same Random Forest classifier.

## Open it

Feature selection and comparison:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alimusy/weather-mi-vs-pso-random-forest/blob/main/notebooks/weather_forecast_mi_vs_pso.ipynb)

## Method

1. 16 engineered features, 100,000+ observations, 77/23 class imbalance.
2. k-tuning across k in {8, 10, 12, 13, 14, 15, 16}; k=8 kept as a deliberate
   accuracy vs efficiency trade-off (50% fewer features).
3. SMOTE applied inside each fold of a stratified 5-fold CV, never before the
   split, so no leakage.
4. Random Forest tuned with GridSearchCV.

## Results

| Model | Features | Accuracy | AUC-ROC | Selection time |
|---|---|---|---|---|
| RF, all features | 16 | 84.45% | 87.78% | n/a |
| MI + RF | 8 | 82.30% | | 24.76 s |
| PSO + RF | 8 | 83.70% | | 1,720.75 s |

PSO beats MI by 1.40 points and picks a more meteorologically varied subset
(Rainfall, Evaporation, several wind-speed measures) where MI leans heavily on
correlated temperature features. MI is roughly 70x faster.

## Repo layout

```
notebooks/   training, feature selection and comparison
app/         Streamlit app, three pages: baseline, MI, PSO
```

## Running the app

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The three Random Forest pickles are ~2.8 GB in total and are not committed.
Re-train them with the notebook, then drop `model_baseline.pkl`, `model_mi.pkl`,
`model_pso.pkl`, `scaler.pkl` and `feature_info.json` into `app/models/`.
