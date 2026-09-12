# Predicting Global AI Adoption Trends

A predictive modeling project that estimates AI adoption rates across
**countries, industries, and company sizes** from 2018–2025, and
predicts adoption trajectory for future scenarios (e.g., 2026).

## Project Structure
```
ai_adoption_project/
├── data/
│   └── ai_adoption_dataset.csv       # generated dataset (5,760 rows)
├── scripts/
│   ├── 01_generate_data.py           # builds the calibrated dataset
│   ├── 02_eda.py                     # exploratory analysis + charts
│   ├── 03_train_model.py             # trains & evaluates baseline models
│   ├── 04_predict_new.py             # run predictions on new scenarios
│   ├── 05_feature_engineering.py     # adds lag/momentum features -> v2 dataset
│   ├── 06_hyperparameter_tuning.py   # tunes one model at a time (pass model name as arg)
│   ├── 06b_finalize_tuning.py        # picks best tuned model, time-holdout check, plots
│   └── 07_forecast_future.py         # recursive 2026-2030 forecast
├── models/
│   ├── best_regressor_GradientBoostingRegressor.joblib   # v1 baseline
│   ├── best_classifier_LogisticRegression.joblib         # v1 baseline
│   ├── tuned_{GradientBoosting,RandomForest,XGBoost}.joblib
│   └── best_regressor_v2_XGBoost.joblib                  # v2 final (with lag features)
├── outputs/                          # all charts + results JSON
└── README.md
```

## About the Data
No single clean public dataset tracks AI adoption row-by-row across
country × industry × company-size × year, so this project uses a
**synthetic dataset calibrated to real, published benchmarks**:

- **Kaggle 2020 Machine Learning & Data Science Survey** — country-level
  ML adoption (Israel 63%, Netherlands 57%, US 56% highest; Nigeria 23%,
  Morocco 24%, Egypt 31% lowest) and company-size adoption gaps
  (large 61%, medium 45%, small 33%).
- **McKinsey Global AI Survey (2017–2025)** — the overall adoption
  growth trend, including the acceleration after the 2022–2023
  generative AI boom.
- **Stanford HAI AI Index** — the general pattern that AI investment
  and adoption concentrate in North America, Europe, and Greater China.

These benchmarks were used as **base rates and multipliers**, combined
with random noise, to produce realistic row-level data. It is not raw
observed data — it's a research-grade proxy suitable for learning and
demonstrating an end-to-end ML pipeline. This is documented directly
in `scripts/01_generate_data.py`.

**Features:** year, country, industry, company size, number of
employees, R&D spend (% of GDP), internet penetration (%), and a
digital maturity index.
**Targets:** `ai_adoption_rate` (continuous %) and `adoption_level`
(Low / Medium / High, derived from the rate).

## Key EDA Findings
- Global average AI adoption climbed steadily from 2018 to 2025, with
  a visible acceleration after 2022 (the generative AI inflection point).
- **Israel, Netherlands, and the US** lead in adoption; **Nigeria,
  Morocco, and Egypt** trail — consistent with the source survey data.
- **Technology, Gaming, and Financial Services** are the highest-adopting
  industries; **Agriculture and Government** lag.
- **Large companies adopt AI at roughly double the rate of small
  companies**, and that gap persists across all years.
- `digital_maturity_index`, `internet_penetration_pct`, and
  `rd_spend_pct_gdp` are all positively correlated with adoption rate.

## Modeling Results

### Regression (predicting continuous adoption rate %)
| Model | MAE | RMSE | R² | 5-fold CV R² |
|---|---|---|---|---|
| Linear Regression | 5.52 | 7.00 | 0.929 | 0.927 |
| Random Forest | 5.99 | 7.62 | 0.915 | 0.910 |
| **Gradient Boosting (best)** | **4.39** | **5.55** | **0.955** | **0.950** |

### Classification (predicting Low / Medium / High adoption level)
| Model | Accuracy | Macro F1 |
|---|---|---|
| **Logistic Regression (best)** | **86.3%** | **0.859** |
| Random Forest | 81.4% | 0.810 |

The **year** feature carries the most predictive weight (capturing the
adoption growth trend over time), followed by **company size**,
**digital maturity index**, and **GDP tier** — all consistent with the
patterns visible in the EDA.

## How to Use
```bash
# 1. Regenerate the dataset (optional — already generated)
python3 scripts/01_generate_data.py

# 2. Run EDA (produces charts in outputs/)
python3 scripts/02_eda.py

# 3. Train and evaluate models (saves best models to models/)
python3 scripts/03_train_model.py

# 4. Predict on new/hypothetical scenarios
python3 scripts/04_predict_new.py
```
To predict your own scenario, edit the `new_data` DataFrame at the top
of `04_predict_new.py` with your own country/industry/company size/year
combination.

## Known Limitations
- Data is synthetic (calibrated, not observed) — treat outputs as
  illustrative rather than authoritative forecasts.
- The regression and classification models are trained independently,
  so their predictions can occasionally disagree near bucket boundaries
  (e.g., a ~46% predicted rate is technically "Medium" but the
  classifier may independently say "Low"). A production system would
  typically derive the bucket directly from the regression output, or
  ensemble the two.
- Real-world AI adoption data would also need to account for
  regulatory environment, workforce skill availability, and
  company-specific strategy — none of which are modeled here.

## v2 Improvements: Feature Engineering, Tuning & Forecasting

Building on the baseline models above, three additional scripts push
the project further:

### 1. Lag/momentum features (`05_feature_engineering.py`)
Added `prior_year_adoption_rate`, `adoption_momentum` (YoY change), and
`rolling_3yr_avg` per (country, industry, company_size) group — since
adoption is highly autocorrelated year-over-year, these are usually the
single biggest lever in a growth-rate prediction problem.

### 2. Hyperparameter tuning (`06_hyperparameter_tuning.py` + `06b_finalize_tuning.py`)
Ran `RandomizedSearchCV` (3-fold CV) across Gradient Boosting, Random
Forest, and XGBoost:

| Model | Test R² | Test MAE |
|---|---|---|
| Gradient Boosting (tuned) | 0.9994 | 0.44 |
| Random Forest (tuned) | 0.9993 | 0.35 |
| **XGBoost (tuned, best)** | **0.9995** | **0.37** |

**Time-based holdout check** (train only on 2018–2023, test on
2024–2025, no shuffling): **R² = 0.997, MAE = 0.61** — confirming the
model generalizes forward in time and the near-perfect scores aren't
just random-split leakage.

Feature importance is now dominated by the lag features:
`prior_year_adoption_rate` (~79%), `rolling_3yr_avg` (~16%),
`adoption_momentum` (~5%) — everything else (country, industry, size)
contributes only a small residual, which makes sense: last year's
adoption rate is overwhelmingly the best predictor of this year's.

### 3. Forecasting 2026–2030 (`07_forecast_future.py`)
Since the model depends on lag features, forecasting multiple years
ahead requires a **recursive one-step-ahead loop**: predict 2026 from
real 2025 data, feed that prediction back in as the "prior year" input
to predict 2027, and so on through 2030. Slow-moving drivers (digital
maturity, R&D spend, internet penetration) are extrapolated using each
group's own 2022–2025 trend, capped at realistic bounds.

Output: `data/forecast_2026_2030.csv` (3,600 rows) plus two charts:
- `outputs/12_forecast_global_trend.png` — global average, historical + forecast
- `outputs/13_forecast_by_country.png` — per-country trajectories (US, Israel, Germany, China, India, Nigeria)

**Important limitation:** the forecast shows adoption gaps between
countries *narrowing* over 2026–2030 (e.g., Nigeria and India catching
up toward ~85-90%). This is a direct consequence of how the underlying
synthetic data was generated — country/industry gaps are modeled as
*multiplicative* factors on top of a *shared additive* year-over-year
trend, so a fixed absolute yearly gain is proportionally larger for
lower-base countries, causing convergence. Real-world AI adoption gaps
may persist far longer due to infrastructure, capital, regulatory, and
skills constraints not captured here. Treat the forecast as a
demonstration of the recursive forecasting *technique*, not a literal
prediction — this is exactly the kind of assumption you'd need to
revisit with a real dataset.

## Possible Extensions
- Swap in a real dataset (e.g., Stanford AI Index public data tables,
  or a Kaggle CSV you download yourself) using the same schema.
- Add time-series forecasting (e.g., Prophet/ARIMA) to project 2026–2030
  adoption rates per country.
- Build a small Streamlit/Gradio app around `04_predict_new.py` for an
  interactive "what-if" adoption calculator.
