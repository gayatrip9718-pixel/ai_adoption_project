"""
06_hyperparameter_tuning.py

Retrains the regression model on the ENRICHED dataset (with lag/momentum
features) and runs a randomized hyperparameter search for ONE model at a
time (pass "GradientBoosting", "RandomForest", or "XGBoost" as argv[1]).
Results and fitted pipelines are saved incrementally so progress isn't
lost between runs. Run scripts/06b_finalize_tuning.py after all three
have been run to pick the overall winner and produce final plots.
"""

import sys
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_PATH = "/home/claude/ai_adoption_project/data/ai_adoption_dataset_v2.csv"
OUT = "/home/claude/ai_adoption_project/outputs"
MODELS = "/home/claude/ai_adoption_project/models"

df = pd.read_csv(DATA_PATH)

categorical_features = ["country", "industry", "company_size"]
numeric_features = [
    "year", "years_since_2018", "gdp_tier", "num_employees", "rd_spend_pct_gdp",
    "internet_penetration_pct", "digital_maturity_index",
    "prior_year_adoption_rate", "adoption_momentum", "rolling_3yr_avg",
]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numeric_features),
])

X = df[categorical_features + numeric_features]
y = df["ai_adoption_rate"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

search_spaces = {
    "GradientBoosting": (
        GradientBoostingRegressor(random_state=42),
        {
            "model__n_estimators": [150, 250],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.85, 1.0],
        },
    ),
    "RandomForest": (
        RandomForestRegressor(random_state=42, n_jobs=2),
        {
            "model__n_estimators": [150, 250],
            "model__max_depth": [10, 16],
            "model__min_samples_leaf": [1, 2],
        },
    ),
    "XGBoost": (
        XGBRegressor(random_state=42, n_jobs=2, objective="reg:squarederror"),
        {
            "model__n_estimators": [150, 250],
            "model__max_depth": [3, 4, 5],
            "model__learning_rate": [0.05, 0.1],
        },
    ),
}

name = sys.argv[1]
estimator, param_dist = search_spaces[name]

results_path = f"{OUT}/tuning_results_summary.json"
results = {}
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)

pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
search = RandomizedSearchCV(
    pipe, param_distributions=param_dist, n_iter=6, cv=3,
    scoring="r2", random_state=42, n_jobs=4,
)
search.fit(X_train, y_train)
best_pipe = search.best_estimator_
preds = best_pipe.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds) ** 0.5
r2 = r2_score(y_test, preds)

results[name] = {
    "best_params": search.best_params_,
    "cv_best_r2": search.best_score_,
    "test_MAE": mae,
    "test_RMSE": rmse,
    "test_R2": r2,
}
print(f"[{name}] best CV R2={search.best_score_:.4f} | test R2={r2:.4f} MAE={mae:.2f} RMSE={rmse:.2f}")
print(f"   best params: {search.best_params_}")

joblib.dump(best_pipe, f"{MODELS}/tuned_{name}.joblib")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"Saved model -> {MODELS}/tuned_{name}.joblib and updated {results_path}")
