"""
06b_finalize_tuning.py

Reads the per-model tuning results, selects the overall best model,
runs a strict TIME-BASED holdout check (train on 2018-2023, test on
2024-2025 only — no random shuffling) to confirm the model isn't just
exploiting random-split leakage, and produces final plots.
"""

import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone

DATA_PATH = "/home/claude/ai_adoption_project/data/ai_adoption_dataset_v2.csv"
OUT = "/home/claude/ai_adoption_project/outputs"
MODELS = "/home/claude/ai_adoption_project/models"

with open(f"{OUT}/tuning_results_summary.json") as f:
    results = json.load(f)

model_names = ["GradientBoosting", "RandomForest", "XGBoost"]
best_name = max(model_names, key=lambda n: results[n]["test_R2"])
best_pipe = joblib.load(f"{MODELS}/tuned_{best_name}.joblib")
print(f"Overall best tuned model: {best_name} (test R2={results[best_name]['test_R2']:.4f})")

joblib.dump(best_pipe, f"{MODELS}/best_regressor_v2_{best_name}.joblib")

df = pd.read_csv(DATA_PATH)
categorical_features = ["country", "industry", "company_size"]
numeric_features = [
    "year", "years_since_2018", "gdp_tier", "num_employees", "rd_spend_pct_gdp",
    "internet_penetration_pct", "digital_maturity_index",
    "prior_year_adoption_rate", "adoption_momentum", "rolling_3yr_avg",
]
X = df[categorical_features + numeric_features]
y = df["ai_adoption_rate"]

# --- Strict time-based holdout: train <=2023, test >=2024 ---
train_mask = df["year"] <= 2023
test_mask = df["year"] >= 2024
X_tr, y_tr = X[train_mask], y[train_mask]
X_te, y_te = X[test_mask], y[test_mask]

time_pipe = clone(best_pipe)
time_pipe.fit(X_tr, y_tr)
time_preds = time_pipe.predict(X_te)
time_r2 = r2_score(y_te, time_preds)
time_mae = mean_absolute_error(y_te, time_preds)
print(f"Time-based holdout (train<=2023, test>=2024): R2={time_r2:.4f}  MAE={time_mae:.2f}")
results["time_based_holdout"] = {"model": best_name, "R2": time_r2, "MAE": time_mae}
results["overall_best_model"] = best_name

with open(f"{OUT}/tuning_results_summary.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# --- Plots on the original random test split (re-derive it) ---
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
preds_best = best_pipe.predict(X_test)

plt.figure(figsize=(6.5, 6))
plt.scatter(y_test, preds_best, alpha=0.3, s=15)
plt.plot([0, 100], [0, 100], "r--", linewidth=1.5)
plt.xlabel("Actual AI Adoption Rate (%)")
plt.ylabel("Predicted AI Adoption Rate (%)")
plt.title(f"Tuned {best_name} — Predicted vs Actual (R²={results[best_name]['test_R2']:.4f})")
plt.tight_layout()
plt.savefig(f"{OUT}/10_tuned_pred_vs_actual.png", dpi=150)
plt.close()

model_step = best_pipe.named_steps["model"]
if hasattr(model_step, "feature_importances_"):
    feat_names = (
        list(best_pipe.named_steps["prep"].named_transformers_["cat"]
             .get_feature_names_out(categorical_features))
        + numeric_features
    )
    imp_df = pd.DataFrame({
        "feature": feat_names,
        "importance": model_step.feature_importances_,
    }).sort_values("importance", ascending=False).head(15)
    plt.figure(figsize=(8, 7))
    sns.barplot(data=imp_df, x="importance", y="feature", color="teal")
    plt.title(f"Top 15 Feature Importances — Tuned {best_name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/11_tuned_feature_importance.png", dpi=150)
    plt.close()
    print("\nTop 5 features:")
    print(imp_df.head())

# Comparison chart: baseline (v1) vs tuned v2
print("\n=== Summary: baseline vs tuned ===")
for n in model_names:
    print(f"{n}: test R2={results[n]['test_R2']:.4f}  MAE={results[n]['test_MAE']:.2f}")
