"""
03_train_model.py

Trains two predictive models on the AI adoption dataset:
  A) REGRESSION: predict the continuous ai_adoption_rate (%)
  B) CLASSIFICATION: predict the adoption_level bucket (Low/Medium/High)

Compares multiple algorithms, evaluates with a held-out test set,
reports feature importances, and saves the best model of each type.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report, confusion_matrix
)

DATA_PATH = "/home/claude/ai_adoption_project/data/ai_adoption_dataset.csv"
OUT = "/home/claude/ai_adoption_project/outputs"
MODELS = "/home/claude/ai_adoption_project/models"

df = pd.read_csv(DATA_PATH)

categorical_features = ["country", "industry", "company_size"]
numeric_features = ["year", "gdp_tier", "num_employees", "rd_spend_pct_gdp",
                     "internet_penetration_pct", "digital_maturity_index"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numeric_features),
])

X = df[categorical_features + numeric_features]

results = {}

# =================================================================
# A) REGRESSION: predict ai_adoption_rate
# =================================================================
y_reg = df["ai_adoption_rate"]
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

reg_models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
    "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.08, random_state=42),
}

reg_scores = {}
for name, model in reg_models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    cv_r2 = cross_val_score(pipe, X, y_reg, cv=5, scoring="r2").mean()
    reg_scores[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2_5fold": cv_r2}
    print(f"[REG] {name}: MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}  CV_R2={cv_r2:.3f}")

best_reg_name = max(reg_scores, key=lambda k: reg_scores[k]["R2"])
best_reg_pipe = Pipeline([("prep", preprocessor), ("model", reg_models[best_reg_name])])
best_reg_pipe.fit(X_train, y_train)
joblib.dump(best_reg_pipe, f"{MODELS}/best_regressor_{best_reg_name}.joblib")
print(f"\nBest regressor: {best_reg_name} -> saved.")

# Predicted vs actual plot
preds = best_reg_pipe.predict(X_test)
plt.figure(figsize=(6.5, 6))
plt.scatter(y_test, preds, alpha=0.3, s=15)
lims = [0, 100]
plt.plot(lims, lims, "r--", linewidth=1.5)
plt.xlabel("Actual AI Adoption Rate (%)")
plt.ylabel("Predicted AI Adoption Rate (%)")
plt.title(f"Predicted vs Actual — {best_reg_name} (R²={reg_scores[best_reg_name]['R2']:.3f})")
plt.tight_layout()
plt.savefig(f"{OUT}/07_regression_pred_vs_actual.png", dpi=150)
plt.close()

# Feature importance (tree-based only)
if hasattr(best_reg_pipe.named_steps["model"], "feature_importances_"):
    feat_names = (
        list(best_reg_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_features))
        + numeric_features
    )
    importances = best_reg_pipe.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)
    plt.figure(figsize=(8, 7))
    sns.barplot(data=imp_df, x="importance", y="feature", color="steelblue")
    plt.title(f"Top 15 Feature Importances — {best_reg_name} (Regression)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/08_regression_feature_importance.png", dpi=150)
    plt.close()

results["regression"] = reg_scores
results["best_regressor"] = best_reg_name

# =================================================================
# B) CLASSIFICATION: predict adoption_level (Low/Medium/High)
# =================================================================
y_clf = df["adoption_level"]
X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

clf_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
}

clf_scores = {}
best_clf_pipe = None
best_acc = -1
for name, model in clf_models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    clf_scores[name] = {"Accuracy": acc, "Macro_F1": f1}
    print(f"[CLF] {name}: Accuracy={acc:.3f}  Macro_F1={f1:.3f}")
    if acc > best_acc:
        best_acc = acc
        best_clf_name = name
        best_clf_pipe = pipe
        best_preds = preds

joblib.dump(best_clf_pipe, f"{MODELS}/best_classifier_{best_clf_name}.joblib")
print(f"\nBest classifier: {best_clf_name} -> saved.")

print("\nClassification report (best model):")
print(classification_report(y_test, best_preds))

# Confusion matrix
cm = confusion_matrix(y_test, best_preds, labels=["Low", "Medium", "High"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix — {best_clf_name}")
plt.tight_layout()
plt.savefig(f"{OUT}/09_confusion_matrix.png", dpi=150)
plt.close()

results["classification"] = clf_scores
results["best_classifier"] = best_clf_name

with open(f"{OUT}/model_results_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAll results saved to", OUT)
