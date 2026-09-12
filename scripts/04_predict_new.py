"""
04_predict_new.py

Loads the saved best models and predicts AI adoption for new,
hypothetical scenarios. Edit the `new_data` dict/rows below to try
your own scenarios.
"""

import glob
import joblib
import pandas as pd

MODELS = "/home/claude/ai_adoption_project/models"

reg_path = glob.glob(f"{MODELS}/best_regressor_*.joblib")[0]
clf_path = glob.glob(f"{MODELS}/best_classifier_*.joblib")[0]

reg_model = joblib.load(reg_path)
clf_model = joblib.load(clf_path)

# Example new scenarios to predict on
new_data = pd.DataFrame([
    {
        "country": "India", "industry": "Technology", "company_size": "Large",
        "year": 2026, "gdp_tier": 1, "num_employees": 20000,
        "rd_spend_pct_gdp": 1.2, "internet_penetration_pct": 55,
        "digital_maturity_index": 60,
    },
    {
        "country": "Nigeria", "industry": "Agriculture", "company_size": "Small",
        "year": 2026, "gdp_tier": 1, "num_employees": 30,
        "rd_spend_pct_gdp": 0.4, "internet_penetration_pct": 45,
        "digital_maturity_index": 25,
    },
    {
        "country": "Germany", "industry": "Manufacturing", "company_size": "Large",
        "year": 2026, "gdp_tier": 3, "num_employees": 15000,
        "rd_spend_pct_gdp": 3.1, "internet_penetration_pct": 92,
        "digital_maturity_index": 88,
    },
])

pred_rate = reg_model.predict(new_data)
pred_level = clf_model.predict(new_data)

new_data["predicted_adoption_rate_pct"] = pred_rate.round(1)
new_data["predicted_adoption_level"] = pred_level

print(new_data[["country", "industry", "company_size", "year",
                 "predicted_adoption_rate_pct", "predicted_adoption_level"]]
      .to_string(index=False))
