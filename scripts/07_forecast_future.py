"""
07_forecast_future.py

Uses the tuned model to RECURSIVELY forecast AI adoption rates from
2026-2030 for every (country, industry, company_size) group.

Since the model relies heavily on prior_year_adoption_rate, forecasting
multiple years ahead requires a one-step-ahead recursive loop: predict
2026 using 2025's real data, then feed that 2026 prediction back in as
the "prior year" input to predict 2027, and so on. This is standard
practice for lag-feature-based forecasting models.

Auxiliary drivers (digital_maturity_index, rd_spend_pct_gdp,
internet_penetration_pct) are grown forward using each group's own
recent historical trend (average yearly change over 2022-2025),
capped at realistic bounds.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "/home/claude/ai_adoption_project/data/ai_adoption_dataset_v2.csv"
MODELS = "/home/claude/ai_adoption_project/models"
OUT = "/home/claude/ai_adoption_project/outputs"

df = pd.read_csv(DATA_PATH)
model = joblib.load(f"{MODELS}/best_regressor_v2_XGBoost.joblib")

group_cols = ["country", "industry", "company_size"]
FORECAST_YEARS = range(2026, 2031)

# Compute each group's recent (2022-2025) average yearly change for the
# slow-moving auxiliary drivers, to extrapolate them forward.
recent = df[df["year"].between(2022, 2025)].sort_values(group_cols + ["year"])
drift_cols = ["digital_maturity_index", "rd_spend_pct_gdp", "internet_penetration_pct"]
drift = (
    recent.groupby(group_cols)[drift_cols]
    .apply(lambda g: (g.iloc[-1] - g.iloc[0]) / max(len(g) - 1, 1))
    .rename(columns={c: f"{c}_drift" for c in drift_cols})
    .reset_index()
)

last_known = df[df["year"] == 2025].merge(drift, on=group_cols, how="left")

bounds = {
    "digital_maturity_index": (5, 100),
    "rd_spend_pct_gdp": (0.1, 6.0),
    "internet_penetration_pct": (10, 99.5),
}

forecast_rows = []
state = last_known.copy().set_index(group_cols)
state["_predicted_rate"] = state["ai_adoption_rate"]

# Track rolling history of adoption rates per group for the 3yr avg
history = {
    idx: list(df[(df["country"] == idx[0]) & (df["industry"] == idx[1]) &
                 (df["company_size"] == idx[2]) & (df["year"].between(2023, 2025))]
              .sort_values("year")["ai_adoption_rate"])
    for idx in state.index
}

for year in FORECAST_YEARS:
    batch = []
    idx_list = []
    for idx, row in state.iterrows():
        new_row = row.copy()
        new_row["country"], new_row["industry"], new_row["company_size"] = idx
        new_row["year"] = year
        new_row["years_since_2018"] = year - 2018
        for c in drift_cols:
            drift_val = row.get(f"{c}_drift", 0) or 0
            lo, hi = bounds[c]
            new_row[c] = float(np.clip(row[c] + drift_val, lo, hi))
        prior_rate = row["ai_adoption_rate"] if year == FORECAST_YEARS.start else row["_predicted_rate"]
        new_row["prior_year_adoption_rate"] = prior_rate
        hist = history[idx][-3:]
        new_row["rolling_3yr_avg"] = float(np.mean(hist)) if hist else prior_rate
        batch.append(new_row)
        idx_list.append(idx)

    batch_df = pd.DataFrame(batch)
    feature_cols = ["country", "industry", "company_size", "year", "years_since_2018",
                     "gdp_tier", "num_employees", "rd_spend_pct_gdp",
                     "internet_penetration_pct", "digital_maturity_index",
                     "prior_year_adoption_rate", "adoption_momentum", "rolling_3yr_avg"]
    preds = model.predict(batch_df[feature_cols])
    preds = np.clip(preds, 0.5, 99.5)

    for i, idx in enumerate(idx_list):
        rate = float(preds[i])
        row = state.loc[idx].copy()
        row["ai_adoption_rate"] = rate
        row["_predicted_rate"] = rate
        row["adoption_momentum"] = rate - batch_df.iloc[i]["prior_year_adoption_rate"]
        for c in drift_cols:
            row[c] = batch_df.iloc[i][c]
        state.loc[idx] = row
        history[idx].append(rate)

        forecast_rows.append({
            "country": idx[0], "industry": idx[1], "company_size": idx[2],
            "year": year, "predicted_adoption_rate": round(rate, 2),
        })

forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_csv("/home/claude/ai_adoption_project/data/forecast_2026_2030.csv", index=False)
print(f"Saved forecast -> data/forecast_2026_2030.csv ({len(forecast_df)} rows)")

# --- Combine historical + forecast for a global trend chart ---
hist_avg = df.groupby("year")["ai_adoption_rate"].mean().reset_index()
hist_avg["type"] = "Historical"
fut_avg = forecast_df.groupby("year")["predicted_adoption_rate"].mean().reset_index()
fut_avg = fut_avg.rename(columns={"predicted_adoption_rate": "ai_adoption_rate"})
fut_avg["type"] = "Forecast"
combined = pd.concat([hist_avg, fut_avg], ignore_index=True)

plt.figure(figsize=(10, 5.5))
for t, sub in combined.groupby("type"):
    style = "-o" if t == "Historical" else "--o"
    plt.plot(sub["year"], sub["ai_adoption_rate"], style, label=t, linewidth=2.5)
plt.axvline(2025.5, color="gray", linestyle=":", linewidth=1)
plt.title("Global Average AI Adoption Rate: Historical (2018-2025) + Forecast (2026-2030)")
plt.xlabel("Year")
plt.ylabel("Avg AI Adoption Rate (%)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/12_forecast_global_trend.png", dpi=150)
plt.close()

# --- Per-country forecast for a handful of interesting countries ---
sample_countries = ["United States", "Israel", "India", "Nigeria", "Germany", "China"]
hist_c = df[df["country"].isin(sample_countries)].groupby(["year", "country"])["ai_adoption_rate"].mean().reset_index()
fut_c = forecast_df[forecast_df["country"].isin(sample_countries)].groupby(["year", "country"])["predicted_adoption_rate"].mean().reset_index()
fut_c = fut_c.rename(columns={"predicted_adoption_rate": "ai_adoption_rate"})

plt.figure(figsize=(10, 6))
palette = sns.color_palette("tab10", len(sample_countries))
for i, c in enumerate(sample_countries):
    h = hist_c[hist_c["country"] == c]
    f = fut_c[fut_c["country"] == c]
    plt.plot(h["year"], h["ai_adoption_rate"], "-o", color=palette[i], label=c)
    plt.plot(f["year"], f["ai_adoption_rate"], "--o", color=palette[i])
plt.axvline(2025.5, color="gray", linestyle=":", linewidth=1)
plt.title("AI Adoption Forecast by Country (solid=historical, dashed=forecast)")
plt.xlabel("Year")
plt.ylabel("Avg AI Adoption Rate (%)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/13_forecast_by_country.png", dpi=150)
plt.close()

print("\nForecast summary (2030 avg by sample country):")
print(fut_c[fut_c["year"] == 2030][["country", "ai_adoption_rate"]].to_string(index=False))
print("\nCharts saved to", OUT)
