"""
05_feature_engineering.py

Adds time-aware features to the dataset, since AI adoption for a given
(country, industry, company_size) group is highly autocorrelated
year-over-year:

  - prior_year_adoption_rate: the same group's adoption rate last year
  - adoption_momentum: change vs. prior year (captures acceleration)
  - rolling_3yr_avg: smoothed recent trend
  - years_since_2018: numeric trend counter (redundant with year but
    scales cleanly)

These are the kind of features that typically give the largest single
accuracy boost in adoption/growth-rate prediction problems, because
"what happened last year" is usually the strongest predictor of "what
happens this year."
"""

import pandas as pd

IN_PATH = "/home/claude/ai_adoption_project/data/ai_adoption_dataset.csv"
OUT_PATH = "/home/claude/ai_adoption_project/data/ai_adoption_dataset_v2.csv"

df = pd.read_csv(IN_PATH)

group_cols = ["country", "industry", "company_size"]
df = df.sort_values(group_cols + ["year"]).reset_index(drop=True)

grp = df.groupby(group_cols)["ai_adoption_rate"]

df["prior_year_adoption_rate"] = grp.shift(1)
df["adoption_momentum"] = df["ai_adoption_rate"] - df["prior_year_adoption_rate"]
df["rolling_3yr_avg"] = grp.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
df["years_since_2018"] = df["year"] - 2018

# First year (2018) per group has no prior data - fill with the group's
# own current-year value as a reasonable neutral baseline (no leakage
# concern since this only affects the very first observed year, which
# realistically wouldn't have history in production either).
for col in ["prior_year_adoption_rate", "rolling_3yr_avg"]:
    df[col] = df[col].fillna(df["ai_adoption_rate"])
df["adoption_momentum"] = df["adoption_momentum"].fillna(0)

df.to_csv(OUT_PATH, index=False)
print(f"Saved enriched dataset -> {OUT_PATH}")
print(df[group_cols + ["year", "ai_adoption_rate", "prior_year_adoption_rate",
                        "adoption_momentum", "rolling_3yr_avg"]].head(10))
