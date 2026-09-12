"""
01_generate_data.py

Generates a realistic synthetic dataset of AI adoption rates across
countries, industries, company sizes, and years (2018-2025).

WHY SYNTHETIC:
No single clean, labeled, row-level public dataset tracks AI adoption
consistently across country x industry x company-size x year. Instead,
this generator is CALIBRATED to real published benchmarks so the
patterns in the data reflect reality, e.g.:

  - Kaggle 2020 ML & DS Survey: ML adoption ~45% overall; Israel 63%,
    Netherlands 57%, US 56% (highest); Egypt 31%, Morocco 24%,
    Nigeria 23% (lowest).
  - Adoption by company size: large companies ~61%, medium ~45%,
    small ~33% (same survey).
  - McKinsey Global AI Survey (2017-2025): overall enterprise AI
    adoption grew from roughly 20% (2017) to 50-72% (2024/2025),
    with acceleration after the 2022-2023 generative AI boom.
  - Stanford HAI AI Index: private AI investment and adoption is
    heavily concentrated in North America, Greater China, and Europe.

The generator encodes these as base rates + trend + noise, so the
resulting dataset is suitable for realistic regression/classification
modeling exercises, while being clearly documented as synthetic.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------
# 1. Reference tables (calibrated to real-world benchmarks)
# ---------------------------------------------------------------

# Base 2020-era adoption rate by country (grounded in Kaggle 2020 survey
# and extended with plausible values for additional countries based on
# their general digital-economy maturity tier).
country_base_rate = {
    "Israel": 0.63, "Netherlands": 0.57, "United States": 0.56,
    "United Kingdom": 0.54, "Germany": 0.52, "South Korea": 0.51,
    "Singapore": 0.55, "China": 0.50, "Canada": 0.50, "Japan": 0.48,
    "France": 0.47, "Australia": 0.46, "Sweden": 0.53, "India": 0.42,
    "Brazil": 0.35, "Mexico": 0.33, "South Africa": 0.30,
    "Egypt": 0.31, "Morocco": 0.24, "Nigeria": 0.23,
}

# GDP-per-capita tier drives R&D spend and digital infra proxies (rough,
# illustrative buckets - not exact World Bank figures).
country_gdp_tier = {
    "Israel": 3, "Netherlands": 3, "United States": 3, "United Kingdom": 3,
    "Germany": 3, "South Korea": 3, "Singapore": 3, "China": 2,
    "Canada": 3, "Japan": 3, "France": 3, "Australia": 3, "Sweden": 3,
    "India": 1, "Brazil": 2, "Mexico": 2, "South Africa": 1,
    "Egypt": 1, "Morocco": 1, "Nigeria": 1,
}

industries = [
    "Technology", "Healthcare", "Manufacturing", "Financial Services",
    "Retail", "Gaming", "Media & Entertainment", "Education",
    "Automotive", "Legal", "Government", "Agriculture",
]

# Relative industry adoption multipliers (Tech/Finance/Healthcare lead;
# Agriculture/Government/Legal lag) - grounded loosely in McKinsey /
# AI Index sector breakdowns.
industry_multiplier = {
    "Technology": 1.35, "Financial Services": 1.20, "Healthcare": 1.15,
    "Manufacturing": 1.05, "Retail": 1.00, "Gaming": 1.25,
    "Media & Entertainment": 1.10, "Automotive": 1.10, "Education": 0.85,
    "Legal": 0.80, "Government": 0.70, "Agriculture": 0.55,
}

company_sizes = ["Small", "Medium", "Large"]
# Kaggle 2020 survey: large 61%, medium 45%, small 33% -> normalize
# to multipliers relative to the "medium" baseline used above.
size_multiplier = {"Small": 0.73, "Medium": 1.00, "Large": 1.36}

years = list(range(2018, 2026))
# Year trend: adoption climbs over time, accelerating post-2022
# (generative AI boom), per Stanford HAI / McKinsey trend lines.
year_trend = {
    2018: -0.22, 2019: -0.16, 2020: -0.08, 2021: -0.02,
    2022: 0.04, 2023: 0.14, 2024: 0.24, 2025: 0.32,
}

rows = []
for country, base in country_base_rate.items():
    gdp_tier = country_gdp_tier[country]
    for industry in industries:
        for size in company_sizes:
            for year in years:
                # Core signal: base country rate * industry multiplier
                # * company size multiplier, shifted by year trend.
                raw = (
                    base
                    * industry_multiplier[industry]
                    * size_multiplier[size]
                    + year_trend[year]
                )
                noise = np.random.normal(0, 0.05)
                adoption_rate = np.clip(raw + noise, 0.02, 0.97)

                # Derived / auxiliary features useful for modeling
                rd_spend_pct_gdp = np.clip(
                    np.random.normal(0.5 + 0.7 * gdp_tier, 0.3), 0.1, 5.0
                )
                internet_penetration = np.clip(
                    np.random.normal(40 + 20 * gdp_tier, 8), 10, 99
                )
                digital_maturity_index = np.clip(
                    np.random.normal(30 + 20 * gdp_tier, 10)
                    + 15 * industry_multiplier[industry]
                    - 15,
                    5, 100
                )
                num_employees = {
                    "Small": np.random.randint(5, 100),
                    "Medium": np.random.randint(100, 1000),
                    "Large": np.random.randint(1000, 50000),
                }[size]

                rows.append({
                    "year": year,
                    "country": country,
                    "gdp_tier": gdp_tier,
                    "industry": industry,
                    "company_size": size,
                    "num_employees": num_employees,
                    "rd_spend_pct_gdp": round(rd_spend_pct_gdp, 2),
                    "internet_penetration_pct": round(internet_penetration, 1),
                    "digital_maturity_index": round(digital_maturity_index, 1),
                    "ai_adoption_rate": round(adoption_rate * 100, 2),  # target (%)
                })

df = pd.DataFrame(rows)

# Classification target derived from the continuous rate
def bucket(rate):
    if rate < 30:
        return "Low"
    elif rate < 55:
        return "Medium"
    else:
        return "High"

df["adoption_level"] = df["ai_adoption_rate"].apply(bucket)

# Shuffle rows (avoid any ordering artifacts) and save
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
out_path = "/home/claude/ai_adoption_project/data/ai_adoption_dataset.csv"
df.to_csv(out_path, index=False)

print(f"Generated {len(df):,} rows -> {out_path}")
print(df.head())
print("\nTarget distribution (adoption_level):")
print(df["adoption_level"].value_counts())
