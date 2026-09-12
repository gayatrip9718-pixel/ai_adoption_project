"""
02_eda.py
Exploratory data analysis on the AI adoption dataset.
Produces summary stats + saved charts in outputs/.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("/home/claude/ai_adoption_project/data/ai_adoption_dataset.csv")
OUT = "/home/claude/ai_adoption_project/outputs"

print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nNumeric summary:\n", df.describe())

# 1. Adoption rate trend over years (avg across all countries/industries)
plt.figure(figsize=(9, 5))
trend = df.groupby("year")["ai_adoption_rate"].mean()
sns.lineplot(x=trend.index, y=trend.values, marker="o", linewidth=2.5)
plt.title("Average Global AI Adoption Rate Over Time")
plt.xlabel("Year")
plt.ylabel("Avg AI Adoption Rate (%)")
plt.tight_layout()
plt.savefig(f"{OUT}/01_trend_over_years.png", dpi=150)
plt.close()

# 2. Top / bottom countries by average adoption
plt.figure(figsize=(9, 8))
country_avg = df.groupby("country")["ai_adoption_rate"].mean().sort_values()
colors = sns.color_palette("RdYlGn", len(country_avg))
sns.barplot(x=country_avg.values, y=country_avg.index, palette=colors)
plt.title("Average AI Adoption Rate by Country (2018-2025)")
plt.xlabel("Avg AI Adoption Rate (%)")
plt.tight_layout()
plt.savefig(f"{OUT}/02_adoption_by_country.png", dpi=150)
plt.close()

# 3. Adoption by industry
plt.figure(figsize=(9, 7))
industry_avg = df.groupby("industry")["ai_adoption_rate"].mean().sort_values()
sns.barplot(x=industry_avg.values, y=industry_avg.index, palette="viridis")
plt.title("Average AI Adoption Rate by Industry")
plt.xlabel("Avg AI Adoption Rate (%)")
plt.tight_layout()
plt.savefig(f"{OUT}/03_adoption_by_industry.png", dpi=150)
plt.close()

# 4. Adoption by company size over time
plt.figure(figsize=(9, 5))
size_year = df.groupby(["year", "company_size"])["ai_adoption_rate"].mean().reset_index()
sns.lineplot(data=size_year, x="year", y="ai_adoption_rate", hue="company_size",
             marker="o", linewidth=2.5, hue_order=["Small", "Medium", "Large"])
plt.title("AI Adoption Rate by Company Size Over Time")
plt.xlabel("Year")
plt.ylabel("Avg AI Adoption Rate (%)")
plt.tight_layout()
plt.savefig(f"{OUT}/04_adoption_by_company_size.png", dpi=150)
plt.close()

# 5. Correlation heatmap of numeric features
plt.figure(figsize=(7, 6))
numeric_cols = ["year", "gdp_tier", "num_employees", "rd_spend_pct_gdp",
                 "internet_penetration_pct", "digital_maturity_index", "ai_adoption_rate"]
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT}/05_correlation_heatmap.png", dpi=150)
plt.close()

# 6. Distribution of adoption levels (classification target)
plt.figure(figsize=(6, 5))
order = ["Low", "Medium", "High"]
sns.countplot(data=df, x="adoption_level", order=order, palette="Set2")
plt.title("Distribution of AI Adoption Level (classification target)")
plt.tight_layout()
plt.savefig(f"{OUT}/06_adoption_level_distribution.png", dpi=150)
plt.close()

print("\nEDA charts saved to:", OUT)
