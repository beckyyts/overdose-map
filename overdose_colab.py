## ============================================================
##  OVERDOSE MAP - Google Colab Data Processing Script
##  Paste this entire script into a Colab cell and run it.
## ============================================================

# Step 1: Upload your files
from google.colab import files
print("Please upload your two CSV files:")
uploaded = files.upload()
# → Upload: 2022_county_overdose_analysis_final.csv
# → Upload: 2024_county_overdose_analysis_final.csv

# ─────────────────────────────────────────────
# Step 2: Load and merge data
# ─────────────────────────────────────────────
import pandas as pd
import json

df22 = pd.read_csv("2022_county_overdose_analysis_final.csv", dtype={"county_fips": str})
df24 = pd.read_csv("2024_county_overdose_analysis_final.csv", dtype={"county_fips": str})

# Rename overlapping columns before merge
df22 = df22.rename(columns={
    "overdose_deaths_2022": "overdose_deaths_2022",
    "overdose_rate_per_100k": "overdose_rate_2022",
    "n_months_with_data": "n_months_2022",
    "any_suppressed": "suppressed_2022"
})
df24 = df24.rename(columns={
    "overdose_deaths_2024": "overdose_deaths_2024",
    "overdose_rate_per_100k": "overdose_rate_2024",
    "n_months_with_data": "n_months_2024",
    "any_suppressed": "suppressed_2024"
})

# Socioeconomic columns (use 2024 as primary, more recent)
socio_cols = [
    "county_fips", "county_name",
    "total_pop", "median_income", "unemployment_rate",
    "poverty_rate", "bachelor_plus_pct", "uninsured",
    "male", "female",
    "age_20to24", "age_25to34", "age_35to44", "age_45to54",
    "white", "black", "native", "asian", "hispanic", "two_more",
]

df_socio = df24[socio_cols].copy()

# Overdose data from each year
df_od22 = df22[["county_fips", "overdose_deaths_2022", "overdose_rate_2022", "suppressed_2022"]]
df_od24 = df24[["county_fips", "overdose_deaths_2024", "overdose_rate_2024", "suppressed_2024"]]

# Merge everything
df = df_socio.merge(df_od22, on="county_fips", how="left")
df = df.merge(df_od24, on="county_fips", how="left")

# ─────────────────────────────────────────────
# Step 3: Calculate rate change
# ─────────────────────────────────────────────
df["rate_change"] = df["overdose_rate_2024"] - df["overdose_rate_2022"]
df["rate_change_pct"] = ((df["overdose_rate_2024"] - df["overdose_rate_2022"]) / df["overdose_rate_2022"] * 100).round(1)

# Extract state FIPS (first 2 digits of county_fips)
df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
df["state_fips"] = df["county_fips"].str[:2]

print(f"Total counties: {len(df)}")
print(f"Counties with both years data: {df[['overdose_rate_2022','overdose_rate_2024']].dropna().shape[0]}")
print(f"Rate change range: {df['rate_change'].min():.1f} to {df['rate_change'].max():.1f}")

# ─────────────────────────────────────────────
# Step 4: Export to JSON for D3
# ─────────────────────────────────────────────
def safe_val(v):
    """Convert NaN/NA to None for JSON"""
    if pd.isna(v):
        return None
    if isinstance(v, (float,)):
        return round(float(v), 2)
    return v

records = []
for _, row in df.iterrows():
    records.append({
        "fips":             row["county_fips"],
        "state_fips":       row["state_fips"],
        "name":             row["county_name"],
        "pop":              safe_val(row["total_pop"]),
        # Socioeconomic
        "median_income":    safe_val(row["median_income"]),
        "unemployment":     safe_val(row["unemployment_rate"]),
        "poverty":          safe_val(row["poverty_rate"]),
        "bachelors":        safe_val(row["bachelor_plus_pct"]),
        "uninsured":        safe_val(row["uninsured"]),
        # Race (%)
        "white":            safe_val(row["white"]),
        "black":            safe_val(row["black"]),
        "hispanic":         safe_val(row["hispanic"]),
        "native":           safe_val(row["native"]),
        "asian":            safe_val(row["asian"]),
        # Age groups (%)
        "age_20to24":       safe_val(row["age_20to24"]),
        "age_25to34":       safe_val(row["age_25to34"]),
        "age_35to44":       safe_val(row["age_35to44"]),
        "age_45to54":       safe_val(row["age_45to54"]),
        # Overdose
        "rate_2022":        safe_val(row["overdose_rate_2022"]),
        "rate_2024":        safe_val(row["overdose_rate_2024"]),
        "deaths_2022":      safe_val(row["overdose_deaths_2022"]),
        "deaths_2024":      safe_val(row["overdose_deaths_2024"]),
        "rate_change":      safe_val(row["rate_change"]),
        "rate_change_pct":  safe_val(row["rate_change_pct"]),
        "suppressed_2022":  bool(row["suppressed_2022"]) if pd.notna(row["suppressed_2022"]) else True,
        "suppressed_2024":  bool(row["suppressed_2024"]) if pd.notna(row["suppressed_2024"]) else True,
    })

with open("counties.json", "w") as f:
    json.dump(records, f)

print(f"\n✅ Saved counties.json with {len(records)} counties")
print("Now download it below ↓")

# ─────────────────────────────────────────────
# Step 5: Download the file
# ─────────────────────────────────────────────
files.download("counties.json")
