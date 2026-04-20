"""
把 Model A3 (Final Vulnerability Model) 的 residual、fitted value、outlier flag
加進 counties_with_residuals.json

使用方式：
  python3 add_residuals.py \
    --csv2022 data/2022_county_overdose_analysis_final.csv \
    --csv2024 data/2024_county_overdose_analysis_final.csv \
    --input   data/counties_with_residuals.json \
    --output  data/counties_with_residuals.json
"""

import argparse
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ── 0. CLI 參數 ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv2022", required=True)
parser.add_argument("--csv2024", required=True)
parser.add_argument("--input",   required=True)
parser.add_argument("--output",  required=True)
parser.add_argument("--outlier_n", type=int, default=20)
parser.add_argument("--min_pop",   type=int, default=10000)
args = parser.parse_args()

# ── 1. 載入資料 ───────────────────────────────────────────────────────────────
df_2022 = pd.read_csv(args.csv2022)
df_2024 = pd.read_csv(args.csv2024)

df = df_2022.merge(df_2024, on="county_fips", suffixes=("_2022", "_2024"))
df["rate_change"] = df["overdose_rate_per_100k_2024"] - df["overdose_rate_per_100k_2022"]
df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

print(f"合併後總縣數：{len(df)}")

# ── 2. 清理資料 ───────────────────────────────────────────────────────────────
# A3 Final variables（直接 hardcode，對齊 report）
VARS_A3 = [
    "poverty_rate_2022",
    "uninsured_2022",
    "other_2022",
    "two_more_2022",
    "age_25to34_2022",
    "age_45to54_2022",
]

# 移除任何一期有 suppressed data 的縣（對齊 report 的 panel）
df = df[
    (df["any_suppressed_2022"].astype(str) != "True") &
    (df["any_suppressed_2024"].astype(str) != "True")
]
print(f"移除 suppressed 後：{len(df)} 個縣")

df_clean = df.dropna(subset=VARS_A3 + ["rate_change"]).copy()
print(f"完整資料的縣：{len(df_clean)} 個")

# ── 3. 跑 OLS A3 Final Model（HC3 robust SE）─────────────────────────────────
X = sm.add_constant(df_clean[VARS_A3])
y = df_clean["rate_change"]
model = sm.OLS(y, X).fit(cov_type="HC3")

print(f"\nA3  R² = {model.rsquared:.3f}")
print(f"N  = {int(model.nobs)}")
print("\n係數：")
for var in VARS_A3:
    coef = model.params[var]
    pval = model.pvalues[var]
    sig  = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {var:<30} β={coef:+.3f}  p={pval:.4f} {sig}")

# ── 4. 計算 residual / fitted ─────────────────────────────────────────────────
df_clean = df_clean.copy()
df_clean["fitted"]   = model.fittedvalues
df_clean["residual"] = model.resid

# ── 5. Outlier 計算（排除 rate 兩期都是 0，或人口 < min_pop）────────────────
valid_mask = ~(
    ((df_clean["overdose_rate_per_100k_2022"] == 0) &
     (df_clean["overdose_rate_per_100k_2024"] == 0)) |
    (df_clean["total_pop_2022"] < args.min_pop)
)

threshold = df_clean.loc[valid_mask, "residual"].nlargest(args.outlier_n).min()
df_clean["is_outlier"] = (df_clean["residual"] >= threshold) & valid_mask

# outlier 排名（1 = 最極端）
df_clean["outlier_rank"] = 0
for rank, idx in enumerate(
    df_clean[df_clean["is_outlier"]].nlargest(args.outlier_n, "residual").index,
    start=1
):
    df_clean.loc[idx, "outlier_rank"] = rank

n_outliers = df_clean["is_outlier"].sum()
print(f"\nOutlier 縣數：{n_outliers}（min_pop filter = {args.min_pop}）")
print(df_clean[df_clean["is_outlier"]][
    ["county_fips", "county_name_2022", "total_pop_2022", "rate_change", "fitted", "residual", "outlier_rank"]
].sort_values("outlier_rank").to_string())

# ── 6. 建立 fips → residual lookup ───────────────────────────────────────────
residual_lookup = {}
for _, row in df_clean.iterrows():
    fips = str(row["county_fips"]).zfill(5)
    residual_lookup[fips] = {
        "residual":     round(float(row["residual"]), 2),
        "fitted":       round(float(row["fitted"]), 2),
        "is_outlier":   bool(row["is_outlier"]),
        "outlier_rank": int(row["outlier_rank"]),
    }

print(f"\nLookup 建好，共 {len(residual_lookup)} 筆")

# ── 7. Merge 進 counties json ─────────────────────────────────────────────────
with open(args.input) as f:
    counties = json.load(f)

added = 0
for county in counties:
    fips = str(county.get("fips", "")).zfill(5)
    if fips in residual_lookup:
        county.update(residual_lookup[fips])
        added += 1
    else:
        county["residual"]     = None
        county["fitted"]       = None
        county["is_outlier"]   = False
        county["outlier_rank"] = 0

print(f"成功 merge {added} 筆 residual 進 json")
print(f"（其餘 {len(counties) - added} 個縣無 residual）")

# ── 8. 輸出 ──────────────────────────────────────────────────────────────────
with open(args.output, "w") as f:
    json.dump(counties, f, separators=(",", ":"))

print(f"\n✅ 輸出完成：{args.output}")

# ── 9. 驗證 Highland County ───────────────────────────────────────────────────
highland = [c for c in counties if str(c.get("fips", "")).zfill(5) == "51091"]
if highland:
    h = highland[0]
    print(f"\nHighland County 驗證：is_outlier={h['is_outlier']}, outlier_rank={h['outlier_rank']}, residual={h['residual']}")
