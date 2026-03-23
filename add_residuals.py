"""
把 Model A (Final Vulnerability Model) 的 residual、fitted value、outlier flag
加進 counties.json，輸出 counties_with_residuals.json

使用方式：
  python add_residuals.py \
    --csv2022 data/2022_county_overdose_analysis_final.csv \
    --csv2024 data/2024_county_overdose_analysis_final.csv \
    --input   data/counties.json \
    --output  data/counties_with_residuals.json
"""

import argparse
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# ── 0. CLI 參數 ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv2022", required=True)
parser.add_argument("--csv2024", required=True)
parser.add_argument("--input",   required=True,  help="原本的 counties.json")
parser.add_argument("--output",  required=True,  help="輸出路徑")
parser.add_argument("--outlier_n", type=int, default=20, help="定義 outlier 的 top N（預設 20）")
args = parser.parse_args()

# ── 1. 載入資料 ───────────────────────────────────────────────────────────────
df_2022 = pd.read_csv(args.csv2022)
df_2024 = pd.read_csv(args.csv2024)

df = df_2022.merge(df_2024, on="county_fips", suffixes=("_2022", "_2024"))
df["rate_change"] = df["overdose_rate_per_100k_2024"] - df["overdose_rate_per_100k_2022"]

# 確保 fips 是字串並補零到 5 碼
df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)

print(f"合併後總縣數：{len(df)}")

# ── 2. 對齊 analysi.ipynb 的變數名稱 ─────────────────────────────────────────
# 這裡列出 2022 baseline 的所有候選變數（跟你 notebook 的 VARS_2022 一致）
VARS_2022 = [
    "median_income_2022", "unemployment_rate_2022", "poverty_rate_2022",
    "bachelor_plus_pct_2022", "uninsured_2022",
    "white_2022", "black_2022", "native_2022", "asian_2022",
    "pacific_2022", "other_2022", "two_more_2022", "hispanic_2022",
    "age_20to24_2022", "age_25to34_2022", "age_35to44_2022",
    "age_45to54_2022", "age_55to59_2022", "age_60to64_2022",
    "male_2022", "female_2022",
]

# 只保留實際存在的欄位
VARS_2022 = [v for v in VARS_2022 if v in df.columns]
print(f"可用的 baseline 變數：{len(VARS_2022)} 個")

# ── 3. 清理資料（與 notebook 相同邏輯）────────────────────────────────────────
df_clean = df.dropna(subset=VARS_2022 + ["rate_change"]).copy()
print(f"完整資料的縣：{len(df_clean)} 個")

# ── 4. LASSO 變數選擇（10-fold CV）────────────────────────────────────────────
X_all = df_clean[VARS_2022].values
y_all = df_clean["rate_change"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

lasso = LassoCV(cv=10, random_state=42, max_iter=10000)
lasso.fit(X_scaled, y_all)

selected_A = [col for col, coef in zip(VARS_2022, lasso.coef_) if coef != 0]
print(f"LASSO A 選出 {len(selected_A)} 個變數：{selected_A}")

# ── 5. VIF 檢查 ───────────────────────────────────────────────────────────────
from statsmodels.stats.outliers_influence import variance_inflation_factor

def remove_high_vif(cols, data, threshold=10):
    remaining = list(cols)
    while True:
        sub = data[remaining].dropna().reset_index(drop=True).values
        vifs = [variance_inflation_factor(sub, i) for i in range(len(remaining))]
        max_vif = max(vifs)
        if max_vif <= threshold:
            break
        drop_col = remaining[vifs.index(max_vif)]
        print(f"  VIF={max_vif:.1f}，移除：{drop_col}")
        remaining.remove(drop_col)
    return remaining

selected_A = remove_high_vif(selected_A, df_clean)
print(f"VIF 後剩餘：{selected_A}")

# ── 6. OLS A1 → A2 → A3（複製 notebook 邏輯）────────────────────────────────
def fit_ols(cols, df_input):
    sub = df_input.dropna(subset=cols + ["rate_change"]).copy()
    X = sm.add_constant(sub[cols])
    y = sub["rate_change"]
    mdl = sm.OLS(y, X).fit(cov_type="HC3")
    return mdl, sub

model_A1, df_A1 = fit_ols(selected_A, df_clean)
print(f"A1  R² = {model_A1.rsquared:.3f}")

sig_A = [v for v in selected_A
         if v in model_A1.pvalues.index and model_A1.pvalues[v] < 0.05]
model_A2, df_A2 = fit_ols(sig_A, df_clean)
print(f"A2  R² = {model_A2.rsquared:.3f}  variables: {sig_A}")

sig_A3 = [v for v in sig_A
          if v in model_A2.pvalues.index and model_A2.pvalues[v] < 0.05]
if sig_A3 and set(sig_A3) != set(sig_A):
    model_final, df_final = fit_ols(sig_A3, df_clean)
    print(f"A3  R² = {model_final.rsquared:.3f}  variables: {sig_A3}")
else:
    model_final, df_final = model_A2, df_A2
    print("A3 與 A2 相同，使用 A2 為 Final Model")

# ── 7. 計算 residual / fitted / outlier ──────────────────────────────────────
df_final = df_final.copy()
df_final["fitted"]   = model_final.fittedvalues
df_final["residual"] = model_final.resid

# Outlier = residual top N（實際比預測更差）
threshold = df_final["residual"].nlargest(args.outlier_n).min()
df_final["is_outlier"] = df_final["residual"] >= threshold

# outlier 排名（1 = 最極端）
df_final["outlier_rank"] = 0
top_idx = df_final.nlargest(args.outlier_n, "residual").index
for rank, idx in enumerate(df_final.nlargest(args.outlier_n, "residual").index, start=1):
    df_final.loc[idx, "outlier_rank"] = rank

n_outliers = df_final["is_outlier"].sum()
print(f"\nOutlier 縣數：{n_outliers}")
print(df_final[df_final["is_outlier"]][["county_fips", "county_name_2022", "rate_change", "fitted", "residual"]].head(10).to_string())

# ── 8. 建立 fips → residual lookup dict ──────────────────────────────────────
residual_lookup = {}
for _, row in df_final.iterrows():
    fips = str(row["county_fips"]).zfill(5)
    residual_lookup[fips] = {
        "residual":     round(float(row["residual"]), 2),
        "fitted":       round(float(row["fitted"]), 2),
        "is_outlier":   bool(row["is_outlier"]),
        "outlier_rank": int(row["outlier_rank"]),
    }

print(f"\nLookup 建好，共 {len(residual_lookup)} 筆")

# ── 9. 把 residual 資料 merge 進 counties.json ────────────────────────────────
with open(args.input) as f:
    counties = json.load(f)

added = 0
for county in counties:
    fips = str(county.get("fips", "")).zfill(5)
    if fips in residual_lookup:
        county.update(residual_lookup[fips])
        added += 1
    else:
        # 沒進迴歸（suppressed / missing data），給 null
        county["residual"]     = None
        county["fitted"]       = None
        county["is_outlier"]   = False
        county["outlier_rank"] = 0

print(f"成功 merge {added} 筆 residual 進 counties.json")
print(f"（其餘 {len(counties) - added} 個縣因 suppressed/missing data 無 residual）")

# ── 10. 輸出 ─────────────────────────────────────────────────────────────────
with open(args.output, "w") as f:
    json.dump(counties, f, separators=(",", ":"))

print(f"\n✅ 輸出完成：{args.output}")

# 簡單驗證
sample_outliers = [c for c in counties if c.get("is_outlier")]
print(f"\nOutlier 縣（前 5 筆）：")
for c in sorted(sample_outliers, key=lambda x: x.get("outlier_rank", 999))[:5]:
    print(f"  [{c['outlier_rank']}] {c['name']}  residual={c['residual']}  rate_change={c.get('rate_change')}")
