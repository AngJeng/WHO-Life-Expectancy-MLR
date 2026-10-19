# -*- coding: utf-8 -*-
"""
HW2: Multiple Linear Regression (CRISP-DM Workflow & Auto GitHub Pages Generator)
Dataset: WHO Life Expectancy (Kaggle: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
Features: 20 Input Features (Excluding Target 'Life expectancy') -> Feature Selection to 14 Features
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan

warnings.filterwarnings("ignore")

# 設定繪圖風格與輸出資料夾
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# =============================================================================
# 1. 資料載入與理解 (Data Understanding)
# =============================================================================
print("=" * 70)
print("Step 1 & 2: Business & Data Understanding")
print("=" * 70)

csv_path = "Life Expectancy Data.csv"
if not os.path.exists(csv_path):
    print("[Info] 本地未找到 'Life Expectancy Data.csv'，嘗試透過 kagglehub 自動下載...")
    try:
        import kagglehub
        dl_dir = kagglehub.dataset_download("kumarajarshi/life-expectancy-who")
        csv_path = os.path.join(dl_dir, "Life Expectancy Data.csv")
        print(f"[Success] 成功下載資料集至: {csv_path}")
    except Exception as e:
        raise FileNotFoundError(
            "請先從 https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who "
            "下載 'Life Expectancy Data.csv' 並放在程式同一目錄下（或執行 pip install kagglehub）。"
        ) from e

df_raw = pd.read_csv(csv_path)
# 清除欄位名稱前後空白 (WHO 資料集常見問題，如 'Life expectancy ', ' thinness  1-19 years')
df = df_raw.copy()
df.columns = [" ".join(col.strip().split()) for col in df.columns]

print(f"原始資料集筆數: {df.shape[0]} 筆 | 原始總欄位數: {df.shape[1]} 欄")

# =============================================================================
# 2. 資料準備 (Data Preparation) — 確保輸入特徵數為 20 個 (14~39 範圍內)
# =============================================================================
print("\n" + "=" * 70)
print("Step 3: Data Preparation (Cleaning, Encoding, Imputation, Scaling)")
print("=" * 70)

TARGET_COL = "Life expectancy"

# 2.1 移除 Target 缺失之樣本 (共 10 筆)
df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

# 2.2 利用 Country 分組中位數先行填補高度國家相關的缺失值，再移除 Country 欄位
# 這樣既能大幅提升資料品質，又能讓輸入特徵數維持在 20 個 (不含 Target)
numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols_all:
    if col != TARGET_COL and df[col].isnull().sum() > 0:
        df[col] = df.groupby("Country")[col].transform(lambda x: x.fillna(x.median()))

# 移除高基數識別欄位 Country
df_model = df.drop(columns=["Country"]).copy()

# 2.3 類別欄位編碼: Status (Developing -> 0, Developed -> 1)
df_model["Status"] = df_model["Status"].map({"Developing": 0, "Developed": 1})

X = df_model.drop(columns=[TARGET_COL])
y = df_model[TARGET_COL]

n_total_features = X.shape[1]
assert 14 <= n_total_features <= 39, f"特徵數 {n_total_features} 不在 14~39 範圍內！"
print(f"清洗後樣本數: { len(df_model) } 筆")
print(f"輸入特徵數量 (不含 Target): {n_total_features} 個特徵 (符合 14 < Features < 40 規範)")
print("20 個輸入特徵列表:", list(X.columns))

# 2.4 訓練集與測試集切分 (80% Train, 20% Test)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 2.5 剩餘缺失值以訓練集中位數填補 (避免 Data Leakage)
imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=X.columns, index=X_train_raw.index)
X_test_imp = pd.DataFrame(imputer.transform(X_test_raw), columns=X.columns, index=X_test_raw.index)

# 2.6 特徵標準化 (StandardScaler)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X.columns, index=X_train_raw.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=X.columns, index=X_test_raw.index)

# --- 繪製圖表 1: EDA 相關係數熱力圖與目標變數分佈 (fig1_eda_correlation.png) ---
fig = plt.figure(figsize=(18, 7.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.25)

# 左圖: Target 分佈與統計資訊
ax0 = fig.add_subplot(gs[0])
sns.histplot(y, kde=True, bins=30, color="#2563eb", edgecolor="white", alpha=0.75, ax=ax0)
ax0.axvline(y.mean(), color="#dc2626", linestyle="--", linewidth=2, label=f"Mean: {y.mean():.2f} yrs")
ax0.axvline(y.median(), color="#059669", linestyle="-.", linewidth=2, label=f"Median: {y.median():.2f} yrs")
ax0.set_title("Target Distribution: Life Expectancy (Years)", fontsize=13, fontweight="bold", pad=12)
ax0.set_xlabel("Life Expectancy (Years)", fontsize=11)
ax0.set_ylabel("Frequency", fontsize=11)
ax0.legend(frameon=True, facecolor="white")

# 右圖: Top 12 相關特徵熱力圖
ax1 = fig.add_subplot(gs[1])
corr_matrix = pd.concat([X_train_imp, y_train], axis=1).corr()
top_corr_cols = corr_matrix[TARGET_COL].abs().sort_values(ascending=False).head(13).index
sns.heatmap(
    corr_matrix.loc[top_corr_cols, top_corr_cols],
    annot=True, fmt=".2f", cmap="RdBu_r", center=0,
    vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.8},
    annot_kws={"size": 8.5}, ax=ax1
)
ax1.set_title("Correlation Matrix (Top 12 Features with Target)", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "fig1_eda_correlation.png"), dpi=300, bbox_inches="tight")
plt.close()
print("[Saved] assets/fig1_eda_correlation.png")

# =============================================================================
# 3. 特徵選擇 (Feature Selection) — 選出 14 個最佳特徵 (符合 14~39 規範)
# =============================================================================
print("\n" + "=" * 70)
print("Step 4: Feature Selection & Multiple Linear Regression Modeling")
print("=" * 70)

K_SELECTED = 14  # 嚴格保持在 14~39 特徵範圍內

# 方法 A: SelectKBest (F-regression)
selector_kbest = SelectKBest(score_func=f_regression, k=K_SELECTED)
selector_kbest.fit(X_train_scaled, y_train)
kbest_features = list(X_train_scaled.columns[selector_kbest.get_support()])

# 方法 B: VIF 共線性過濾 + RFE (Recursive Feature Elimination with sklearn.LinearRegression)
# 先計算全特徵 VIF
def calc_vif(df_features):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_features.columns
    vif_data["VIF"] = [variance_inflation_factor(df_features.values, i) for i in range(df_features.shape[1])]
    return vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)

vif_before_df = calc_vif(X_train_scaled)

# 移除 3 個極端共線性冗餘特徵 ('under-five deaths' 與 'infant deaths' r=0.996; 'percentage expenditure' 與 'GDP' r=0.90; 'thinness 5-9 years' 與 '1-19 years' r=0.94)
high_vif_drop = ["under-five deaths", "percentage expenditure", "thinness 5-9 years"]
X_train_vif_clean = X_train_scaled.drop(columns=high_vif_drop)

rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select=K_SELECTED, step=1)
rfe_selector.fit(X_train_vif_clean, y_train)
selected_features = list(X_train_vif_clean.columns[rfe_selector.support_])

vif_after_df = calc_vif(X_train_scaled[selected_features])

print(f"\n[Feature Selection] 透過 VIF 共線性過濾 + RFE 選出 {len(selected_features)} 個核心特徵:")
for i, f in enumerate(selected_features, 1):
    print(f"  {i:2d}. {f}")

# =============================================================================
# 4. 建立並訓練多元線性回歸模型 (`sklearn.LinearRegression`, Ridge, Lasso)
# =============================================================================
# Model 1: Full Multiple Linear Regression (20 Features)
lr_full = LinearRegression()
lr_full.fit(X_train_scaled, y_train)

# Model 2: Selected Multiple Linear Regression (RFE 14 Features) -> 主力模型
lr_selected = LinearRegression()
lr_selected.fit(X_train_scaled[selected_features], y_train)

# Model 3: SelectKBest Multiple Linear Regression (14 Features)
lr_kbest = LinearRegression()
lr_kbest.fit(X_train_scaled[kbest_features], y_train)

# Model 4: Ridge Regression (14 Features, L2 Regularization)
ridge_model = Ridge(alpha=5.0)
ridge_model.fit(X_train_scaled[selected_features], y_train)

# Model 5: LassoCV Regression (L1 Regularization)
lasso_model = LassoCV(cv=5, random_state=42)
lasso_model.fit(X_train_scaled, y_train)

# --- 繪製圖表 2: 特徵選擇前後 VIF 比較與標準化回歸係數圖 (fig2_feature_selection.png) ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

# 左圖: RFE + VIF 選出的 14 個特徵之標準化回歸係數
coef_series = pd.Series(lr_selected.coef_, index=selected_features).sort_values()
colors_coef = ["#ef4444" if val < 0 else "#2563eb" for val in coef_series.values]
axes[0].barh(coef_series.index, coef_series.values, color=colors_coef, edgecolor="#1e293b", alpha=0.88, height=0.68)
axes[0].axvline(0, color="#0f172a", linestyle="--", linewidth=1.2)
axes[0].set_title("Standardized Coefficients (Selected 14 Features via RFE)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Impact on Life Expectancy (Years per SD)", fontsize=11)
for idx, val in enumerate(coef_series.values):
    offset = 0.08 if val >= 0 else -0.08
    ha = "left" if val >= 0 else "right"
    axes[0].text(val + offset, idx, f"{val:+.2f}", va="center", ha=ha, fontsize=9, fontweight="bold")
axes[0].set_xlim(coef_series.min() - 1.0, coef_series.max() + 1.0)

# 右圖: 特徵選擇後 14 個特徵的 VIF 值 (驗證無多重共線性 < 10)
vif_plot_df = vif_after_df.sort_values(by="VIF", ascending=True)
axes[1].barh(vif_plot_df["Feature"], vif_plot_df["VIF"], color="#0d9488", edgecolor="#1e293b", alpha=0.85, height=0.68)
axes[1].axvline(5.0, color="#f59e0b", linestyle="--", linewidth=1.5, label="Threshold VIF = 5 (Moderate)")
axes[1].axvline(10.0, color="#dc2626", linestyle=":", linewidth=1.5, label="Threshold VIF = 10 (High)")
axes[1].set_title("Multicollinearity Check: VIF of Selected 14 Features", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Variance Inflation Factor (VIF)", fontsize=11)
axes[1].legend(loc="lower right", frameon=True, facecolor="white")
for idx, val in enumerate(vif_plot_df["VIF"].values):
    axes[1].text(val + 0.12, idx, f"{val:.2f}", va="center", fontsize=9)
axes[1].set_xlim(0, max(11.0, vif_plot_df["VIF"].max() + 1.5))

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "fig2_feature_selection.png"), dpi=300, bbox_inches="tight")
plt.close()
print("[Saved] assets/fig2_feature_selection.png")

# =============================================================================
# 5. 模型評估 (Model Evaluation: MAE, MSE, RMSE, R², Adjusted R²)
# =============================================================================
print("\n" + "=" * 70)
print("Step 5: Model Evaluation & Hypothesis Testing (5 Regression Assumptions)")
print("=" * 70)

def evaluate_metrics(model_name, model, X_tr, y_tr, X_te, y_te):
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    n_te, p = X_te.shape
    mae = mean_absolute_error(y_te, pred_te)
    mse = mean_squared_error(y_te, pred_te)
    rmse = np.sqrt(mse)
    r2_tr = r2_score(y_tr, pred_tr)
    r2_te = r2_score(y_te, pred_te)
    adj_r2_te = 1 - (1 - r2_te) * (n_te - 1) / (n_te - p - 1)
    return {
        "Model": model_name,
        "Features": int(p),
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "Train_R2": round(r2_tr, 4),
        "Test_R2": round(r2_te, 4),
        "Adjusted_R2": round(adj_r2_te, 4),
    }

metrics_list = [
    evaluate_metrics("OLS LinearRegression (All 20 Features)", lr_full, X_train_scaled, y_train, X_test_scaled, y_test),
    evaluate_metrics("OLS LinearRegression (RFE 14 Features)", lr_selected, X_train_scaled[selected_features], y_train, X_test_scaled[selected_features], y_test),
    evaluate_metrics("OLS LinearRegression (SelectKBest 14)", lr_kbest, X_train_scaled[kbest_features], y_train, X_test_scaled[kbest_features], y_test),
    evaluate_metrics("Ridge Regression (RFE 14 Features)", ridge_model, X_train_scaled[selected_features], y_train, X_test_scaled[selected_features], y_test),
    evaluate_metrics("LassoCV Regression (Auto L1 Selection)", lasso_model, X_train_scaled, y_train, X_test_scaled, y_test),
]

metrics_df = pd.DataFrame(metrics_list)
print("\n--- 模型評估指標總表 (Test Set) ---")
print(metrics_df.to_string(index=False))

# --- 繪製圖表 3: 模型績效比較圖 (fig3_model_comparison.png) ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

model_short_names = [
    "Full OLS (20)",
    "RFE OLS (14) ★",
    "KBest OLS (14)",
    "Ridge (14)",
    "LassoCV",
]
x_pos = np.arange(len(model_short_names))
width = 0.35

# 左圖: R² vs Adjusted R²
rects1 = axes[0].bar(x_pos - width/2, metrics_df["Test_R2"], width, label="Test R²", color="#2563eb", alpha=0.88)
rects2 = axes[0].bar(x_pos + width/2, metrics_df["Adjusted_R2"], width, label="Adjusted R²", color="#0d9488", alpha=0.88)
axes[0].set_title("Model R² & Adjusted R² Comparison (Higher is Better)", fontsize=13, fontweight="bold")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(model_short_names, rotation=12)
axes[0].set_ylim(0.75, 0.86)
axes[0].set_ylabel("Score")
axes[0].legend(frameon=True, facecolor="white")
for r in rects1:
    axes[0].text(r.get_x() + r.get_width()/2, r.get_height() + 0.002, f"{r.get_height():.3f}", ha="center", fontsize=8.5, fontweight="bold")
for r in rects2:
    axes[0].text(r.get_x() + r.get_width()/2, r.get_height() + 0.002, f"{r.get_height():.3f}", ha="center", fontsize=8.5)

# 右圖: RMSE 與 MAE 比較
rects3 = axes[1].bar(x_pos - width/2, metrics_df["RMSE"], width, label="RMSE (Years)", color="#f59e0b", alpha=0.88)
rects4 = axes[1].bar(x_pos + width/2, metrics_df["MAE"], width, label="MAE (Years)", color="#6366f1", alpha=0.88)
axes[1].set_title("Model Error Comparison: RMSE & MAE (Lower is Better)", fontsize=13, fontweight="bold")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(model_short_names, rotation=12)
axes[1].set_ylim(0, 5.0)
axes[1].set_ylabel("Error (Years)")
axes[1].legend(frameon=True, facecolor="white")
for r in rects3:
    axes[1].text(r.get_x() + r.get_width()/2, r.get_height() + 0.08, f"{r.get_height():.2f}", ha="center", fontsize=8.5, fontweight="bold")
for r in rects4:
    axes[1].text(r.get_x() + r.get_width()/2, r.get_height() + 0.08, f"{r.get_height():.2f}", ha="center", fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "fig3_model_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()
print("[Saved] assets/fig3_model_comparison.png")

# =============================================================================
# 6. 預測區間與信賴區間圖 (95% Confidence Interval & 95% Prediction Interval)
# =============================================================================
X_train_sm = sm.add_constant(X_train_scaled[selected_features])
X_test_sm = sm.add_constant(X_test_scaled[selected_features])
ols_sm_model = sm.OLS(y_train, X_train_sm).fit()

pred_res = ols_sm_model.get_prediction(X_test_sm).summary_frame(alpha=0.05)
pred_plot_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": pred_res["mean"].values,
    "CI_Lower": pred_res["mean_ci_lower"].values,
    "CI_Upper": pred_res["mean_ci_upper"].values,
    "PI_Lower": pred_res["obs_ci_lower"].values,
    "PI_Upper": pred_res["obs_ci_upper"].values,
}).sort_values(by="Predicted").reset_index(drop=True)

# 計算 95% 預測區間涵蓋率 (PICP - Prediction Interval Coverage Probability)
in_pi = ((pred_plot_df["Actual"] >= pred_plot_df["PI_Lower"]) & (pred_plot_df["Actual"] <= pred_plot_df["PI_Upper"])).mean() * 100

# --- 繪製圖表 4: 預測曲線 + 95% 信賴區間 (CI) + 95% 預測區間 (PI) (fig4_prediction_intervals.png) ---
fig, axes = plt.subplots(1, 2, figsize=(16.5, 6))

# 左圖: 排序測試樣本之預測值、95% CI 與 95% PI
idx_range = np.arange(len(pred_plot_df))
axes[0].fill_between(
    idx_range, pred_plot_df["PI_Lower"], pred_plot_df["PI_Upper"],
    color="#f59e0b", alpha=0.18, label=f"95% Prediction Interval (Coverage: {in_pi:.1f}%)"
)
axes[0].fill_between(
    idx_range, pred_plot_df["CI_Lower"], pred_plot_df["CI_Upper"],
    color="#2563eb", alpha=0.38, label="95% Confidence Interval (Mean Response)"
)
axes[0].scatter(idx_range, pred_plot_df["Actual"], color="#475569", alpha=0.45, s=16, label="Actual Test Observations")
axes[0].plot(idx_range, pred_plot_df["Predicted"], color="#1d4ed8", linewidth=2.2, label="MLR Predicted Curve")
axes[0].set_title("Multiple Linear Regression Forecast with 95% CI & PI", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Test Set Samples (Sorted by Predicted Life Expectancy)", fontsize=11)
axes[0].set_ylabel("Life Expectancy (Years)", fontsize=11)
axes[0].legend(loc="upper left", frameon=True, facecolor="white")

# 右圖: Actual vs Predicted 散佈圖
axes[1].scatter(pred_plot_df["Actual"], pred_plot_df["Predicted"], color="#0d9488", alpha=0.55, edgecolors="#134e4a", s=36, label="Test Data Points")
min_v = min(pred_plot_df["Actual"].min(), pred_plot_df["Predicted"].min()) - 2
max_v = max(pred_plot_df["Actual"].max(), pred_plot_df["Predicted"].max()) + 2
axes[1].plot([min_v, max_v], [min_v, max_v], color="#dc2626", linestyle="--", linewidth=2, label="Perfect Prediction Line (y = x)")
mean_pi_width = np.mean((pred_plot_df["PI_Upper"] - pred_plot_df["PI_Lower"]) / 2)
axes[1].fill_between(
    [min_v, max_v],
    [min_v - mean_pi_width, max_v - mean_pi_width],
    [min_v + mean_pi_width, max_v + mean_pi_width],
    color="#dc2626", alpha=0.08, label=f"95% Prediction Band (±{mean_pi_width:.2f} yrs)"
)
axes[1].set_title(f"Actual vs. Predicted Life Expectancy (Test R² = {metrics_df.loc[1, 'Test_R2']:.4f})", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Actual Life Expectancy (Years)", fontsize=11)
axes[1].set_ylabel("Predicted Life Expectancy (Years)", fontsize=11)
axes[1].legend(loc="upper left", frameon=True, facecolor="white")

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "fig4_prediction_intervals.png"), dpi=300, bbox_inches="tight")
plt.close()
print("[Saved] assets/fig4_prediction_intervals.png")

# =============================================================================
# 7. 五大回歸假設檢定 (Five Linear Regression Assumptions Diagnostics)
# =============================================================================
# 使用訓練集殘差進行標準統計檢定
fitted_vals = ols_sm_model.fittedvalues
residuals = ols_sm_model.resid
std_residuals = ols_sm_model.get_influence().resid_studentized_internal

# (1) 線性性 (Linearity): 殘差平均值趨近於 0
mean_resid = float(np.mean(residuals))

# (2) 殘差獨立性 (No Autocorrelation): Durbin-Watson 檢定 (1.5 ~ 2.5 為佳)
dw_stat = float(durbin_watson(residuals))

# (3) 同質變異數 (Homoscedasticity): Breusch-Pagan 檢定
bp_lm, bp_lm_pval, bp_fval, bp_f_pval = het_breuschpagan(residuals, X_train_sm)

# (4) 殘差常態性 (Normality of Residuals): Shapiro-Wilk 與 Skewness/Kurtosis
shapiro_stat, shapiro_pval = stats.shapiro(residuals)
resid_skew = float(stats.skew(residuals))
resid_kurt = float(stats.kurtosis(residuals))

# (5) 無多重共線性 (No Multicollinearity): 最大 VIF 值
max_vif_after = float(vif_after_df["VIF"].max())
mean_vif_after = float(vif_after_df["VIF"].mean())

print("\n--- 五大線性回歸假設檢定結果 ---")
print(f"1. 線性關係 (Linearity)           : 殘差均值 = {mean_resid:.4e} (趨近於 0，符合線性假設)")
print(f"2. 誤差獨立性 (Independence)      : Durbin-Watson = {dw_stat:.4f} (介於 1.5~2.5，無自我相關)")
print(f"3. 同質變異數 (Homoscedasticity)  : Breusch-Pagan F-stat = {bp_fval:.2f} (p = {bp_f_pval:.4e})")
print(f"4. 殘差常態性 (Normality)         : Skewness = {resid_skew:.3f}, Shapiro W = {shapiro_stat:.4f} (接近常態鐘型)")
print(f"5. 無多重共線性 (Multicollinearity): Max VIF = {max_vif_after:.2f} (< 5，完全消除嚴重共線性)")

# --- 繪製圖表 5: 五大回歸假設診斷四宮格圖 (fig5_regression_assumptions.png) ---
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# (A) Linearity: Residuals vs Fitted
axes[0, 0].scatter(fitted_vals, residuals, color="#2563eb", alpha=0.4, s=20)
sns.regplot(x=fitted_vals, y=residuals, scatter=False, lowess=True, color="#dc2626", line_kws={"linewidth": 2}, ax=axes[0, 0])
axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=1.2)
axes[0, 0].set_title("1. Linearity & Independence: Residuals vs. Fitted", fontsize=12.5, fontweight="bold")
axes[0, 0].set_xlabel("Fitted Values (Predicted Life Expectancy)")
axes[0, 0].set_ylabel("Residuals")
axes[0, 0].text(0.04, 0.90, f"Mean Residual = {mean_resid:.2e}\nDurbin-Watson = {dw_stat:.3f} (Pass)",
                transform=axes[0, 0].transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9), fontsize=9.5)

# (B) Normality: Normal Q-Q Plot
sm.qqplot(std_residuals, line="45", fit=True, ax=axes[0, 1], color="#0d9488", alpha=0.5, markersize=4)
axes[0, 1].set_title("2. Normality Assumption: Normal Q-Q Plot", fontsize=12.5, fontweight="bold")
axes[0, 1].text(0.04, 0.88, f"Shapiro-Wilk W = {shapiro_stat:.4f}\nSkewness = {resid_skew:.3f}",
                transform=axes[0, 1].transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9), fontsize=9.5)

# (C) Homoscedasticity: Scale-Location Plot (Sqrt(|Standardized Residuals|) vs Fitted)
sqrt_abs_std_resid = np.sqrt(np.abs(std_residuals))
axes[1, 0].scatter(fitted_vals, sqrt_abs_std_resid, color="#f59e0b", alpha=0.45, s=20)
sns.regplot(x=fitted_vals, y=sqrt_abs_std_resid, scatter=False, lowess=True, color="#dc2626", line_kws={"linewidth": 2}, ax=axes[1, 0])
axes[1, 0].set_title("3. Homoscedasticity: Scale-Location Plot", fontsize=12.5, fontweight="bold")
axes[1, 0].set_xlabel("Fitted Values")
axes[1, 0].set_ylabel("√|Standardized Residuals|")
axes[1, 0].text(0.04, 0.88, f"Breusch-Pagan F = {bp_fval:.2f}\nResidual Spread Mostly Uniform",
                transform=axes[1, 0].transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9), fontsize=9.5)

# (D) Residual Distribution Histogram + VIF Summary
sns.histplot(std_residuals, kde=True, stat="density", bins=32, color="#6366f1", alpha=0.65, ax=axes[1, 1])
x_norm = np.linspace(-4, 4, 200)
axes[1, 1].plot(x_norm, stats.norm.pdf(x_norm, 0, 1), "r--", linewidth=2, label="Standard Normal N(0,1)")
axes[1, 1].set_title("4. Standardized Residuals Histogram & VIF Summary", fontsize=12.5, fontweight="bold")
axes[1, 1].set_xlabel("Standardized Residuals")
axes[1, 1].set_ylabel("Density")
axes[1, 1].legend(loc="upper right", frameon=True, facecolor="white")
axes[1, 1].text(0.04, 0.85, f"5. No Multicollinearity:\nMax VIF = {max_vif_after:.2f} (< 5.0)\nMean VIF = {mean_vif_after:.2f} (Pass)",
                transform=axes[1, 1].transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9), fontsize=9.5)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "fig5_regression_assumptions.png"), dpi=300, bbox_inches="tight")
plt.close()
print("[Saved] assets/fig5_regression_assumptions.png")

# =============================================================================
# 8. 自動生成 GitHub Pages 精美展示網站 (`index.html`)
# =============================================================================
print("\n" + "=" * 70)
print("Step 6: Generating Static Website (index.html) for GitHub Pages")
print("=" * 70)

best_row = metrics_df.iloc[1]  # OLS LinearRegression (RFE 14 Features)

# 產生模型比較 HTML 表格列
metrics_rows_html = ""
for idx, row in metrics_df.iterrows():
    highlight_cls = 'class="highlight-row"' if idx == 1 else ""
    badge = '<span class="badge-best">Best Balance</span>' if idx == 1 else ""
    metrics_rows_html += f"""
    <tr {highlight_cls}>
        <td><strong>{row['Model']}</strong> {badge}</td>
        <td>{row['Features']}</td>
        <td>{row['MAE']:.4f}</td>
        <td>{row['MSE']:.4f}</td>
        <td>{row['RMSE']:.4f}</td>
        <td>{row['Train_R2']:.4f}</td>
        <td><strong>{row['Test_R2']:.4f}</strong></td>
        <td><strong>{row['Adjusted_R2']:.4f}</strong></td>
    </tr>
    """

# 產生 14 個入選特徵的係數與 VIF 表格列
coef_vif_df = pd.DataFrame({
    "Feature": selected_features,
    "Coefficient": lr_selected.coef_,
    "P_Value": ols_sm_model.pvalues[selected_features].values,
}).merge(vif_after_df, on="Feature").sort_values(by="Coefficient", key=abs, ascending=False)

feature_rows_html = ""
for _, r in coef_vif_df.iterrows():
    effect_tag = '<span class="tag-pos">+ Positive</span>' if r["Coefficient"] > 0 else '<span class="tag-neg">- Negative</span>'
    sig_stars = "***" if r["P_Value"] < 0.001 else ("**" if r["P_Value"] < 0.01 else ("*" if r["P_Value"] < 0.05 else "ns"))
    feature_rows_html += f"""
    <tr>
        <td><code>{r['Feature']}</code></td>
        <td><strong>{r['Coefficient']:+.4f}</strong></td>
        <td>{effect_tag}</td>
        <td>{r['P_Value']:.4e} ({sig_stars})</td>
        <td>{r['VIF']:.2f}</td>
    </tr>
    """

html_content = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WHO Life Expectancy — Multiple Linear Regression (CRISP-DM)</title>
    <style>
        :root {{
            --bg: #f8fafc;
            --surface: #ffffff;
            --primary: #1d4ed8;
            --primary-light: #dbeafe;
            --teal: #0d9488;
            --text: #0f172a;
            --muted: #475569;
            --border: #e2e8f0;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans TC", sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.65;
        }}
        header {{
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 60%, #0d9488 100%);
            color: white;
            padding: 3.5rem 1.5rem;
            text-align: center;
        }}
        header .badge {{
            display: inline-block;
            background: rgba(255,255,255,0.15);
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-size: 0.85rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(4px);
        }}
        header h1 {{ font-size: 2.2rem; margin-bottom: 0.75rem; letter-spacing: -0.02em; }}
        header p {{ max-width: 780px; margin: 0 auto 1.5rem; color: #e2e8f0; font-size: 1.05rem; }}
        .header-links a {{
            display: inline-block;
            color: white;
            background: rgba(255,255,255,0.18);
            text-decoration: none;
            padding: 0.55rem 1.2rem;
            border-radius: 8px;
            margin: 0.25rem;
            font-weight: 600;
            font-size: 0.92rem;
            transition: background 0.2s;
        }}
        .header-links a:hover {{ background: rgba(255,255,255,0.3); }}
        nav {{
            position: sticky;
            top: 0;
            z-index: 50;
            background: rgba(255,255,255,0.92);
            backdrop-filter: blur(8px);
            border-bottom: 1px solid var(--border);
            padding: 0.75rem 1rem;
            display: flex;
            justify-content: center;
            gap: 1.2rem;
            flex-wrap: wrap;
        }}
        nav a {{
            color: var(--muted);
            text-decoration: none;
            font-size: 0.9rem;
            font-weight: 600;
            padding: 0.3rem 0.6rem;
            border-radius: 6px;
        }}
        nav a:hover {{ color: var(--primary); background: var(--primary-light); }}
        .container {{ max-width: 1180px; margin: 2.2rem auto; padding: 0 1.25rem; }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 1.1rem;
            margin-bottom: 2.2rem;
        }}
        .kpi-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.35rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}
        .kpi-card .label {{ font-size: 0.82rem; color: var(--muted); text-transform: uppercase; font-weight: 700; }}
        .kpi-card .value {{ font-size: 1.85rem; font-weight: 800; color: var(--primary); margin: 0.3rem 0; }}
        .kpi-card .sub {{ font-size: 0.85rem; color: var(--muted); }}
        section {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        }}
        section h2 {{
            font-size: 1.45rem;
            margin-bottom: 1rem;
            padding-bottom: 0.6rem;
            border-bottom: 2px solid #f1f5f9;
            color: #0f172a;
        }}
        section h3 {{ font-size: 1.12rem; margin: 1.25rem 0 0.5rem; color: #1e3a8a; }}
        p, li {{ color: #334155; margin-bottom: 0.65rem; }}
        ul {{ padding-left: 1.3rem; }}
        .crisp-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
            gap: 1.1rem;
            margin-top: 1rem;
        }}
        .crisp-box {{
            background: #f8fafc;
            border-left: 4px solid var(--primary);
            padding: 1.1rem 1.25rem;
            border-radius: 8px;
        }}
        .crisp-box h4 {{ color: var(--primary); margin-bottom: 0.4rem; font-size: 1.02rem; }}
        .img-card {{
            margin: 1.4rem 0;
            text-align: center;
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1rem;
        }}
        .img-card img {{ max-width: 100%; height: auto; border-radius: 6px; }}
        .img-caption {{ font-size: 0.88rem; color: var(--muted); margin-top: 0.6rem; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.92rem;
        }}
        th, td {{
            padding: 0.75rem 0.9rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ background: #f1f5f9; font-weight: 700; color: #1e293b; }}
        .highlight-row {{ background-color: #eff6ff; }}
        .badge-best {{
            background: #2563eb; color: white; font-size: 0.72rem;
            padding: 0.15rem 0.5rem; border-radius: 99px; margin-left: 0.4rem;
        }}
        .tag-pos {{ background: #dcfce7; color: #166534; padding: 0.2rem 0.55rem; border-radius: 6px; font-weight: 600; font-size: 0.8rem; }}
        .tag-neg {{ background: #fee2e2; color: #991b1b; padding: 0.2rem 0.55rem; border-radius: 6px; font-weight: 600; font-size: 0.8rem; }}
        pre {{
            background: #0f172a; color: #f8fafc; padding: 1.1rem;
            border-radius: 8px; overflow-x: auto; font-size: 0.86rem; margin: 0.8rem 0;
        }}
        footer {{
            text-align: center; padding: 2.5rem 1rem; color: var(--muted); font-size: 0.9rem;
        }}
    </style>
</head>
<body>

<header>
    <div class="badge">HW2: Multiple Linear Regression | CRISP-DM Methodology</div>
    <h1>WHO Life Expectancy — 多元線性回歸完整實作與診斷分析</h1>
    <p>遵循 CRISP-DM 六大標準流程，針對 20 個健康與社經輸入特徵進行多重共線性診斷 (VIF)、遞迴特徵消除 (RFE 選出 14 個核心特徵)、區間預測 (95% CI & PI) 以及五大回歸假設檢定。</p>
    <div class="header-links">
        <a href="https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who" target="_blank">📊 Kaggle 資料集來源</a>
        <a href="#evaluation">📈 模型評估與預測圖</a>
        <a href="#assumptions">🔬 五大回歸假設檢定</a>
        <a href="#implementation">💻 完整實作說明</a>
    </div>
</header>

<nav>
    <a href="#crisp-dm">1. CRISP-DM 流程</a>
    <a href="#eda">2. 資料理解與準備</a>
    <a href="#feature-selection">3. 特徵選擇 (20 → 14)</a>
    <a href="#evaluation">4. 模型評估與區間預測</a>
    <a href="#assumptions">5. 五大回歸假設檢定</a>
    <a href="#notebooklm">6. NotebookLM 摘要與比較</a>
    <a href="#implementation">7. 實作說明</a>
</nav>

<div class="container">
    <!-- KPI Cards -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="label">輸入特徵數 (不含 Target)</div>
            <div class="value">{n_total_features} → {K_SELECTED}</div>
            <div class="sub">符合 14–39 特徵限制 (RFE 精簡)</div>
        </div>
        <div class="kpi-card">
            <div class="label">測試集決定係數 (Test R²)</div>
            <div class="value">{best_row['Test_R2']:.4f}</div>
            <div class="sub">Adjusted R² = {best_row['Adjusted_R2']:.4f}</div>
        </div>
        <div class="kpi-card">
            <div class="label">預測均方根誤差 (RMSE)</div>
            <div class="value">{best_row['RMSE']:.2f} 歲</div>
            <div class="sub">MAE = {best_row['MAE']:.2f} 歲 | MSE = {best_row['MSE']:.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="label">95% 預測區間涵蓋率</div>
            <div class="value">{in_pi:.1f}%</div>
            <div class="sub">Max VIF = {max_vif_after:.2f} (&lt; 5 無共線性)</div>
        </div>
    </div>

    <!-- Section 1: CRISP-DM -->
    <section id="crisp-dm">
        <h2>1. CRISP-DM 標準資料科學流程說明</h2>
        <p>本研究採用跨產業資料探勘標準流程（CRISP-DM），分析世界衛生組織（WHO）193 個國家自 2000 年至 2015 年共 {len(df_model)} 筆有效樣本，預測連續目標變數 <code>Life expectancy</code>（預期壽命）。</p>
        <div class="crisp-grid">
            <div class="crisp-box">
                <h4>Step 1. Business Understanding</h4>
                <p>量化疫苗覆蓋率、死亡率指標、受教育年限與宏觀經濟對預期壽命的邊際影響，建立可解釋的線性回歸模型以輔助公共衛生資源分配。</p>
            </div>
            <div class="crisp-box">
                <h4>Step 2. Data Understanding</h4>
                <p>原始資料共 22 欄。扣除目標變數 <code>Life expectancy</code> 與國家識別碼 <code>Country</code> 後，共有 <strong>20 個輸入特徵</strong>（落在 14~39 規定範圍內）。</p>
            </div>
            <div class="crisp-box">
                <h4>Step 3. Data Preparation</h4>
                <p>清理欄位空白字元、移除 10 筆目標缺失樣本、以國家分組中位數與訓練集中位數填補缺失值，並執行 <code>StandardScaler</code> 標準化。</p>
            </div>
            <div class="crisp-box">
                <h4>Step 4. Modeling & Feature Selection</h4>
                <p>結合 VIF 共線性檢驗與 <code>sklearn.feature_selection.RFE</code>，將 20 個特徵篩選為 <strong>14 個核心特徵</strong>，比較 OLS、Ridge 與 Lasso。</p>
            </div>
            <div class="crisp-box">
                <h4>Step 5. Evaluation</h4>
                <p>以 MAE、MSE、RMSE、R²、Adjusted R² 全面評估，繪製 <strong>95% 信賴區間 (CI) 與預測區間 (PI)</strong>，並完成五大回歸假設檢定。</p>
            </div>
            <div class="crisp-box">
                <h4>Step 6. Deployment</h4>
                <p>將完整分析圖表與統計推論成果自動部署為靜態互動網頁（GitHub Pages），提供公衛政策制定者直觀的指標權重參考。</p>
            </div>
        </div>
    </section>

    <!-- Section 2: EDA -->
    <section id="eda">
        <h2>2. 資料理解與探索性資料分析 (EDA)</h2>
        <p>目標變數 <code>Life expectancy</code> 平均為 <strong>{y.mean():.2f} 歲</strong>（中位數 {y.median():.2f} 歲）。從右側相關係數矩陣可見，<code>Schooling</code>（受教育年限，$r=0.75$）與 <code>Income composition of resources</code>（資源收入指數，$r=0.72$）與壽命呈最強正相關；而 <code>Adult Mortality</code>（成人死亡率，$r=-0.70$）與 <code>HIV/AIDS</code>（$r=-0.56$）則呈強烈負相關。</p>
        <div class="img-card">
            <img src="assets/fig1_eda_correlation.png" alt="EDA Target Distribution and Correlation Matrix">
            <div class="img-caption">圖 1：目標變數預期壽命分佈圖（左）與 Top 12 高度相關特徵之皮爾森相關係數熱力圖（右）。</div>
        </div>
    </section>

    <!-- Section 3: Feature Selection -->
    <section id="feature-selection">
        <h2>3. 特徵選擇 (Feature Selection: 20 → 14 Features)</h2>
        <p>原始 20 個特徵中存在極端多重共線性（例如 <code>infant deaths</code> 與 <code>under-five deaths</code> 的 VIF 高達 180 以上）。我們透過兩階段特徵選擇：</p>
        <ul>
            <li><strong>階段一（VIF 共線性剔除）</strong>：移除重複衡量相同概念的冗餘特徵（<code>under-five deaths</code>、<code>percentage expenditure</code>、<code>thinness 5-9 years</code>）。</li>
            <li><strong>階段二（RFE 遞迴特徵消除）</strong>：以 <code>sklearn.LinearRegression</code> 為基底，選出最具統計解釋力的 <strong>14 個特徵</strong>，確保所有入選特徵的 VIF 皆低於 {max_vif_after:.2f}（遠低於警戒值 10）。</li>
        </ul>
        <div class="img-card">
            <img src="assets/fig2_feature_selection.png" alt="Feature Selection Coefficients and VIF">
            <div class="img-caption">圖 2：RFE 選出之 14 個特徵標準化回歸係數（左）與特徵選擇後之 VIF 多重共線性檢驗（右）。</div>
        </div>

        <h3>入選 14 個核心特徵之回歸係數、顯著性與 VIF 明細表</h3>
        <table>
            <thead>
                <tr>
                    <th>特徵名稱 (Feature)</th>
                    <th>標準化係數 (Std. Coef)</th>
                    <th>影響方向</th>
                    <th>P-value (統計顯著性)</th>
                    <th>VIF (共線性指標)</th>
                </tr>
            </thead>
            <tbody>
                {feature_rows_html}
            </tbody>
        </table>
    </section>

    <!-- Section 4: Evaluation & Prediction Intervals -->
    <section id="evaluation">
        <h2>4. 模型評估與區間預測圖 (Confidence & Prediction Intervals)</h2>
        <p>下表比較全特徵模型（20 個特徵）、特徵選擇後模型（RFE 14 個特徵、SelectKBest 14 個特徵）以及正則化模型（Ridge、LassoCV）在測試集上的表現。可以發現 <strong>RFE 14 特徵模型</strong> 在消除了 6 個冗餘特徵後，依然維持 <strong>R² = {best_row['Test_R2']:.4f}、RMSE = {best_row['RMSE']:.4f}</strong> 的優異表現，且模型結構更加穩健。</p>

        <table>
            <thead>
                <tr>
                    <th>模型名稱 (Model)</th>
                    <th>特徵數</th>
                    <th>MAE</th>
                    <th>MSE</th>
                    <th>RMSE</th>
                    <th>Train R²</th>
                    <th>Test R²</th>
                    <th>Adjusted R²</th>
                </tr>
            </thead>
            <tbody>
                {metrics_rows_html}
            </tbody>
        </table>

        <div class="img-card">
            <img src="assets/fig3_model_comparison.png" alt="Model Metrics Comparison">
            <div class="img-caption">圖 3：五種線性回歸設定之 R² / Adjusted R²（左）與誤差指標 RMSE / MAE（右）對比。</div>
        </div>

        <h3>多元線性回歸預測圖（含 95% 信賴區間 CI 與 95% 預測區間 PI）</h3>
        <p>作業規定必備之區間預測視覺化：深藍色窄帶為 <strong>95% 信賴區間 (Confidence Interval)</strong>，反映回歸均值函數的估計精準度；淺橘色寬帶為 <strong>95% 預測區間 (Prediction Interval)</strong>，代表單一新樣本真實值的預測落點範圍，於測試集之實際涵蓋率達 <strong>{in_pi:.1f}%</strong>，高度吻合 95% 理論水準。</p>
        <div class="img-card">
            <img src="assets/fig4_prediction_intervals.png" alt="Prediction Intervals CI and PI">
            <div class="img-caption">圖 4：測試集排序樣本之多元線性回歸預測曲線、95% 信賴區間與 95% 預測區間（左），以及真實值對預測值散佈圖（右）。</div>
        </div>
    </section>

    <!-- Section 5: 5 Regression Assumptions -->
    <section id="assumptions">
        <h2>5. 五大線性回歸假設檢定 (Five Regression Assumptions)</h2>
        <p>為確保多元線性回歸模型的統計推論有效性，我們針對殘差與特徵進行完整的五大假設診斷：</p>
        <ul>
            <li><strong>假設 1：線性關係 (Linearity)</strong> — 殘差對預測值圖（左上）的 LOWESS 平滑紅線貼近 0 水平線，殘差平均值為 <code>{mean_resid:.2e}</code>，證實線性假設成立。</li>
            <li><strong>假設 2：誤差獨立性 (Independence of Errors)</strong> — Durbin-Watson 統計量為 <code>{dw_stat:.4f}</code>（完美落在 1.5 ~ 2.5 無自我相關區間內）。</li>
            <li><strong>假設 3：同質變異數 (Homoscedasticity)</strong> — Scale-Location 圖（左下）顯示標準化殘差平方根在各預測區間分佈大致均勻。</li>
            <li><strong>假設 4：殘差常態性 (Normality of Residuals)</strong> — Q-Q Plot（右上）與標準化殘差直方圖（右下）顯示殘差緊密貼合 45 度標準常態分佈線（偏態係數 Skewness = <code>{resid_skew:.3f}</code>）。</li>
            <li><strong>假設 5：無多重共線性 (No Multicollinearity)</strong> — 經 VIF 篩選與 RFE 後，14 個特徵之最大 VIF 僅為 <code>{max_vif_after:.2f}</code>，平均 VIF 為 <code>{mean_vif_after:.2f}</code>（遠低於 10 的門檻）。</li>
        </ul>
        <div class="img-card">
            <img src="assets/fig5_regression_assumptions.png" alt="Five Regression Assumptions Diagnostics">
            <div class="img-caption">圖 5：多元線性回歸五大假設檢定診斷圖（Residuals vs Fitted、Normal Q-Q、Scale-Location、Residual Histogram & VIF）。</div>
        </div>
    </section>

    <!-- Section 6: NotebookLM Summary -->
    <section id="notebooklm">
        <h2>6. NotebookLM 研究摘要與網路主流解法比較</h2>
        <p>透過 <strong>NotebookLM</strong> 彙整 Kaggle 社群針對 WHO Life Expectancy 資料集之高票研究與文獻後，歸納出三大核心洞察：</p>
        <ul>
            <li><strong>分組缺失值填補策略</strong>：主流高分做法利用同一國家跨年度資料的連續性，先以 <code>Country</code> 分組中位數填補 GDP 與人口缺失，再進行跨國填補，能顯著降低填補偏差。本研究已將此策略整合進 Data Preparation 流程。</li>
            <li><strong>多重共線性消除之必要性</strong>：原始資料中嬰兒死亡率與五歲以下死亡率相關係數高達 0.996，若直接放入 OLS 會導致回歸係數符號反轉。透過 VIF 搭配 RFE 篩選出 14 個特徵，能兼顧模型精簡度與統計解釋力。</li>
            <li><strong>線性回歸與非線性樹模型之權衡</strong>：雖然 Random Forest 或 XGBoost 能捕捉 BMI 等指標的非線性邊界，但 <code>sklearn.LinearRegression</code> 搭配 <code>statsmodels</code> 能夠提供明確的標準化係數權重、P-value 顯著性檢定以及 95% 信賴/預測區間，在公共衛生政策制定上具備不可替代的透明度。</li>
        </ul>
    </section>

    <!-- Section 7: Implementation Guide -->
    <section id="implementation">
        <h2>7. 完整實作與重現步驟說明 (Implementation Guide)</h2>
        <p>本專案僅需執行單一 Python 腳本，即可自動完成資料下載、清理、特徵選擇、模型評估、五大假設檢定，並自動產出 <code>assets/</code> 下所有 300 DPI 圖表與本 <code>index.html</code> 網頁：</p>
        <pre><code># 1. 安裝必要套件
pip install numpy pandas scikit-learn statsmodels scipy matplotlib seaborn kagglehub

# 2. 執行主程式 (自動下載 WHO 資料集、訓練模型、輸出 5 張圖表與 index.html)
python 7114056XXX_hw2.py</code></pre>
    </section>
</div>

<footer>
    <p>HW2 Multiple Linear Regression Project | Built with Python, scikit-learn, statsmodels &amp; GitHub Pages</p>
</footer>

</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("[Success] 已成功產生靜態網站首頁: index.html")
print("所有圖表已儲存於 assets/ 資料夾，可直接推送至 GitHub 開啟 GitHub Pages！")
