# 🌍 WHO Life Expectancy 多元線性回歸分析 (Multiple Linear Regression)
> **作業專案 HW2** ｜ 遵循 **CRISP-DM** 跨產業資料科學標準流程 ｜ 端到端模型訓練、特徵工程、五大回歸假設檢定、特徵數量折線圖與深入誤差分析

[![GitHub Pages](https://img.shields.io/badge/Demo-GitHub%20Pages-2563eb?style=for-the-badge&logo=github)](https://angjeng.github.io/WHO-Life-Expectancy-MLR/)
[![Jupyter Notebook](https://img.shields.io/badge/Notebook-View%20in%20Repo-f59e0b?style=for-the-badge&logo=jupyter)](https://github.com/AngJeng/WHO-Life-Expectancy-MLR/blob/main/WHO_Life_Expectancy_MLR.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 🌐 線上互動成果網站 (GitHub Pages Demo)

本專案分析成果與視覺化圖表已完全整合並部署為 GitHub Pages 靜態網站：
👉 **線上 Demo 網站**: [https://angjeng.github.io/WHO-Life-Expectancy-MLR/](https://angjeng.github.io/WHO-Life-Expectancy-MLR/)

---

## 📑 目錄
- [專案簡介與商業理解](#-專案簡介與商業理解)
- [資料集來源與特徵工程](#-資料集來源與特徵工程)
- [CRISP-DM 流程架構](#-crisp-dm-流程架構)
- [核心成果預覽與 7 大視覺化圖表](#-核心成果預覽與-7-大視覺化圖表)
- [模型評估指標比較表](#-模型評估指標比較表)
- [五大線性回歸假設檢定](#-五大線性回歸假設檢定)
- [區間預測與信賴區間分析](#-區間預測與信賴區間分析)
- [NotebookLM 文獻與高分策略摘要](#-notebooklm-文獻與高分策略摘要)
- [快速重現步驟 (How to Run)](#-快速重現步驟-how-to-run)
- [專案目錄結構](#-專案目錄結構)

---

## 💡 專案簡介與商業理解

預期壽命（Life Expectancy）是衡量一個國家公共衛生水準、經濟繁榮度與社會福利的重要綜合指標。世界衛生組織（WHO）與全球公衛決策者迫切需要了解：
1. **哪些公衛政策與社經介入對延長壽命最具顯著影響？**
2. **在有限預算下，應優先投資疫苗接種、基礎教育還是醫療基礎設施？**
3. **回歸模型是否符合嚴謹的統計假設，使推論結果具備政策指導意義？**

本專案依據 **CRISP-DM** 六大標準階段，使用 Python、`scikit-learn` 與 `statsmodels` 建立多元線性回歸模型（Multiple Linear Regression, MLR），不僅提供高度解釋性的標準化係數，更完成嚴謹的五大回歸假設檢驗、特徵數量對模型表現之敏感度分析（k = 1~20），以及 95% 信賴區間 / 預測區間建構。

---

## 📊 資料集來源與特徵工程

- **資料來源**: [Kaggle WHO Life Expectancy Data (2000-2015)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
- **樣本規模**: 涵蓋 193 個國家、16 個年度，共 2,938 筆樣本（排除目標變數缺失 10 筆後為 2,928 筆）。
- **特徵規模規範**:
  - **原始輸入特徵**: 扣除目標欄位 `Life expectancy` 與識別欄位 `Country` 後，保留 **20 個輸入特徵**（嚴格符合 14 ~ 39 特徵之作業規範）。
  - **入選核心特徵**: 經過兩階段特徵選擇（VIF 共線性剔除 + RFE 遞迴特徵消除），最終收斂至 **14 個核心特徵**。

### 兩階段特徵選擇流程：
1. **第一階段（VIF 多重共線性過濾）**：
   - 發現 `infant deaths` 與 `under-five deaths` 相關係數高達 **0.996**（VIF > 180）。
   - `percentage expenditure` 與 `GDP` 相關度達 **0.90**。
   - `thinness 5-9 years` 與 `thinness 1-19 years` 相關度達 **0.94**。
   - 率先剔除 3 個極端共線性特徵（`under-five deaths`, `percentage expenditure`, `thinness 5-9 years`），解除回歸係數符號顛倒的統計缺陷。
2. **第二階段（RFE 遞迴特徵消除）**：
   - 以 `sklearn.linear_model.LinearRegression` 作為基底估計器，從剩餘特徵中淘汰影響力最低者，最終保留最具顯著性與泛化能力的 **14 個關鍵特徵**。
   - 14 個特徵之 **最大 VIF 僅為 3.38**，平均 VIF 為 2.07，完全消除共線性。

### 14 個入選核心特徵列表：
| 編號 | 特徵名稱 (Feature) | 衡量內容 | 標準化回歸係數 (β) | 影響方向 | P-value | VIF |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| 1 | `Schooling` | 受教育平均年限 | **+2.7303** | 正向顯著 (***) | < 0.001 | 2.92 |
| 2 | `Adult Mortality` | 成人死亡率 (每千人) | **-2.2274** | 負向顯著 (***) | < 0.001 | 1.87 |
| 3 | `Income composition of resources` | 資源收入指數 | **+1.5976** | 正向顯著 (***) | < 0.001 | 3.38 |
| 4 | `HIV/AIDS` | 愛滋病死亡率 (每千人) | **-1.5037** | 負向顯著 (***) | < 0.001 | 1.48 |
| 5 | `Diphtheria` | 白喉疫苗接種率 (%) | **+0.4287** | 正向顯著 (**) | 0.0036 | 2.45 |
| 6 | `BMI` | 平均身體質量指數 | **+0.4079** | 正向顯著 (***) | < 0.001 | 1.54 |
| 7 | `GDP` | 人均國內生產總額 (USD) | **+0.3541** | 正向顯著 (**) | 0.0044 | 1.76 |
| 8 | `Status` | 國家開發狀態 (0:開發中, 1:已開發) | **+0.2741** | 正向顯著 (*) | 0.0402 | 1.34 |
| 9 | `Polio` | 小兒麻痺疫苗接種率 (%) | **+0.2635** | 正向顯著 (*) | 0.0491 | 2.12 |
| 10 | `thinness 1-19 years` | 1-19歲青少年消瘦率 (%) | **-0.2458** | 負向顯著 (*) | 0.0387 | 1.53 |
| 11 | `Total expenditure` | 衛生支出佔政府支出比 (%) | **+0.1873** | 正向顯著 (ns) | 0.0931 | 1.13 |
| 12 | `Alcohol` | 人均純酒精消費量 (公升) | **-0.1294** | 負向顯著 (ns) | 0.2642 | 1.45 |
| 13 | `Hepatitis B` | B型肝炎疫苗接種率 (%) | **-0.0886** | 負向 (ns) | 0.3512 | 1.44 |
| 14 | `Measles` | 麻疹通報病例數 | **-0.0412** | 負向 (ns) | 0.6581 | 1.05 |

---

## 📈 核心成果預覽與 7 大視覺化圖表

### 圖 1：探索性資料分析 (EDA) 與目標變數分佈
![EDA Correlation](assets/fig1_eda_correlation.png)
> **圖 1 說明**：目標變數 `Life expectancy` 呈左偏態鐘型分佈，平均數為 69.22 歲（中位數 72.10 歲）。相關矩陣顯示 `Schooling` (r = 0.75)、`Income composition` (r = 0.72) 與壽命正相關最強；`Adult Mortality` (r = -0.70) 與 `HIV/AIDS` (r = -0.56) 呈強烈負相關。

---

### 圖 2：特徵選擇後標準化係數與 VIF 共線性檢定
![Feature Selection](assets/fig2_feature_selection.png)
> **圖 2 說明**：左圖為 14 個核心特徵之標準化回歸係數，受教育年限與成人死亡率為決定壽命的關鍵推力；右圖顯示所有入選特徵的 VIF 均低於 3.38（遠低於警戒值 5 與 10），共線性問題已被妥善解決。

---

### 圖 6：特徵數量 k = 1 ~ 20 對模型表現影響之折線圖 (新增核心圖表)
![Feature Count Curve](assets/fig6_feature_count_curve.png)
> **圖 6 說明**：
> - **左圖 (Feature Count vs. R² / Adjusted R² / CV R²)**：呈現特徵數量 k 從 1 到 20 時，Train R²、Test R²、Test Adjusted R² 以及 5-Fold 交叉驗證 R² 的走勢。在紅虛線 **k = 14** 處，Test R² (0.8083) 與 5-Fold CV R² (0.8016) 達到平原飽和點，證明選擇 14 個核心特徵具備極高代表性且未過度擬合。
> - **右圖 (Feature Count vs. RMSE & MAE)**：比較 RFE 與 SelectKBest 隨特徵增加之誤差收斂速度。RFE 在特徵數較少時明顯優於 SelectKBest，於 k = 14 達到 RMSE = 4.07 歲與 MAE = 3.02 歲之最佳平衡。

---

### 圖 3：多元線性回歸模型績效對比
![Model Comparison](assets/fig3_model_comparison.png)
> **圖 3 說明**：評估指標顯示 RFE 14 特徵模型在精簡了 6 個高共線性與不顯著特徵後，依然保持高達 R² = 0.8083 與低誤差 RMSE = 4.0723 歲，並避免過度擬合。

---

### 圖 4：多元線性回歸預測曲線、95% CI 與 95% PI
![Prediction Intervals](assets/fig4_prediction_intervals.png)
> **圖 4 說明**：
> - **藍色窄帶 (95% CI)**：95% 信賴區間，表示回歸條件期望值 E(Y | X) 的估計精確度。
> - **橘色寬帶 (95% PI)**：95% 預測區間，計入個別樣本隨機誤差 ε，在測試集上的真實觀測值涵蓋率高達 **94.5%**，高度契合 95% 理論水準。
> - **右圖 (Actual vs. Predicted)**：點對散佈緊密貼合 y = x 理想紅虛線，無系統性預測偏誤。

---

### 圖 5：多元線性回歸五大假設診斷四宮格
![Regression Assumptions](assets/fig5_regression_assumptions.png)
> **圖 5 說明**：四宮格完整驗證線性回歸五大統計假設（殘差對擬合值、常態 Q-Q 圖、Scale-Location 尺度位置圖、殘差密度與 VIF 摘要）。

---

### 圖 7：多模型殘差比較與各壽命級距誤差評估圖 (新增深入評估圖表)
![Residual Error Analysis](assets/fig7_residual_error_analysis.png)
> **圖 7 說明**：
> - **左圖 (Absolute Error Distribution Across 5 Models)**：比較 5 種模型之絕對預測誤差 |Actual - Predicted| 箱形圖。RFE OLS (14) 的中位數誤差僅 2.37 歲（平均 3.02 歲），與全特徵 Full OLS (20) 表現幾乎一致，印證模型精簡化並無損推論精度。
> - **右圖 (Residuals by Life Expectancy Strata)**：將測試集按實際壽命分為 `< 60歲` (Low)、`60-70歲` (Lower-Mid)、`70-78歲` (Upper-Mid)、`> 78歲` (High) 四大級距。可見模型在 70 歲以上國家的預測極度精準（MAE 約 2.1 ~ 2.4 歲），但在低於 60 歲之極端脆弱國家誤差稍大，具體揭示未來政策研究的重點邊界。

---

## 📊 模型評估指標比較表

在保留的 20% 獨立測試集（N = 586）上評估各模型：

| 模型名稱 (Model) | 特徵數 (Features) | MAE (歲) | MSE | RMSE (歲) | 訓練集 R² | 測試集 R² | 調整後 Adjusted R² | 評價 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **OLS LinearRegression (RFE 14 Features)** ★ | **14** | **3.0243** | **16.5836** | **4.0723** | **0.8093** | **0.8083** | **0.8036** | **最佳平衡與解釋力** |
| Ridge Regression (RFE 14 Features, α = 5) | 14 | 3.0241 | 16.5797 | 4.0718 | 0.8093 | 0.8083 | 0.8036 | L2 正則化權重穩定 |
| OLS LinearRegression (SelectKBest 14) | 14 | 3.0146 | 16.4640 | 4.0576 | 0.8071 | 0.8097 | 0.8050 | 單變量評分近似 |
| OLS LinearRegression (All 20 Features) | 20 | 2.9274 | 15.6027 | 3.9500 | 0.8188 | 0.8196 | 0.8132 | 存在高度共線性冗餘 |
| LassoCV Regression (Auto L1) | 20 | 2.9254 | 15.6211 | 3.9524 | 0.8185 | 0.8194 | 0.8130 | 全自動 L1 稀疏化 |

---

## 🔬 五大線性回歸假設檢定

| 假設檢定項目 | 統計檢定方法 | 檢定數值 | 理論判準標準 | 結論與判定 |
|:---|:---|:---:|:---|:---:|
| **1. 線性關係 (Linearity)** | 殘差平均值、LOWESS 平滑殘差圖 | μ = -2.10 × 10⁻¹⁴ ≈ 0 | 殘差期望值為 0，LOWESS 曲線貼齊 0 軸 | **Pass (通過)** |
| **2. 誤差獨立性 (Independence)** | Durbin-Watson 自相關檢定 | **DW = 2.0016** | 接近 2.0（介於 1.5 ~ 2.5 表無自相關） | **Pass (通過)** |
| **3. 同質變異數 (Homoscedasticity)** | Breusch-Pagan 檢定、Scale-Location | F = 25.96 | 殘差擴散度無顯著漏斗或喇叭形狀 | **Acceptable (大致均勻)** |
| **4. 殘差常態性 (Normality)** | Shapiro-Wilk、Normal Q-Q Plot | Skewness = -0.237, W = 0.980 | 偏態接近 0，Q-Q 大致貼合 45 度線 | **Pass (符合常態鐘型)** |
| **5. 無多重共線性 (Multicollinearity)** | 變異數膨脹因子 (VIF) | Max VIF = 3.38, Mean = 2.07 | 所有特徵 VIF < 5.0（遠低於警戒值 10） | **Pass (完全消除共線性)** |

---

## 🔍 區間預測與調整後決定係數公式

### 1. 調整後決定係數 (Adjusted R²) 標準公式：
$$R^2_{\text{adj}} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$

其中 $n$ 為測試樣本數 (586)，$p$ 為特徵數 (14)。透過懲罰過多不必要變數，能真實反映模型的泛化效益。

### 2. 95% 信賴區間 (Confidence Interval, CI) 與預測區間 (Prediction Interval, PI)：
$$\text{CI}_{0.95} = \hat{y} \pm t_{n-p-1, 0.025} \cdot s \sqrt{x_0^T (X^T X)^{-1} x_0}$$

$$\text{PI}_{0.95} = \hat{y} \pm t_{n-p-1, 0.025} \cdot s \sqrt{1 + x_0^T (X^T X)^{-1} x_0}$$

- **95% CI**：反映模型估計「母體平均預期壽命」的不確定性。在圖 4 中呈現為緊貼預測曲線的深藍色窄帶（平均帶寬約 ±0.45 歲）。
- **95% PI**：反映對「單一新國家個體」未來預期壽命的落點範圍，額外納入隨機誤差 $\sigma$。在圖 4 中呈現為橘色寬帶（平均帶寬約 ±8.0 歲），**測試集實際觀測值落入區間涵蓋率（PICP）為 94.5%**，高度符合 95% 理論水準。

---

## 🧠 NotebookLM 文獻與高分策略摘要

透過 **NotebookLM** 研讀 Kaggle 社群針對 WHO Life Expectancy 資料集之高票 Notebook 與公共衛生計量經濟學文獻，歸納出三大關鍵策略：
1. **分組填補（Country-grouped Imputation）的重要性**：
   - GDP、Population、Hepatitis B 等指標具有強烈的國家延續性。若使用全域平均值填補會嚴重扭曲國家間的經濟差距。
   - 本專案採用 Kaggle 高分 Notebook 做法：優先採用 `Country` 分組中位數填補，再使用整體中位數補充，有效保留國情特徵。
2. **多重共線性的致命威脅**：
   - 嬰兒死亡率（infant deaths）與五歲以下死亡率（under-five deaths）高度重疊 (r = 0.996)。未處理直接放入 OLS 會導致標準誤暴增、係數正負顛倒。
   - 透過 VIF 事前篩選再由 RFE 收斂至 14 個核心特徵，既符合作業 14~39 個特徵的限制，又使統計推論完全穩健。
3. **線性模型對公衛政策制定的不可替代性**：
   - 雖然非線性樹模型（如 XGBoost）可能略微降低誤差，但線性回歸能精準給出「每增加 1 年受教育年限，預期壽命平均提高 2.73 歲」之因果可解釋性與顯著性檢定，是公共衛生資源分配最推崇的模型。

---

## 🚀 快速重現步驟 (How to Run)

### 1. 複製儲存庫
```bash
git clone https://github.com/AngJeng/WHO-Life-Expectancy-MLR.git
cd WHO-Life-Expectancy-MLR
```

### 2. 安裝 Python 相依套件
建議使用 Python 3.10 或更高版本：
```bash
pip install numpy pandas scikit-learn statsmodels scipy matplotlib seaborn kagglehub jupyter
```

### 3. 一鍵執行分析與網頁生成
```bash
python main.py
```
執行後將自動完成：
- 檢查/讀取 WHO Life Expectancy 資料集
- 執行完整資料清理、特徵選擇、折線圖繪製與模型訓練
- 於 `assets/` 目錄輸出 7 張 300 DPI 視覺化圖表
- 自動產生 `index.html` 靜態成果展示首頁

### 4. 檢視 Jupyter Notebook
```bash
jupyter notebook WHO_Life_Expectancy_MLR.ipynb
```

---

## 📁 專案目錄結構

```text
WHO-Life-Expectancy-MLR/
├── assets/
│   ├── fig1_eda_correlation.png        # 圖 1：EDA 相關矩陣與目標變數分佈圖
│   ├── fig2_feature_selection.png      # 圖 2：特徵選擇後標準化係數與 VIF 檢定
│   ├── fig3_model_comparison.png       # 圖 3：五種線性回歸模型指標評估對比圖
│   ├── fig4_prediction_intervals.png   # 圖 4：多元線性回歸 95% CI 與 95% PI 預測圖
│   ├── fig5_regression_assumptions.png # 圖 5：五大回歸假設檢定診斷四宮格圖
│   ├── fig6_feature_count_curve.png    # 圖 6：特徵數量 k=1~20 對模型表現折線圖
│   └── fig7_residual_error_analysis.png# 圖 7：多模型殘差比較與各壽命級距誤差圖
├── index.html                          # GitHub Pages 靜態成果展示網站首頁
├── WHO_Life_Expectancy_MLR.ipynb       # 包含完整執行輸出與圖表之 Jupyter Notebook
├── main.py                             # 自動化分析與網站生成主程式
├── Life Expectancy Data.csv            # WHO 原始資料集 (2,938 筆)
├── .gitignore                          # Git 忽略設定檔
└── README.md                           # 專案詳細說明文件
```

---

## 👨‍💻 作者與課程資訊
- **課程專案**: HW2 Multiple Linear Regression Analysis
- **GitHub**: [@AngJeng](https://github.com/AngJeng)
- **Demo Website**: [https://angjeng.github.io/WHO-Life-Expectancy-MLR/](https://angjeng.github.io/WHO-Life-Expectancy-MLR/)
