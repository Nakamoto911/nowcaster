Here is the raw Markdown content for the specifications. You can copy the block below.

```markdown
# Technical Specifications: US Economic Regime Nowcaster (The "Teacher" Model)

## 1. Project Overview

**Objective:** To construct a "Ground Truth" regime classifier that identifies the current state of the US Economy in continuous probabilistic terms.
**Role:** This model acts as the "Teacher" (Target Generator) for future predictive models.
**Output format:** A monthly vector of four probabilities summing to 1.0 (e.g., `[0.1, 0.0, 0.2, 0.7]`).

## 2. Data Architecture

### 2.1 Source Data

* **Database:** FRED-MD (Monthly Database for Macroeconomic Research).
* **Frequency:** Monthly.
* **Range:** 1960-01-01 to Present.

### 2.2 Feature Selection (The "Real Economy" Constraint)

To ensure the model measures *economic state* rather than *market sentiment*, strict exclusion criteria are applied.

* **INCLUDED Groups:**
  * **Strategy:** Use **ALL** series within these groups to leverage PCA's ability to extract common signals from redundant data.
  * Group 1: Output & Income (e.g., `INDPRO`, `IPMANSICS`)
  * Group 2: Labor Market (e.g., `PAYEMS`, `UNRATE`)
  * Group 3: Housing (e.g., `HOUST`, `PERMIT`) - *Primary Leading Indicator*
  * Group 4: Consumption, Orders, & Inventories (e.g., `RSAFS`)
  * Group 5: Money & Credit (e.g., `BUSLOANS`) - *Selected for credit activity*
  * Group 7: Prices (e.g., `CPIAUCSL`, `PCEPI`) - *Required for Inflation Axis*

* **EXCLUDED Groups:**
  * Group 6: Interest & Exchange Rates (Yields, Spreads, FX).
  * Group 8: Stock Market (S&P 500, VIX).

### 2.3 Training vs. Testing Split

* **Training Set:** 1960-01-01 to 2019-12-31.
* **Rationale:** The COVID-19 shock (2020-2021) contains 10-sigma moves that would skew the normalization (StandardScaler) if included in the fitting process.
* **Testing/Live Set:** 2020-01-01 to Present (Out-of-Sample stress test).

### 2.4 Missing Data Strategy

PCA requires a complete matrix (no NaNs). The following rules apply strictly:

1. **Historical Gaps (Columns):** Any series (column) that does not have data extending back to **1960-01-01** is dropped entirely. We do not impute historical history.
2. **Intermediate Gaps (Cells):** Small gaps (e.g., missing 1-2 months in the middle of the series) are filled via **Linear Interpolation**. Limit: Max 2 consecutive months.
3. **Ragged Edge (Rows):** If the most recent month (row) is incomplete (e.g., waiting for data release), that **row is dropped**. We do not "nowcast" the inputs for the Teacher model; we only classify complete months.

## 3. Transformation Pipeline (Strict Order)

The data must flow through these operations in this exact order to avoid look-ahead bias and non-stationarity.

1. **Stationarity Transformation (t-codes):**
   * Apply standard FRED-MD transformations (log-diff, double-diff) to convert raw levels into growth rates.

2. **Outlier Clipping (Winsorization):**
   * Clip values at the 0.5% and 99.5% quantiles to prevent single data errors from crushing the variance.

3. **Smoothing (Noise Reduction):**
   * **Method:** Simple Moving Average (SMA).
   * **Window:** 3 Months (Rolling).
   * **Target:** `df.rolling(3).mean()`.
   * *Note: This is applied to the growth rates, not the raw levels.*

4. **Normalization (Z-Scoring):**
   * **Method:** `StandardScaler` (Mean=0, Std=1).
   * **Constraint:** `fit()` is called ONLY on the Pre-2020 data. `transform()` is called on the full dataset.

## 4. Model Architecture

### 4.1 Dimensionality Reduction

* **Algorithm:** Principal Component Analysis (PCA).
* **Components:** `n_components = 2` (Fixed).
* **Semantic Goal:**
  * **PC1:** Proxy for "Real Economic Growth" (Loadings should be high on IndPro, Payrolls).
  * **PC2:** Proxy for "Inflation" (Loadings should be high on CPI, PCE).

### 4.2 Classification Algorithm

* **Algorithm:** Gaussian Mixture Model (GMM).
* **Components (Regimes):** `n_components = 4`.
* **Covariance Type:** `full` (Allows clusters to have different shapes/correlations).
* **Initialization:** `n_init = 10` (To avoid local minima).

### 4.3 Semantic Mapping (The Labeling Logic)

Since GMM is unsupervised, the output labels (0,1,2,3) are random. We must map them to economic quadrants using the PCA Centroids:

| **Regime Name** | **Growth (PC1)** | **Inflation (PC2)** | **Description** |
| :--- | :--- | :--- | :--- |
| **Recovery / Goldilocks** | High (+) | Low (-) | Growth accelerating, Inflation falling. |
| **Overheating / Expansion** | High (+) | High (+) | Boom times, inflationary pressure. |
| **Stagflation** | Low (-) | High (+) | The "worst" quadrant. Slow growth, high prices. |
| **Contraction / Deflation** | Low (-) | Low (-) | Recessionary crash. |

*Note: The signs (+/-) depend on the PCA polarity, which can flip. The mapping logic must check the loadings magnitude.*

## 5. Acceptance Criteria

The model is considered "Valid" if:

1. **Variance Check:** The first 2 Principal Components explain at least **40%** of the total variance in the smoothed dataset.

2. **NBER Alignment:** The "Contraction" probability exceeds 50% during:
   * 2008-2009 (GFC)
   * 2001 (Dotcom)
   * 1990

3. **COVID Reaction:** The "Contraction" probability spikes to >90% in March/April 2020 (despite not training on it).

4. **Stagflation Detection:** It identifies the 1970s and 2022 as high "Stagflation" probability periods, distinct from the 2008 deflationary crash.

## 6. Visualization Requirements

To validate the "Teacher" model, the following plots are required:

### 6.1 Regime Probability Time Series (Primary View)

* **Type:** Multi-line Time Series with Scatter overlay.
* **X-Axis:** Date (1960 - Present).
* **Y-Axis:** Probability (0.0 to 1.0).
* **Components:**
  * Four distinct colored lines representing the 4 regimes (Recovery, Expansion, Stagflation, Contraction).
  * **Markers:** A dot for each month on each line, where the **size** of the dot is proportional to the probability (visual emphasis on high-confidence months).
  * **Overlay:** Shaded grey vertical bands representing official **NBER Recession** periods.
* **Goal:** Visually confirm that the "Contraction" line spikes inside the grey NBER bands and the "Stagflation" line spikes during the 1970s.

### 6.2 PCA Phase Diagram (Semantic Validation)

* **Type:** 2D Scatter Plot.
* **X-Axis:** PC1 (Growth).
* **Y-Axis:** PC2 (Inflation).
* **Coloring:** Each point colored by its assigned GMM Cluster (Hard assignment: max probability).
* **Goal:** Verify the "4-Quadrant" logic defined in Section 4.3.
```