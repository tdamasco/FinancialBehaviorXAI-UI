# FinancialBehaviorXAI — UI Documentation

**Application:** FinancialBehaviorXAI Financial Health Analyzer  
**Framework:** Streamlit  
**Entry Point:** `app.py`  
**Deployed Repo:** [tdamasco/FinancialBehaviorXAI-UI](https://github.com/tdamasco/FinancialBehaviorXAI-UI)

---

## Overview

FinancialBehaviorXAI is a single-page Streamlit dashboard that brings together the project's full machine learning pipeline into an interface accessible to everyday users. A user enters their monthly spending across 14 categories and their monthly income. The application runs that input through the trained model, then displays a prediction of whether the user is likely to overspend next month — alongside a suite of explainability and insight tools that make clear *why* that prediction was made.

The interface is intentionally model-agnostic from the user's perspective. No model names, metrics, or technical output appear anywhere in the UI. All ML work runs silently in the background.

---

## How the UI Connects to the Repository

### Data — `data/v2/model_ready.csv`

This is the fully engineered, target-labeled dataset produced by the `newDataTestMain.ipynb` pipeline. It contains 6,366 individual-month records across 903 synthetic individuals from the Sparkov dataset, with 71 columns covering raw spending, engineered behavioral features, and the binary overspending target (`overspend_target_t1`).

The UI uses this file for two purposes:
1. **Model training** — the dataset is split using a time-aware train/test split and the model is fitted on the training portion at startup.
2. **Peer benchmarking** — the spending columns and income column are retained as a reference table to dynamically compute income-bracket averages for comparison.

### Scripts — `src/newDataScripts/`

The UI imports two functions directly from this directory:

| Script | Function Used | Purpose in UI |
|---|---|---|
| `modeling.py` | `build_preprocessor(X)` | Constructs the sklearn `ColumnTransformer` (median imputation + standard scaling for numeric columns, most-frequent imputation + one-hot encoding for categoricals) used inside the prediction pipeline |
| `modeling.py` | `time_aware_train_test_split(df)` | Sorts the dataset chronologically and splits 80% train / 20% test, ensuring no temporal data leakage when fitting the model at startup |

The `sys.path.insert` at the top of `app.py` adds `src/newDataScripts/` to the Python path so these imports resolve correctly in both local and cloud environments.

### Model

The UI trains and uses a **Gradient Boosting Classifier** with parameters taken from the Optuna hyperparameter tuning results reported in the paper (Section 7-E):

```
n_estimators = 190
learning_rate = 0.084
max_depth = 2
random_state = 42
```

These were the best-performing parameters found across 50 Optuna trials on the walk-forward cross-validation setup. The model is wrapped in a scikit-learn `Pipeline` with the preprocessor as the first step, so raw user input (including NaN values for unavailable time-series features) is handled gracefully through median imputation.

The model and SHAP explainer are initialized once at startup using `@st.cache_resource`, meaning they persist across user sessions without re-training.

### Explainability — SHAP

A `shap.TreeExplainer` is fitted on the trained Gradient Boosting model at startup. When a user submits their input, the preprocessed feature vector is passed through the explainer to produce a SHAP value for every feature. These values are used in two places: the SHAP bar chart and the personalized insights section.

---

## Feature Engineering in the UI (`engineer_row`)

Because the model was trained on a rich set of engineered features, user input cannot be passed to `predict_proba` directly. The `engineer_row` function in `app.py` mirrors the behavioral feature engineering pipeline from `src/newDataScripts/` and computes all features from scratch given the user's inputs.

### Features computed from current-month inputs alone

| Feature Group | Features |
|---|---|
| Monthly totals | `total_amount`, `expense_to_income_ratio` |
| Category concentration | `category_concentration` (Herfindahl index of spending shares) |
| Income-adjusted spend | `{category}_income_adj` for all 14 categories |
| Channel ratios | `grocery_online_ratio`, `misc_online_ratio` |
| Frequency / regularity | `active_categories`, `zero_spend_categories`, `spending_dispersion` |
| Habit indicators | `discretionary_total`, `essential_total`, `discretionary_ratio`, `monthly_savings`, `savings_rate`, `binge_spending_flag`, `dominant_category`, `dominant_category_share` |

### Features computed when prior-month history is provided

| Feature | Requires |
|---|---|
| `expense_mom_change` | Last month's total |
| `rolling_3m_expense` | Last month's total (2-month average) or both prior months (3-month average) |
| `spending_volatility` | Both prior months (coefficient of variation over 3 months) |

Features that require per-category historical data (`trend_{category}`, `category_consistency`) are set to `NaN` regardless, and the pipeline's median imputer substitutes dataset medians for them automatically.

---

## UI Layout

The page uses a two-column layout. The left column holds the input form. The right column is empty until the user clicks **Analyze**, at which point results fill it top to bottom.

---

## Input Form (Left Column)

### Monthly Income
A single number input for the user's total take-home income. Used in the expense-to-income ratio, income-adjusted spending features, savings calculations, and peer benchmarking.

### Spending Categories
14 spending category inputs organized into five collapsible groups:

- **Essentials** — Food & Dining, Grocery (Online), Grocery (In-Store), Health & Fitness, Home
- **Transportation** — Gas & Transport
- **Lifestyle** — Entertainment, Personal Care, Kids & Pets
- **Shopping & Travel** — Shopping (Online), Shopping (In-Store), Travel
- **Miscellaneous** — Misc (Online), Misc (In-Store)

Default values are pre-filled based on realistic monthly spending figures so users can analyze without entering every field.

### Prior Month History (Optional)
A collapsible section where users can enter total spending figures for the previous one or two months. When provided, the time-series features (`expense_mom_change`, `rolling_3m_expense`, `spending_volatility`) are computed from real data rather than imputed — making the prediction more accurate for users with spending history available.

### Live Summary Bar
Three real-time metrics shown beneath the inputs — Total Spending, Monthly Income, and Net Balance — update as the user types before they click Analyze.

---

## Results Panel (Right Column)

### Risk Banner
A color-coded header card displaying the raw overspend probability (0–100%), a risk tier label (Low / Moderate / High), and three supporting metrics: Expense-to-Income Ratio, Estimated Savings, and position relative to the 50% decision threshold. Color transitions from green (< 35%) to amber (35–60%) to red (> 60%).

### Overspend Probability Gauge
A semi-circular gauge displaying the probability as a percentage. The gauge face is divided into three color zones matching the risk tiers. A threshold marker at 50% shows where the model's decision boundary lies.

### Financial Health Score
A composite 0–100 score representing overall financial health, displayed as a large number with a grade label and four component bars. Each component is scored independently:

| Component | Max Points | Scoring Logic |
|---|---|---|
| Savings Rate | 35 | Full points at 20%+ savings rate; zero at 0% or negative |
| Expense Control | 35 | Full points at low expense-to-income ratio; zero at 1.5×+ |
| Low Overspend Risk | 20 | Linear from 20 pts (0% probability) to 0 pts (100% probability) |
| Discretionary Balance | 10 | Full points when discretionary < 50% of essentials; zero at 2×+ |

Grades: **Excellent** (80–100) · **Good** (60–79) · **Fair** (40–59) · **Poor** (20–39) · **Critical** (0–19)

### Spending Breakdown
A donut chart showing the proportional share of each spending category, with the total dollar amount centered in the hole. A text summary below the chart shows discretionary vs. essential totals and their percentage of total spending.

### Why This Prediction Was Made (SHAP)
A horizontal bar chart showing the top 12 features ranked by absolute SHAP value. Red bars indicate features that pushed the prediction toward overspending; blue bars indicate features that reduced the risk. SHAP values are derived from `shap.TreeExplainer` applied to the preprocessed feature vector for the user's specific input — this is a local explanation unique to that user's data, not a global feature importance.

Below the chart, a two-column plain-English summary lists the top three risk-increasing and top three risk-reducing factors by name.

### How You Compare to Similar Earners
A bar chart showing the user's spending in each category as a percentage above or below the average for people with similar incomes. The peer group is dynamically filtered to individuals in `data/v2/model_ready.csv` whose `median_monthly_income` falls within ±25% of the user's income (automatically widened to ±50% if fewer than 15 peers are found at the tighter range). The caption reports the exact income window and peer count. Hover tooltips show the user's dollar amount, the peer average, and the percentage difference.

### Personalized Insights
A set of auto-generated insight cards derived from the user's computed financial ratios and their SHAP values. Up to five insights are shown, covering:

- **Expense-to-income ratio** — flags if spending exceeds income or is approaching the limit
- **Savings rate** — flags negative savings or rates below the recommended 20%
- **Discretionary vs. essential balance** — flags if discretionary spending is more than 1.5× essentials
- **Primary SHAP risk driver** — names the single feature contributing the most to the risk score and directs the user toward the What-If Simulator
- **Spending concentration** — flags if any single category exceeds 30% of total spending

### What-If Simulator
An interactive tool that lets the user explore how changing one spending category would affect their overspend probability.

The user selects a category from a dropdown and moves a slider from −80% to +100% in 5% increments. The simulator pre-computes the full probability curve across that entire range (37 data points) by calling `engineer_row` and `predict_proba` for each step. The results are displayed as a line chart with color-coded risk zone backgrounds (green / amber / red), a dashed threshold line at 50%, and a highlighted star marker at the user's selected adjustment point.

Three metrics above the chart show the current probability, the adjusted probability, and the dollar change to the selected category. A plain-language sentence below summarizes the outcome — for example: *"Reducing Entertainment by 40% (saving $80/month) would lower your overspend probability by 6.2%."*

---

## Session State

User results are persisted in `st.session_state["last_result"]` after each analysis run. This means the results panel remains visible and populated even as the user continues adjusting inputs in the form, and the What-If Simulator always operates on the most recently analyzed scenario.

---

## Deployment Notes

The app is deployed on Streamlit Community Cloud from `tdamasco/FinancialBehaviorXAI-UI`. Key deployment considerations:

- `requirements.txt` lists all runtime dependencies with minimum version constraints: `streamlit`, `plotly`, `pandas`, `numpy`, `scikit-learn`, `shap`, `optuna`
- `data/v2/model_ready.csv` must be present in the repository — the model trains from this file at startup
- The `src/newDataScripts/` directory must be present — `app.py` inserts it into `sys.path` at runtime
- Model training runs once per deployment instance and is cached via `@st.cache_resource`; cold-start time is approximately 10–20 seconds
