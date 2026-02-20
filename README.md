# 📊 FinancialBehaviorXAI

**Behavioral Feature Engineering & Explainable AI for Overspending Prediction**

## 📌 Overview

FinancialBehaviorXAI is a behavioral finance modeling project designed to predict overspending risk using transaction-level personal finance data.

Rather than modeling raw transactions directly, this project engineers **behavioral financial indicators** such as spending volatility, category concentration, transaction frequency, and habit-based patterns to create a structured monthly risk prediction framework.

The project integrates:

* Behavioral feature engineering
* Time-aware forecasting
* Overspending classification
* Explainable AI (SHAP/LIME-ready pipeline)

---

## 📂 Dataset

Source: Kaggle – Personal Finance Dataset
The dataset contains ~4 years of synthetic transaction-level financial records including:

* `Date`
* `Transaction Description`
* `Category`
* `Amount`
* `Type` (Income / Expense)

Each row represents a single transaction.

---

## 🧠 Project Objective

To answer:

> Can behavioral financial patterns predict whether someone will overspend in the next month?

Overspending is defined as:

```
Total Monthly Expenses > Total Monthly Income
```

The model uses features from month **T** to predict overspending in month **T+1**, preventing data leakage.

---

## 🏗️ Project Structure

```
FinancialBehaviorXAI/
│
├── data/
│   ├── Personal_Finance_Dataset.csv
│   └── cleaned_finance_data.csv
│
├── notebooks/
│   └── main.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── evaluation.py
│
├── README.md
└── requirements.txt
```

---

## 🔄 Pipeline Overview

### 1️⃣ Data Cleaning

* Standardized column names
* Date parsing
* Amount normalization
* Transaction sign correction
* Duplicate removal

---

### 2️⃣ Transaction Categorization & Normalization

* Category grouping (Fixed, Essential, Discretionary, etc.)
* Signed transaction amounts
* Log-transformed magnitudes
* Robust scaling by transaction type

---

### 3️⃣ Outlier Handling & Noise Reduction

* IQR-based clipping (by transaction type)
* Category-level magnitude clipping
* Removal of impossible records

---

### 4️⃣ Monthly Aggregation

Transactions are aggregated into monthly summaries:

* Total income
* Total expenses
* Expense-to-income ratio
* Category-level spend totals

---

### 5️⃣ Behavioral Feature Engineering

#### 📅 Monthly & Category Spend

* Expense ratio
* Category concentration
* Income-adjusted spending

#### 🔁 Frequency & Regularity

* Transaction count
* Average inter-transaction gap
* Monthly spending dispersion

#### 📈 Trends & Volatility

* Rolling 3-month averages
* Spending trend slope
* Coefficient of variation

#### 🧠 Habit-Based Indicators

* Recurring expense detection
* Savings rate
* Overspending history
* Lagged financial metrics

---

### 6️⃣ Overspending Target Creation

Target is defined as:

```
Overspend (T+1) = 1 if Expense_T+1 > Income_T+1
```

The target is shifted forward to avoid leakage.

---

### 7️⃣ Modeling

Planned models include:

* Logistic Regression
* Random Forest
* Gradient Boosting (LightGBM/XGBoost)

Evaluation metrics:

* Accuracy
* F1 Score
* ROC-AUC
* Precision-Recall

Time-aware train/test split is used instead of random splitting.

---

### 8️⃣ Explainable AI

The project integrates SHAP and/or LIME to:

* Identify which behavioral factors drive overspending
* Provide interpretable insights such as:

  > “High discretionary spending volatility contributed 32% to overspending risk.”

---

## 📊 Key Insights (Planned)

This project aims to demonstrate:

* Behavioral volatility is often more predictive than raw spending amount.
* Expense-to-income ratio is necessary but insufficient alone.
* Recurring discretionary spikes are strong early-warning indicators.
* Time-aware modeling significantly improves real-world realism.

---

## 🚀 Future Enhancements

* Anomaly detection for early warning systems
* Personalized financial health score
* Dashboard visualization (Tableau / Streamlit)
* Deployment-ready API pipeline

---

## 💼 Why This Project Matters

This project demonstrates:

* Structured data engineering
* Time-series behavioral modeling
* Explainable AI integration
* Finance + machine learning application
* Clean, production-style repository design

---

## 📬 Contact

If you’re interested in behavioral finance modeling, explainable AI, or financial risk analytics, feel free to connect.
* Ayda Sahren (nkb5509@psu.edu)
* Bobby Dodge (rld5566@psu.edu)
* Fateenah Farid (njn5346@psu.edu)
* Kaitlyn Forister (kgf5124@psu.edu)
* Tim Damasco (tjd6015@psu.edu)

---
