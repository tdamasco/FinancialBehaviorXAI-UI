# src/habit_indicators.py

import pandas as pd
import numpy as np


# --------------------------------------------------
# Helper: Build monthly income table from raw df
# --------------------------------------------------

def _get_monthly_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sums all income transactions per month.
    Uses all income types (Salary, Investment, Other, etc.)
    """
    income_df = df[df["type"] == "Income"].copy()
    income_df["month"] = pd.to_datetime(income_df["date"]).dt.to_period("M").astype(str)
    monthly_income = (
        income_df.groupby("month")["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "total_income"})
    )
    return monthly_income


# --------------------------------------------------
# 1. Discretionary vs. Essential Ratio
# --------------------------------------------------

DISCRETIONARY_CATEGORIES = {"Entertainment", "Shopping", "Travel"}
ESSENTIAL_CATEGORIES     = {"Rent", "Utilities", "Food & Drink"}


def add_discretionary_vs_essential(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio of discretionary spending (Entertainment, Shopping, Travel)
    to essential spending (Rent, Utilities, Food & Drink).

    A ratio > 1 means the person spent more on wants than needs that month.
    Stored as NaN when essential spending is 0 to avoid division by zero.
    """
    df = monthly_df.copy()

    disc_cols  = [c for c in df.columns if c in DISCRETIONARY_CATEGORIES]
    essen_cols = [c for c in df.columns if c in ESSENTIAL_CATEGORIES]

    df["discretionary_total"] = df[disc_cols].sum(axis=1)  if disc_cols  else 0.0
    df["essential_total"]     = df[essen_cols].sum(axis=1) if essen_cols else 0.0

    df["discretionary_ratio"] = df["discretionary_total"] / df["essential_total"].replace(0, np.nan)

    return df


# --------------------------------------------------
# 2. Savings Habit
# --------------------------------------------------

def add_savings_habit(monthly_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Savings = total monthly income - total monthly expense.
    Savings rate = savings / total income (NaN when income is 0).

    Requires raw_df (the transaction-level dataframe from preprocess_data)
    to reconstruct monthly income across all income types.
    """
    df = monthly_df.copy()

    monthly_income = _get_monthly_income(raw_df)
    df = df.merge(monthly_income, on="month", how="left")
    df["total_income"] = df["total_income"].fillna(0)

    # total_expense must already exist (added by calculate_trends_and_volatility)
    # or we compute it here as a fallback
    if "total_expense" not in df.columns:
        expense_cols = [c for c in df.columns if c not in {"month", "total_income"}]
        df["total_expense"] = df[expense_cols].sum(axis=1)

    df["monthly_savings"]  = df["total_income"] - df["total_expense"]
    df["savings_rate"]     = df["monthly_savings"] / df["total_income"].replace(0, np.nan)

    return df


# --------------------------------------------------
# 3. Binge Spending Flag
# --------------------------------------------------

def add_binge_spending_flag(monthly_df: pd.DataFrame, percentile: float = 90.0) -> pd.DataFrame:
    """
    Flags months where ANY spending category exceeds its own historical
    {percentile}th percentile across all months.

    binge_spending_flag = 1  →  at least one category spiked unusually high
    binge_category            →  which category triggered the flag (first one found)
    """
    df = monthly_df.copy()

    non_feature_cols = {
        "month", "total_expense", "rolling_3m_expense",
        "expense_mom_change", "spending_volatility",
        "discretionary_total", "essential_total", "discretionary_ratio",
        "total_income", "monthly_savings", "savings_rate",
    }
    spend_cols = [c for c in df.columns if c not in non_feature_cols]

    thresholds = df[spend_cols].quantile(percentile / 100.0)

    flags        = []
    binge_cats   = []

    for _, row in df.iterrows():
        triggered = [col for col in spend_cols if row[col] > thresholds[col]]
        flags.append(1 if triggered else 0)
        binge_cats.append(triggered[0] if triggered else None)

    df["binge_spending_flag"] = flags
    df["binge_category"]      = binge_cats

    return df


# --------------------------------------------------
# 4. Category Dominance
# --------------------------------------------------

def add_category_dominance(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies the single largest spending category each month and
    its share of total spending that month.

    dominant_category       →  category name
    dominant_category_share →  fraction of total spend (0–1)
    """
    df = monthly_df.copy()

    non_feature_cols = {
        "month", "total_expense", "rolling_3m_expense",
        "expense_mom_change", "spending_volatility",
        "discretionary_total", "essential_total", "discretionary_ratio",
        "total_income", "monthly_savings", "savings_rate",
        "binge_spending_flag", "binge_category",
    }
    spend_cols = [c for c in df.columns if c not in non_feature_cols]

    row_totals = df[spend_cols].sum(axis=1).replace(0, np.nan)

    df["dominant_category"]       = df[spend_cols].idxmax(axis=1)
    df["dominant_category_share"] = df[spend_cols].max(axis=1) / row_totals

    return df


# --------------------------------------------------
# 5. Full Habit Indicators Pipeline
# --------------------------------------------------

def save_habit_indicators(
    monthly_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    output_path: str = "../data/habit_indicators.csv"
) -> pd.DataFrame:
    """
    Runs the full habit indicators pipeline and saves the result to a CSV.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Output of calculate_trends_and_volatility().
    raw_df : pd.DataFrame
        Output of preprocess_data().
    output_path : str
        Path to save the CSV file.

    Returns
    -------
    pd.DataFrame
        The enriched dataframe (also saved to output_path).
    """
    df = calculate_habit_indicators(monthly_df, raw_df)
    df.to_csv(output_path, index=False)
    return df


def calculate_habit_indicators(monthly_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function that applies all habit-based feature engineering steps
    in the correct order.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Output of calculate_trends_and_volatility() — already contains
        total_expense, rolling_3m_expense, spending_volatility, etc.
    raw_df : pd.DataFrame
        Output of preprocess_data() — transaction-level data needed
        to reconstruct monthly income.

    Returns
    -------
    pd.DataFrame
        monthly_df enriched with all habit indicators.
    """
    df = monthly_df.copy()
    df = add_discretionary_vs_essential(df)
    df = add_savings_habit(df, raw_df)
    df = add_binge_spending_flag(df)
    df = add_category_dominance(df)
    return df
