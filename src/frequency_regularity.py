# src/frequency_regularity.py

import pandas as pd
import numpy as np


def calculate_frequency_regularity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates behavioral features based on transaction frequency and regularity.

    Expects the cleaned transaction-level DataFrame from preprocess_data, containing:
        - date (datetime)
        - type
        - category
        - amount

    Returns a monthly DataFrame with columns:
        - month
        - transaction_count       : number of expense transactions in the month
        - avg_inter_txn_gap       : average days between transactions (regularity)
        - spending_dispersion     : std dev of individual transaction amounts (within-month spread)
        - unique_categories_used  : number of distinct spending categories active that month
    """

    df = df.copy()

    # Work with expense transactions only
    expenses = df[df["type"] == "Expense"].copy()

    # Month period column
    expenses["month"] = expenses["date"].dt.to_period("M")

    # --- 1. Transaction Count ---
    txn_count = (
        expenses.groupby("month")
        .size()
        .reset_index(name="transaction_count")
    )

    # --- 2. Average Inter-Transaction Gap (days) ---
    def avg_gap(dates: pd.Series) -> float:
        sorted_dates = dates.sort_values()
        if len(sorted_dates) < 2:
            return np.nan
        gaps = sorted_dates.diff().dt.days.dropna()
        return gaps.mean()

    gap_df = (
        expenses.groupby("month")["date"]
        .apply(avg_gap)
        .reset_index(name="avg_inter_txn_gap")
    )

    # --- 3. Spending Dispersion (std dev of transaction amounts within month) ---
    dispersion_df = (
        expenses.groupby("month")["amount"]
        .std()
        .reset_index(name="spending_dispersion")
    )

    # --- 4. Unique Categories Used ---
    categories_df = (
        expenses.groupby("month")["category"]
        .nunique()
        .reset_index(name="unique_categories_used")
    )

    # --- Merge all features ---
    result = txn_count.copy()
    for part in [gap_df, dispersion_df, categories_df]:
        result = result.merge(part, on="month", how="left")

    # Convert Period to string for consistency with other modules
    result["month"] = result["month"].astype(str)

    return result


def save_frequency_regularity(
    df: pd.DataFrame,
    output_path: str = "../data/frequency_regularity.csv"
) -> pd.DataFrame:

    result = calculate_frequency_regularity(df)
    result.to_csv(output_path, index=False)

    return result
