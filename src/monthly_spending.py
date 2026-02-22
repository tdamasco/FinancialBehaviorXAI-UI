# src/feature_engineering.py

import pandas as pd


def create_monthly_spending_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a monthly spending pivot table by category.
    Assumes df already contains:
        - date (datetime)
        - type
        - category
        - amount
    """

    df = df.copy()

    # Ensure datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Keep only Expenses
    df = df[df["type"] == "Expense"]

    # Create Month column (YYYY-MM)
    df["month"] = df["date"].dt.to_period("M")

    # Group by Month and Category
    monthly_spending = (
        df.groupby(["month", "category"])["amount"]
          .sum()
          .reset_index()
    )

    # Convert Period to string for cleaner output
    monthly_spending["month"] = monthly_spending["month"].astype(str)

    # Create Pivot Table
    pivot_table = (
        monthly_spending.pivot(
            index="month",
            columns="category",
            values="amount"
        )
        .fillna(0)
        .reset_index()
    )

    return pivot_table


def save_monthly_spending_table(
    df: pd.DataFrame,
    output_path: str = "../data/monthly_spending_table.csv"
) -> pd.DataFrame:

    pivot_table = create_monthly_spending_table(df)
    pivot_table.to_csv(output_path, index=False)

    return pivot_table