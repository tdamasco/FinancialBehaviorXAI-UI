import pandas as pd
import numpy as np


def create_overspending_target(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the forward-shifted overspending target.

    Overspend (T+1) = 1 if total_expense_(T+1) > total_income_(T+1)

    The target is shifted backward onto month T so that features from month T
    predict whether overspending happens in month T+1.

    Expected columns in monthly_df:
    - month
    - total_expense
    - total_income

    Returns
    -------
    pd.DataFrame
        Same dataframe with:
        - overspend_current_month
        - overspend_target_t1
    """
    df = monthly_df.copy()

    if "month" not in df.columns:
        raise ValueError("monthly_df must contain a 'month' column.")
    if "total_expense" not in df.columns:
        raise ValueError("monthly_df must contain a 'total_expense' column.")
    if "total_income" not in df.columns:
        raise ValueError("monthly_df must contain a 'total_income' column.")

    # Ensure chronological order
    df["month"] = pd.to_datetime(df["month"].astype(str), errors="coerce")
    df = df.sort_values("month").reset_index(drop=True)

    # Current-month overspending flag
    df["overspend_current_month"] = (
        df["total_expense"] > df["total_income"]
    ).astype(int)

    # Shift forward target back onto current row
    df["overspend_target_t1"] = df["overspend_current_month"].shift(-1)

    # Drop final row because it has no T+1 label
    df = df.dropna(subset=["overspend_target_t1"]).copy()
    df["overspend_target_t1"] = df["overspend_target_t1"].astype(int)

    # Convert month back to string if you want consistency with other modules
    df["month"] = df["month"].dt.to_period("M").astype(str)

    return df


def save_target_dataset(
    monthly_df: pd.DataFrame,
    output_path: str = "../data/model_ready_with_target.csv"
) -> pd.DataFrame:
    """
    Create the shifted target and save to CSV.
    """
    df = create_overspending_target(monthly_df)
    df.to_csv(output_path, index=False)
    return df