import pandas as pd
import numpy as np


def create_overspending_target(df: pd.DataFrame) -> pd.DataFrame:
    required = ["first", "last", "period", "total_amount", "median_monthly_income"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(["first", "last", "period"]).reset_index(drop=True)

    df["overspend_current_month"] = (
        df["total_amount"] > df["median_monthly_income"] * 0.80
    ).astype(int)

    results = []

    for (first, last), group in df.groupby(["first", "last"], sort=False):
        group = group.copy()
        group["overspend_target_t1"] = group["overspend_current_month"].shift(-1)
        results.append(group)

    df = pd.concat(results).reset_index(drop=True)
    df = df.dropna(subset=["overspend_target_t1"]).copy()
    df["overspend_target_t1"] = df["overspend_target_t1"].astype(int)

    return df


def save_target_dataset(
    df: pd.DataFrame,
    output_path: str = "../data/target_v2.csv"
) -> pd.DataFrame:
    result = create_overspending_target(df)
    result.to_csv(output_path, index=False)
    return result