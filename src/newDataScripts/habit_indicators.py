import pandas as pd
import numpy as np

SPENDING_COLS = [
    "entertainment", "food_dining", "gas_transport",
    "grocery_net", "grocery_pos", "health_fitness",
    "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel"
]

DISCRETIONARY_COLS = ["entertainment", "shopping_net", "shopping_pos", "travel"]
ESSENTIAL_COLS     = ["food_dining", "grocery_net", "grocery_pos", "home", "health_fitness"]


def add_discretionary_vs_essential(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    disc_cols  = [c for c in DISCRETIONARY_COLS if c in df.columns]
    essen_cols = [c for c in ESSENTIAL_COLS if c in df.columns]
    df["discretionary_total"] = df[disc_cols].sum(axis=1)
    df["essential_total"]     = df[essen_cols].sum(axis=1)
    df["discretionary_ratio"] = (
        df["discretionary_total"] / df["essential_total"].replace(0, np.nan)
    )
    return df


def add_savings_habit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["monthly_savings"] = df["median_monthly_income"] - df["total_amount"]
    df["savings_rate"]    = (
        df["monthly_savings"] / df["median_monthly_income"].replace(0, np.nan)
    )
    return df


def add_binge_spending_flag(df: pd.DataFrame, percentile: float = 90.0) -> pd.DataFrame:
    df = df.copy()
    spend_cols = [c for c in SPENDING_COLS if c in df.columns]
    thresholds = df[spend_cols].quantile(percentile / 100.0)

    flags      = []
    binge_cats = []

    for _, row in df.iterrows():
        triggered = [col for col in spend_cols if row[col] > thresholds[col]]
        flags.append(1 if triggered else 0)
        binge_cats.append(triggered[0] if triggered else None)

    df["binge_spending_flag"] = flags
    df["binge_category"]      = binge_cats
    return df


def add_category_dominance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    spend_cols = [c for c in SPENDING_COLS if c in df.columns]
    row_totals = df[spend_cols].sum(axis=1).replace(0, np.nan)
    df["dominant_category"]       = df[spend_cols].idxmax(axis=1)
    df["dominant_category_share"] = df[spend_cols].max(axis=1) / row_totals
    return df


def calculate_habit_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_discretionary_vs_essential(df)
    df = add_savings_habit(df)
    df = add_binge_spending_flag(df)
    df = add_category_dominance(df)
    return df


def save_habit_indicators(
    df: pd.DataFrame,
    output_path: str = "../data/habit_indicators_v2.csv"
) -> pd.DataFrame:
    result = calculate_habit_indicators(df)
    result.to_csv(output_path, index=False)
    return result