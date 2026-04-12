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


def add_expense_to_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["expense_to_income_ratio"] = (
        df["total_amount"] / df["median_monthly_income"].replace(0, np.nan)
    )
    return df


def add_category_concentration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    spend_cols = [c for c in SPENDING_COLS if c in df.columns]
    row_totals = df[spend_cols].sum(axis=1).replace(0, np.nan)
    shares = df[spend_cols].div(row_totals, axis=0)
    df["category_concentration"] = (shares ** 2).sum(axis=1)
    return df


def add_income_adjusted_spending(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    spend_cols = [c for c in SPENDING_COLS if c in df.columns]
    income = df["median_monthly_income"].replace(0, np.nan)
    for col in spend_cols:
        df[f"{col}_income_adj"] = df[col] / income
    return df


def add_grocery_channel_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    total_grocery = (
        df.get("grocery_net", 0) + df.get("grocery_pos", 0)
    ).replace(0, np.nan)
    df["grocery_online_ratio"] = df.get("grocery_net", 0) / total_grocery
    return df


def add_misc_channel_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    total_misc = (
        df.get("misc_net", 0) + df.get("misc_pos", 0)
    ).replace(0, np.nan)
    df["misc_online_ratio"] = df.get("misc_net", 0) / total_misc
    return df


def calculate_monthly_spending_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = add_expense_to_income_ratio(df)
    df = add_category_concentration(df)
    df = add_income_adjusted_spending(df)
    df = add_grocery_channel_ratio(df)
    df = add_misc_channel_ratio(df)
    return df


def save_monthly_spending_features(
    df: pd.DataFrame,
    output_path: str = "../data/monthly_spending_v2.csv"
) -> pd.DataFrame:
    result = calculate_monthly_spending_features(df)
    result.to_csv(output_path, index=False)
    return result