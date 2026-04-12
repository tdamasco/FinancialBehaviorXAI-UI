import pandas as pd
import numpy as np

SPENDING_COLS = [
    "entertainment", "food_dining", "gas_transport",
    "grocery_net", "grocery_pos", "health_fitness",
    "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel"
]


def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("&", "and")
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df


def build_period(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2),
        format="%Y-%m"
    )
    return df


def fill_missing_spending(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in SPENDING_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df


def validate_income(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["median_monthly_income"].notna()]
    df = df[df["median_monthly_income"] > 0]
    return df


def clip_outliers_iqr(df: pd.DataFrame, column: str, k: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    df[column] = df[column].clip(lower, upper)
    return df


def clip_all_spending(df: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    for col in SPENDING_COLS:
        if col in df.columns:
            df = clip_outliers_iqr(df, col, k=k)
    return df


def preprocess_data(filepath: str) -> pd.DataFrame:
    df = load_data(filepath)
    df = standardize_columns(df)
    df = build_period(df)
    df = fill_missing_spending(df)
    df = validate_income(df)
    df = clip_all_spending(df)
    return df