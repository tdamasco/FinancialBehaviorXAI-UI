# data_preprocessing.py

import pandas as pd
import numpy as np
import re


# --------------------------------------------------
# 1. Load Data
# --------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


# --------------------------------------------------
# 2. Standardize Column Names
# --------------------------------------------------

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


# --------------------------------------------------
# 3. Parse Dates
# --------------------------------------------------

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# --------------------------------------------------
# 4. Clean Amount Column
# --------------------------------------------------

def clean_amount(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace(r"[,$\s]", "", regex=True)
    )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


# --------------------------------------------------
# 5. Normalize Transaction Type
# --------------------------------------------------

def normalize_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["type"] = df["type"].astype(str).str.strip().str.title()
    return df


# --------------------------------------------------
# 6. Correct Sign Convention
# --------------------------------------------------

def apply_signed_amount(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["signed_amount"] = df["amount"]

    df.loc[df["type"] == "Income", "signed_amount"] = \
        df.loc[df["type"] == "Income", "amount"].abs()

    df.loc[df["type"] == "Expense", "signed_amount"] = \
        -df.loc[df["type"] == "Expense", "amount"].abs()

    return df


# --------------------------------------------------
# 7. Remove Missing & Duplicates
# --------------------------------------------------

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["date", "amount", "type", "category"])
    df = df.drop_duplicates()
    return df


# --------------------------------------------------
# 8. Outlier Handling (Transaction Level)
# --------------------------------------------------

def clip_outliers_iqr(df: pd.DataFrame, column: str, k: float = 3.0) -> pd.DataFrame:
    df = df.copy()

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - k * IQR
    upper = Q3 + k * IQR

    df[column] = df[column].clip(lower, upper)

    return df


# --------------------------------------------------
# 9. Full Preprocessing Pipeline
# --------------------------------------------------

def preprocess_data(filepath: str) -> pd.DataFrame:
    df = load_data(filepath)
    df = standardize_columns(df)
    df = parse_dates(df)
    df = clean_amount(df)
    df = normalize_type(df)
    df = apply_signed_amount(df)
    df = basic_cleaning(df)

    # Clip extreme transaction values
    df = clip_outliers_iqr(df, column="signed_amount")

    return df