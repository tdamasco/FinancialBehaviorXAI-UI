import pandas as pd
import numpy as np

SPENDING_COLS = [
    "entertainment", "food_dining", "gas_transport",
    "grocery_net", "grocery_pos", "health_fitness",
    "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel"
]


def calculate_trends_and_volatility(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["first", "last", "period"]).reset_index(drop=True)

    results = []

    for (first, last), group in df.groupby(["first", "last"], sort=False):
        group = group.copy()

        group["rolling_3m_expense"] = (
            group["total_amount"].rolling(window=3, min_periods=1).mean()
        )

        group["expense_mom_change"] = group["total_amount"].pct_change()

        rolling_std  = group["total_amount"].rolling(window=3, min_periods=2).std()
        rolling_mean = group["total_amount"].rolling(window=3, min_periods=2).mean()
        group["spending_volatility"] = rolling_std / rolling_mean.replace(0, np.nan)

        for col in SPENDING_COLS:
            if col in group.columns:
                group[f"trend_{col}"] = (
                    group[col].rolling(window=3, min_periods=1).mean()
                )

        results.append(group)

    return pd.concat(results).reset_index(drop=True)


def save_trends(
    df: pd.DataFrame,
    output_path: str = "../data/trends_v2.csv"
) -> pd.DataFrame:
    result = calculate_trends_and_volatility(df)
    result.to_csv(output_path, index=False)
    return result