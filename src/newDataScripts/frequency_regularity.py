import pandas as pd
import numpy as np

SPENDING_COLS = [
    "entertainment", "food_dining", "gas_transport",
    "grocery_net", "grocery_pos", "health_fitness",
    "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel"
]


def calculate_frequency_regularity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["first", "last", "period"]).reset_index(drop=True)

    spending = df[SPENDING_COLS]

    df["active_categories"] = (spending > 0).sum(axis=1)
    df["zero_spend_categories"] = (spending == 0).sum(axis=1)
    df["spending_dispersion"] = spending.std(axis=1)

    results = []

    for (first, last), group in df.groupby(["first", "last"], sort=False):
        group = group.copy()
        active = (group[SPENDING_COLS] > 0)
        prev_active = active.shift(1)
        consistency = (
            (active & prev_active).sum(axis=1) /
            active.sum(axis=1).replace(0, np.nan)
        )
        group["category_consistency"] = consistency
        results.append(group)

    return pd.concat(results).reset_index(drop=True)


def save_frequency_regularity(
    df: pd.DataFrame,
    output_path: str = "../data/frequency_regularity_v2.csv"
) -> pd.DataFrame:
    result = calculate_frequency_regularity(df)
    result.to_csv(output_path, index=False)
    return result