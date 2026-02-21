from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import iqr


def load_data(path: str | Path) -> pd.DataFrame:
    """Load raw credit card data from CSV."""
    return pd.read_csv(path)


def compute_amount_upper_limit(df: pd.DataFrame) -> float:
    """Upper outlier limit: Q3 + 1.5 * IQR for Amount."""
    amount = df["Amount"]
    return amount.quantile(0.75) + 1.5 * iqr(amount)


def describe_amount_outliers(df: pd.DataFrame) -> pd.Series:
    """Class counts for Amount above IQR-based upper limit."""
    upper_limit = compute_amount_upper_limit(df)
    return df[df["Amount"] > upper_limit]["Class"].value_counts()


def filter_amount(df: pd.DataFrame, max_amount: float = 8000.0) -> pd.DataFrame:
    """Keep only rows with Amount <= max_amount."""
    return df[df["Amount"] <= max_amount].copy()


def undersample_majority(
    df: pd.DataFrame,
    ratios: Tuple[int, int, int] = (10, 20, 50),
    target_col: str = "Class",
    random_state: int = 24,
) -> Dict[str, pd.DataFrame]:

    """
    Build undersampled datasets for given majority:minority ratios.

    Returns dict:
      {
        "1_to_10": df_1_to_10,
        "1_to_20": df_1_to_20,
        "1_to_50": df_1_to_50,
      }
    """
    counts = df[target_col].value_counts()
    minority_cnt = int(counts[1])

    fraud = df[df[target_col] == 1]
    non_fraud = df[df[target_col] == 0]

    out: Dict[str, pd.DataFrame] = {}
    for r in ratios:
        needed = minority_cnt * r
        sampled_non_fraud = non_fraud.sample(needed, random_state=random_state)
        combined = (
            pd.concat([sampled_non_fraud, fraud], axis=0)
            .sample(frac=1, random_state=random_state)
            .reset_index(drop=True)
        )
        out[f"1_to_{r}"] = combined

    return out


def split_X_y(df: pd.DataFrame, target_col: str = "Class") -> Tuple[np.ndarray, np.ndarray]:
    """Convert DataFrame to feature matrix X and label vector y."""
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    return X, y
