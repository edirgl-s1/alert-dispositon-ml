from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from alert_disposition.config import Config


def split_data(df: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Train/val/test split. Default is chronological: the model is always scored on the
    future, mimicking deployment and avoiding leakage from near-duplicate alerts."""
    s = cfg.data.split
    if s.strategy == "time":
        df = df.sort_values(cfg.data.time_col).reset_index(drop=True)
        n = len(df)
        n_test, n_val = int(n * s.test_frac), int(n * s.val_frac)
        n_train = n - n_test - n_val
        return {
            "train": df.iloc[:n_train],
            "val": df.iloc[n_train : n_train + n_val],
            "test": df.iloc[n_train + n_val :],
        }
    train_val, test = train_test_split(
        df, test_size=s.test_frac, random_state=cfg.seed, stratify=df[cfg.data.label_col]
    )
    train, val = train_test_split(
        train_val,
        test_size=s.val_frac / (1 - s.test_frac),
        random_state=cfg.seed,
        stratify=train_val[cfg.data.label_col],
    )
    return {"train": train, "val": val, "test": test}
