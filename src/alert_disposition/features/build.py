"""Feature pipeline. Everything is a sklearn transformer so it is fit on train only and
serialized with the model (no train/serve skew)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from alert_disposition.config import Config


class TimeFeatures(BaseEstimator, TransformerMixin):
    """Hour-of-day / day-of-week cyclic encodings + is_weekend from a datetime column."""

    def __init__(self, col: str):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        t = pd.to_datetime(X[self.col])
        h, d = t.dt.hour.to_numpy(), t.dt.day_of_week.to_numpy()
        return np.column_stack(
            [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), d, (d >= 5).astype(int)]
        )

    def get_feature_names_out(self, names=None):
        return np.array(["hour_sin", "hour_cos", "dow", "is_weekend"])


def build_feature_pipeline(cfg: Config, dense: bool = False) -> ColumnTransformer:
    """``dense=True`` scales numerics (for linear models); trees use raw values."""
    f = cfg.features
    transformers: list[tuple] = []
    if f.categorical:
        transformers.append(
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
                ),
                f.categorical,
            )
        )
    if f.numeric:
        steps = [("impute", SimpleImputer(strategy="median"))]
        if dense:
            steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(steps), f.numeric))
    for col in f.text:
        transformers.append(
            (f"txt_{col}", TfidfVectorizer(max_features=2000, ngram_range=(1, 2)), col)
        )
    if f.time_features:
        transformers.append(("time", TimeFeatures(cfg.data.time_col), [cfg.data.time_col]))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
