"""sklearn adapter: DataFrame with a raw-alert column -> sparse feature matrix.

Lets ``FeaturePipeline`` sit inside a ``sklearn.pipeline.Pipeline`` in front of a classifier,
so one fitted object carries normalization, vocabularies and the model together.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin

from alert_disposition.features.adapters import normalize_alert
from alert_disposition.features.config import FeatureConfig
from alert_disposition.features.extract import FeatureDocument, extract_feature_document
from alert_disposition.features.pipeline import FeaturePipeline


def parse_payload(raw: object, row: int) -> object:
    """JSON strings (the FE ``rawData`` form) become dicts; other payloads pass through.

    TBD: one bad row currently aborts the whole batch (docs/open_issues.md #2).

    Raises:
        ValueError: On malformed JSON. The message names the row, never the content.
    """
    if isinstance(raw, (str, bytes)):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"row {row}: payload is not valid JSON") from exc
    return raw


class AlertFeatureTransformer(BaseEstimator, TransformerMixin):
    """Raw alert payload column -> CSR feature matrix (fit on training rows only)."""

    def __init__(
        self,
        config: FeatureConfig | None = None,
        source_format: str = "ocsf",
        payload_col: str = "alert",
    ) -> None:
        self.config = config
        self.source_format = source_format
        self.payload_col = payload_col

    def _docs(self, X: pd.DataFrame) -> Sequence[FeatureDocument]:
        cfg = self.config or FeatureConfig()
        return [
            extract_feature_document(
                normalize_alert(parse_payload(raw, i), self.source_format, config=cfg), cfg
            )
            for i, raw in enumerate(X[self.payload_col])
        ]

    def fit(self, X: pd.DataFrame, y: object = None) -> AlertFeatureTransformer:
        """Learn vocabularies and categories from ``X`` (training split only)."""
        self.pipeline_ = FeaturePipeline(self.config).fit(self._docs(X))
        return self

    def transform(self, X: pd.DataFrame) -> sp.csr_matrix:
        """Vectorize with the fitted vocabulary; never refits."""
        return self.pipeline_.transform(self._docs(X))

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Prefixed feature names in column order."""
        return np.array(self.pipeline_.get_feature_names_out())
