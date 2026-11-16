from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from alert_disposition.config import Config
from alert_disposition.features.transformer import AlertFeatureTransformer


def build_model(cfg: Config) -> Pipeline:
    """Full pipeline (raw alert payload -> features -> estimator). Register new models here."""
    p = dict(cfg.model.params)
    steps: list[tuple[str, BaseEstimator]] = [
        (
            "features",
            AlertFeatureTransformer(
                cfg.features.to_runtime(), cfg.data.source_format, cfg.data.payload_col
            ),
        )
    ]
    if cfg.model.name == "lightgbm":
        from lightgbm import LGBMClassifier  # lazy: needs libomp on macOS (brew install libomp)

        steps.append(("clf", LGBMClassifier(random_state=cfg.seed, verbose=-1, **p)))
    elif cfg.model.name == "logreg":
        steps.append(("scale", MaxAbsScaler()))  # sparse-safe; keeps counts comparable to tokens
        steps.append(("clf", LogisticRegression(random_state=cfg.seed, **p)))
    else:  # pragma: no cover - guarded by pydantic Literal
        raise ValueError(cfg.model.name)
    return Pipeline(steps)
