from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from alert_disposition.config import Config
from alert_disposition.features.build import build_feature_pipeline


def build_model(cfg: Config) -> Pipeline:
    """Full pipeline (features + estimator). Register new models here."""
    p = dict(cfg.model.params)
    clf: BaseEstimator
    if cfg.model.name == "lightgbm":
        from lightgbm import LGBMClassifier  # lazy: needs libomp on macOS (brew install libomp)

        clf = LGBMClassifier(random_state=cfg.seed, verbose=-1, **p)
        feats = build_feature_pipeline(cfg, dense=False)
    elif cfg.model.name == "logreg":
        clf = LogisticRegression(random_state=cfg.seed, **p)
        feats = build_feature_pipeline(cfg, dense=True)
    else:  # pragma: no cover - guarded by pydantic Literal
        raise ValueError(cfg.model.name)
    return Pipeline([("features", feats), ("clf", clf)])
