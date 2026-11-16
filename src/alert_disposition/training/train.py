"""Training entrypoint: load -> split -> fit -> evaluate -> log everything to MLflow."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml

from alert_disposition.config import ROOT, Config
from alert_disposition.data.loader import load_raw
from alert_disposition.data.split import split_data
from alert_disposition.evaluation.metrics import (
    compute_metrics,
    metrics_by_group,
    select_threshold,
)
from alert_disposition.models.factory import build_model


def _score(model, df: pd.DataFrame, cfg: Config):
    """P(needs_investigation) for each row."""
    return model.predict_proba(df.drop(columns=[cfg.data.label_col]))[:, 1]


def train(cfg: Config, save_dir: Path | None = None) -> dict[str, float]:
    uri = cfg.mlflow.tracking_uri
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{ROOT / uri.removeprefix('sqlite:///')}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.experiment_name)

    splits = split_data(load_raw(cfg), cfg)
    label = cfg.data.label_col
    Xtr, ytr = splits["train"].drop(columns=[label]), splits["train"][label]

    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params(
            {
                "model": cfg.model.name,
                "seed": cfg.seed,
                **{f"model.{k}": v for k, v in cfg.model.params.items()},
            }
        )
        mlflow.log_params({f"n_{k}": len(v) for k, v in splits.items()})
        mlflow.log_text(yaml.safe_dump(cfg.model_dump()), "config.yaml")

        model = build_model(cfg)
        model.fit(Xtr, ytr)

        # The auto-dispose threshold is chosen on val and applied unchanged to test.
        ev = cfg.evaluation
        p_val = _score(model, splits["val"], cfg)
        threshold = select_threshold(splits["val"][label], p_val, ev.max_missed_investigation_rate)
        results: dict[str, float] = {}
        group_metrics: dict = {}
        for name in ("val", "test"):
            p = p_val if name == "val" else _score(model, splits[name], cfg)
            for k, v in compute_metrics(splits[name][label], p, threshold).items():
                results[f"{name}_{k}"] = v
            group_metrics[name] = metrics_by_group(splits[name], label, p, threshold, ev.group_cols)
        mlflow.log_metrics(results)

        out = save_dir or ROOT / "models"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{cfg.run_name}.joblib"
        joblib.dump(model, path)
        mlflow.log_artifact(str(path))
        with tempfile.TemporaryDirectory() as td:
            mp = Path(td) / "metrics.json"
            mp.write_text(json.dumps(results, indent=2))
            mlflow.log_artifact(str(mp))
            gp = Path(td) / "group_metrics.json"
            gp.write_text(json.dumps(group_metrics, indent=2))
            mlflow.log_artifact(str(gp))
    return results
