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
from alert_disposition.evaluation.metrics import compute_metrics
from alert_disposition.models.factory import build_model


def _evaluate(model, df: pd.DataFrame, cfg: Config) -> dict[str, float]:
    X, y = df.drop(columns=[cfg.data.label_col]), df[cfg.data.label_col]
    classes = list(model.classes_)
    return compute_metrics(
        y, model.predict(X), model.predict_proba(X), classes, cfg.evaluation.target_tp_recall
    )


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

        results: dict[str, float] = {}
        for name in ("val", "test"):
            for k, v in _evaluate(model, splits[name], cfg).items():
                results[f"{name}_{k}"] = v
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
    return results
