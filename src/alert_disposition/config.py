"""Typed experiment configuration: layered YAML files + dotted CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SplitConfig(_Base):
    strategy: Literal["time", "random"] = "time"
    test_frac: float = 0.2
    val_frac: float = 0.1


class DataConfig(_Base):
    source: Literal["synthetic", "parquet"] = "synthetic"
    raw_path: str = "data/raw/alerts.parquet"
    processed_dir: str = "data/processed"
    label_col: str = "disposition"
    time_col: str = "created_at"
    id_col: str = "alert_id"
    n_synthetic: int = 5000
    split: SplitConfig = SplitConfig()


class FeatureConfig(_Base):
    categorical: list[str] = []
    numeric: list[str] = []
    text: list[str] = []
    time_features: bool = True


class ModelConfig(_Base):
    name: Literal["lightgbm", "logreg"] = "lightgbm"
    params: dict[str, Any] = {}


class EvalConfig(_Base):
    primary_metric: str = "f1_macro"
    target_tp_recall: float = 0.99


class MlflowConfig(_Base):
    tracking_uri: str = "sqlite:///experiments/mlflow.db"


class Config(_Base):
    experiment_name: str = "alert-disposition"
    run_name: str = "baseline"
    seed: int = 42
    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    evaluation: EvalConfig = EvalConfig()
    mlflow: MlflowConfig = MlflowConfig()


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if (
            k == "model"
            and isinstance(v, dict)
            and v.get("name") not in (None, a.get("model", {}).get("name"))
        ):
            out[k] = v  # switching model: drop the previous model's params
            continue
        out[k] = (
            _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out


def _set_dotted(d: dict, key: str, value: Any) -> None:
    *parents, leaf = key.split(".")
    for p in parents:
        d = d.setdefault(p, {})
    d[leaf] = value


def load_config(paths: list[Path | str], overrides: list[str] | None = None) -> Config:
    """Merge YAML files left-to-right, then apply ``a.b.c=value`` overrides (YAML-parsed values)."""
    merged: dict = {}
    for p in paths:
        merged = _deep_merge(merged, yaml.safe_load(Path(p).read_text()) or {})
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        if key.strip() == "model.name" and yaml.safe_load(raw) != merged.get("model", {}).get(
            "name"
        ):
            merged["model"] = {}  # switching model: drop the previous model's params
        _set_dotted(merged, key.strip(), yaml.safe_load(raw))
    return Config(**merged)
