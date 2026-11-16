"""Typed experiment configuration: layered YAML files + dotted CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from alert_disposition.features.config import (
    EXPERIMENT_FAMILIES,
    ChannelConfig,
    EmptyVocabularyStrategy,
)
from alert_disposition.features.config import FeatureConfig as RuntimeFeatureConfig

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
    label_col: str = "needs_investigation"  # 0/1: 1 = should have been investigated (not TP/FP)
    payload_col: str = "alert"  # raw alert payload (JSON string or dict)
    source_format: str = "ocsf"  # adapter name, see features.adapters
    time_col: str = "created_at"
    id_col: str = "alert_id"
    n_synthetic: int = 5000
    split: SplitConfig = SplitConfig()


class ChannelSettings(_Base):
    min_df: int | float = 10
    max_df: int | float = 1.0
    max_features: int | None = None
    ngram_range: tuple[int, int] = (1, 2)


class FeaturesConfig(_Base):
    """YAML view of ``alert_disposition.features.config.FeatureConfig`` (see its docstring)."""

    experiment: Literal["E0", "E1", "E2", "E3", "E4"] = "E3"
    include_confidence: bool = False
    source_qualified_tokens: bool = False
    attack_fields: list[str] = []
    metadata_allowlist: list[str] = []
    ocsf_full_path_key: str | None = None
    on_empty_vocabulary: Literal["raise", "skip"] = "raise"
    path: ChannelSettings = ChannelSettings()
    command: ChannelSettings = ChannelSettings()
    attack: ChannelSettings = ChannelSettings()
    text: ChannelSettings = ChannelSettings(max_features=10000)

    def to_runtime(self) -> RuntimeFeatureConfig:
        return RuntimeFeatureConfig(
            families=EXPERIMENT_FAMILIES[self.experiment],
            path=ChannelConfig(**self.path.model_dump()),
            command=ChannelConfig(**self.command.model_dump()),
            attack=ChannelConfig(**self.attack.model_dump()),
            text=ChannelConfig(**self.text.model_dump()),
            include_confidence=self.include_confidence,
            source_qualified_tokens=self.source_qualified_tokens,
            attack_fields=tuple(self.attack_fields),
            metadata_allowlist=tuple(self.metadata_allowlist),
            ocsf_full_path_key=self.ocsf_full_path_key,
            on_empty_vocabulary=EmptyVocabularyStrategy(self.on_empty_vocabulary),
        )


class ModelConfig(_Base):
    name: Literal["lightgbm", "logreg"] = "lightgbm"
    params: dict[str, Any] = {}


class EvalConfig(_Base):
    primary_metric: str = "auto_dispose_rate"
    # Max share of should-investigate alerts we accept auto-disposing (missed-investigation rate).
    max_missed_investigation_rate: float = 0.01
    # Diagnostic breakdowns only; never model features.
    group_cols: list[str] = ["source", "tenant_id"]


class MlflowConfig(_Base):
    tracking_uri: str = "sqlite:///experiments/mlflow.db"


class Config(_Base):
    experiment_name: str = "alert-disposition"
    run_name: str = "baseline"
    seed: int = 42
    data: DataConfig = DataConfig()
    features: FeaturesConfig = FeaturesConfig()
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
