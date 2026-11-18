"""Model artifact = pickled pipeline + a portable JSON feature schema next to it.

``<run>.joblib`` is the fitted sklearn pipeline (features + classifier). It only loads under
matching library versions and must come from a trusted source (see docs/model_portability.md).
``<run>.schema.json`` is plain JSON describing the model's input contract: ordered feature
names, the feature config that produced them, the target definition and the library versions.
It can be read without Python, which is what a non-Python runtime (e.g. Kotlin) needs to
check that it builds the same feature vector.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import warnings
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from alert_disposition.config import Config
from alert_disposition.features.pipeline import SCHEMA_VERSION, ArtifactMismatchError

SCHEMA_SUFFIX = ".schema.json"
_TRACKED_PACKAGES = ("alert-disposition", "scikit-learn", "lightgbm", "numpy", "scipy")


def schema_path(model_path: Path) -> Path:
    """``models/run.joblib`` -> ``models/run.schema.json``."""
    return model_path.with_suffix(SCHEMA_SUFFIX)


def names_digest(names: list[str]) -> str:
    """Order-sensitive fingerprint of the feature names, for quick parity checks."""
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for pkg in _TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def _finite(x: float | None) -> float | None:
    """JSON has no infinity; record a non-finite threshold as null."""
    return x if x is not None and math.isfinite(x) else None


def build_schema(
    model: Pipeline, cfg: Config, *, val_threshold: float | None = None
) -> dict[str, Any]:
    """Describe the fitted pipeline's input contract.

    Raises:
        ArtifactMismatchError: If the classifier did not see exactly these features.
    """
    features = model.named_steps["features"]
    names = [str(n) for n in features.get_feature_names_out()]
    n_in = getattr(model.steps[-1][1], "n_features_in_", len(names))
    if n_in != len(names):
        raise ArtifactMismatchError(f"classifier saw {n_in} features, schema has {len(names)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": cfg.run_name,
        "created_at": datetime.now(UTC).isoformat(),
        "model": {"name": cfg.model.name, "params": cfg.model.params},
        "target": {
            "label_col": cfg.data.label_col,
            "positive_class": 1,
            "meaning": "1 = should have been investigated; output is P(needs_investigation)",
        },
        "input": {"source_format": cfg.data.source_format, "payload_col": cfg.data.payload_col},
        "features": {
            "config": cfg.features.model_dump(mode="json"),
            "n_features": len(names),
            "blocks": features.pipeline_.feature_counts_by_block(),
            "names_sha256": names_digest(names),
            "names": names,
        },
        "evaluation": {
            "max_missed_investigation_rate": cfg.evaluation.max_missed_investigation_rate,
            "val_threshold": _finite(val_threshold),
        },
        "versions": _versions(),
    }


def save_model(
    model: Pipeline, cfg: Config, out_dir: Path, *, val_threshold: float | None = None
) -> tuple[Path, Path]:
    """Write ``<run>.joblib`` and ``<run>.schema.json``; returns both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = build_schema(model, cfg, val_threshold=val_threshold)
    model_path = out_dir / f"{cfg.run_name}.joblib"
    joblib.dump(model, model_path)
    sp = schema_path(model_path)
    sp.write_text(json.dumps(schema, indent=2))
    return model_path, sp


def load_model(model_path: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Load a trusted model and verify it against its schema.

    Never load pickles from untrusted sources: loading can execute code.

    Raises:
        FileNotFoundError: If the schema file is missing.
        ArtifactMismatchError: On schema version or feature-order mismatch.
    """
    schema = json.loads(schema_path(model_path).read_text())
    if schema.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactMismatchError(f"schema {schema.get('schema_version')} != {SCHEMA_VERSION}")
    model = joblib.load(model_path)
    names = [str(n) for n in model.named_steps["features"].get_feature_names_out()]
    expected = schema["features"]
    if names != expected["names"] or names_digest(names) != expected["names_sha256"]:
        raise ArtifactMismatchError("model features do not match its schema")
    current = _versions()
    drift = {
        k: (v, current.get(k))
        for k, v in schema.get("versions", {}).items()
        if k != "alert-disposition" and v != current.get(k)
    }
    if drift:
        warnings.warn(f"library versions differ from training: {drift}", stacklevel=2)
    return model, schema
