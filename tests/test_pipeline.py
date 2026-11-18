import json

import numpy as np
import pydantic
import pytest

from alert_disposition.config import ROOT, load_config
from alert_disposition.data.split import split_data
from alert_disposition.data.synthetic import make_synthetic_alerts
from alert_disposition.features.pipeline import ArtifactMismatchError
from alert_disposition.models.artifact import load_model, schema_path
from alert_disposition.training.train import train


@pytest.fixture
def cfg(tmp_path):
    return load_config(
        [ROOT / "configs/default.yaml"],
        [
            "data.n_synthetic=800",
            "model.params.n_estimators=20",
            f"mlflow.tracking_uri=sqlite:///{tmp_path}/mlflow.db",
        ],
    )


def test_default_config_assumes_no_full_path():
    assert load_config([ROOT / "configs/default.yaml"]).features.ocsf_full_path_key is None
    synthetic = load_config([ROOT / "configs/default.yaml", ROOT / "configs/synthetic.yaml"])
    assert synthetic.features.ocsf_full_path_key == "path"


def test_override_and_validation():
    c = load_config([ROOT / "configs/default.yaml"], ["model.params.n_estimators=7"])
    assert c.model.params["n_estimators"] == 7
    with pytest.raises(pydantic.ValidationError):
        load_config([ROOT / "configs/default.yaml"], ["bogus_key=1"])


def test_time_split_has_no_leakage(cfg):
    s = split_data(make_synthetic_alerts(500), cfg)
    assert s["train"]["created_at"].max() <= s["val"]["created_at"].min()
    assert s["val"]["created_at"].max() <= s["test"]["created_at"].min()
    assert sum(len(v) for v in s.values()) == 500


def _has_lightgbm():
    try:
        from lightgbm import LGBMClassifier

        LGBMClassifier(n_estimators=1, verbose=-1).fit([[0], [1], [0], [1]], [0, 1, 0, 1])
    except OSError:
        return False
    return True


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param([], marks=pytest.mark.skipif(not _has_lightgbm(), reason="libomp missing")),
        ["model.name=logreg", "model.params={C: 1.0, max_iter: 200}"],
    ],
)
def test_train_end_to_end(cfg, tmp_path, extra):
    c = load_config(
        [ROOT / "configs/default.yaml", ROOT / "configs/synthetic.yaml"],
        [
            "data.n_synthetic=800",
            "model.params.n_estimators=20",
            f"mlflow.tracking_uri=sqlite:///{tmp_path}/mlflow.db",
            *extra,
        ],
    )
    res = train(c, save_dir=tmp_path)
    assert res["test_roc_auc"] > 0.6
    assert np.isfinite(res["test_auto_dispose_rate"])
    assert 0 <= res["test_missed_investigation_rate"] <= 1

    # The saved model carries a schema that matches what the classifier was trained on.
    model_path = tmp_path / f"{c.run_name}.joblib"
    model, schema = load_model(model_path)
    names = schema["features"]["names"]
    assert schema["features"]["n_features"] == len(names) == model.steps[-1][1].n_features_in_
    assert names[0].startswith("structured:")
    assert sum(schema["features"]["blocks"].values()) == len(names)
    assert schema["target"]["label_col"] == "needs_investigation"
    assert schema["versions"]["scikit-learn"] is not None
    json.dumps(schema, allow_nan=False)  # strict JSON, readable outside Python

    # A schema that no longer matches the pickle is rejected.
    sp = schema_path(model_path)
    tampered = json.loads(sp.read_text())
    tampered["features"]["names"] = names[::-1]
    sp.write_text(json.dumps(tampered))
    with pytest.raises(ArtifactMismatchError):
        load_model(model_path)
