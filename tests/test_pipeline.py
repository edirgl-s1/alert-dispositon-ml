import numpy as np
import pydantic
import pytest

from alert_disposition.config import ROOT, load_config
from alert_disposition.data.split import split_data
from alert_disposition.data.synthetic import make_synthetic_alerts
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
        [ROOT / "configs/default.yaml"],
        [
            "data.n_synthetic=800",
            "model.params.n_estimators=20",
            f"mlflow.tracking_uri=sqlite:///{tmp_path}/mlflow.db",
            *extra,
        ],
    )
    res = train(c, save_dir=tmp_path)
    assert res["test_f1_macro"] > 0.4
    assert np.isfinite(res["test_autoclose_rate"])
