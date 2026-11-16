import numpy as np
import pandas as pd

from alert_disposition.evaluation.metrics import (
    auto_dispose_stats,
    metrics_by_group,
    select_threshold,
)


def test_threshold_respects_missed_rate():
    y = np.array([1] * 100 + [0] * 100)
    p = np.r_[np.linspace(0.3, 1.0, 100), np.linspace(0.0, 0.5, 100)]
    thr = select_threshold(y, p, 0.05)
    stats = auto_dispose_stats(y, p, thr)
    assert stats["missed_investigation_rate"] <= 0.05
    assert stats["auto_dispose_rate"] > 0.3


def test_no_positives_disposes_nothing():
    thr = select_threshold([0, 0], [0.1, 0.2], 0.01)
    assert auto_dispose_stats([0, 0], [0.1, 0.2], thr)["auto_dispose_rate"] == 0.0


def test_zero_tolerance_misses_nothing():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.4, 0.9, 0.1, 0.2])
    thr = select_threshold(y, p, 0.0)
    assert auto_dispose_stats(y, p, thr)["missed_investigations"] == 0


def test_group_breakdown_uses_global_threshold():
    df = pd.DataFrame({"y": [1, 0, 1, 0], "source": ["EDR", "EDR", "CWS", "CWS"]})
    out = metrics_by_group(df, "y", [0.9, 0.1, 0.8, 0.2], 0.5, ["source", "absent"])
    assert set(out) == {"source"}
    assert out["source"]["EDR"]["auto_dispose_rate"] == 0.5
