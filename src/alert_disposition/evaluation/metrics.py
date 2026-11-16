from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

POSITIVE = "true_positive"


def compute_metrics(
    y_true, y_pred, y_proba, classes: list[str], target_recall: float
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    m: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    for c, pi, ri, fi in zip(classes, p, r, f, strict=True):
        m[f"precision_{c}"], m[f"recall_{c}"], m[f"f1_{c}"] = pi, ri, fi
    m.update(auto_close_at_recall(y_true, y_proba, classes, target_recall))
    return m


def auto_close_at_recall(
    y_true, y_proba, classes: list[str], target_recall: float
) -> dict[str, float]:
    """Operational metric: pick the highest P(true_positive) threshold that still catches
    ``target_recall`` of real threats; everything below it is auto-closable.
    Reports the fraction of alerts that could be auto-closed at that safety level."""
    if POSITIVE not in classes:
        return {}
    p_tp = y_proba[:, classes.index(POSITIVE)]
    is_tp = np.asarray(y_true) == POSITIVE
    if is_tp.sum() == 0:
        return {}
    # threshold = quantile of TP scores s.t. target_recall of TPs score >= threshold
    thr = float(np.quantile(p_tp[is_tp], 1 - target_recall))
    return {
        "autoclose_threshold": thr,
        "autoclose_rate": float((p_tp < thr).mean()),
        "autoclose_tp_missed": float((is_tp & (p_tp < thr)).sum()),
    }
