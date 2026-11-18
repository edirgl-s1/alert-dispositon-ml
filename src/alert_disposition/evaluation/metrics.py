"""Metrics for the "needs investigation" target (y=1: should have been investigated).

Headline operational metric: the share of alerts we could auto-dispose while keeping the
missed-investigation rate (should-investigate alerts auto-disposed / all should-investigate
alerts) within a limit. Select the threshold on validation data and apply it unchanged to test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

MISSING_GROUP = "(missing)"


def select_threshold(y_true, p, max_missed_rate: float) -> float:
    """TBD: point estimate on val, not a guarantee on new data (docs/open_issues.md #1).

    Highest score cutoff such that auto-disposing ``p < cutoff`` misses at most
    ``max_missed_rate`` of the y=1 alerts. Returns -inf when there are no positives to protect
    against (nothing can be justified), so callers never auto-dispose on an empty positive set."""
    y, p = np.asarray(y_true).astype(int), np.asarray(p, dtype=float)
    pos = np.sort(p[y == 1])
    if len(pos) == 0:
        return float("-inf")
    k = int(np.floor(max_missed_rate * len(pos)))
    return float(pos[k]) if k < len(pos) else float("inf")


def auto_dispose_stats(y_true, p, threshold: float) -> dict[str, float]:
    """Auto-disposal rate and missed investigations at a fixed threshold."""
    y, p = np.asarray(y_true).astype(int), np.asarray(p, dtype=float)
    disposed = p < threshold
    n_pos = int((y == 1).sum())
    missed = int((disposed & (y == 1)).sum())
    return {
        "auto_dispose_threshold": threshold,
        "auto_dispose_rate": float(disposed.mean()) if len(y) else float("nan"),
        "missed_investigations": float(missed),
        "missed_investigation_rate": missed / n_pos if n_pos else float("nan"),
    }


def compute_metrics(y_true, p, threshold: float) -> dict[str, float]:
    """Ranking, probability-quality and operational metrics for one split."""
    y, p = np.asarray(y_true).astype(int), np.asarray(p, dtype=float)
    m: dict[str, float] = {"prevalence": float(y.mean()), "brier": float(brier_score_loss(y, p))}
    if len(np.unique(y)) == 2:
        m["roc_auc"] = float(roc_auc_score(y, p))
        m["pr_auc"] = float(average_precision_score(y, p))
    m.update(auto_dispose_stats(y, p, threshold))
    return m


def metrics_by_group(
    df: pd.DataFrame, y_col: str, p, threshold: float, group_cols: list[str]
) -> dict[str, dict[str, dict[str, float]]]:
    """Per-group n / prevalence / auto-dispose stats at the global threshold (diagnostics only).

    Rows whose group value is null are reported under ``MISSING_GROUP``.
    """
    p = np.asarray(p, dtype=float)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for col in group_cols:
        if col not in df.columns:
            continue
        # Null group values are reported as their own group rather than silently dropped.
        keys = df[col].astype(object).where(df[col].notna(), MISSING_GROUP).to_numpy()
        groups: dict[str, dict[str, float]] = {}
        for value, idx in pd.Series(keys).groupby(keys).indices.items():
            y = df[y_col].to_numpy()[idx]
            groups[str(value)] = {
                "n": float(len(idx)),
                "prevalence": float(y.mean()),
                **auto_dispose_stats(y, p[idx], threshold),
            }
        out[col] = groups
    return out
