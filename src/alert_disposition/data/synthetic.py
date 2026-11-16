"""Synthetic alert generator so the pipeline runs end-to-end without customer data.

Schema is a stand-in; replace/extend once the real S1 alert export schema is fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ["true_positive", "false_positive", "benign"]


def make_synthetic_alerts(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = rng.choice(LABELS, size=n, p=[0.15, 0.55, 0.30])
    is_tp = label == "true_positive"
    is_fp = label == "false_positive"

    rules = np.array([f"rule_{i:02d}" for i in range(20)])
    # TPs concentrate on a few rules, FPs on others -> learnable signal
    rule_p_tp = np.r_[np.full(5, 0.16), np.full(15, 0.2 / 15)]
    rule_p_tp /= rule_p_tp.sum()
    rule_p_other = np.r_[np.full(5, 0.02), np.full(15, 0.9 / 15)]
    rule_p_other /= rule_p_other.sum()
    rule_name = np.where(
        is_tp, rng.choice(rules, n, p=rule_p_tp), rng.choice(rules, n, p=rule_p_other)
    )

    start = pd.Timestamp("2026-01-01")
    created = start + pd.to_timedelta(np.sort(rng.integers(0, 90 * 24 * 3600, n)), unit="s")

    return pd.DataFrame(
        {
            "alert_id": [f"a{i:07d}" for i in range(n)],
            "created_at": created,
            "rule_name": rule_name,
            "severity": rng.choice(
                ["low", "medium", "high", "critical"], n, p=[0.3, 0.4, 0.2, 0.1]
            ),
            "endpoint_os": rng.choice(["windows", "linux", "macos"], n, p=[0.7, 0.2, 0.1]),
            "detection_engine": rng.choice(["static", "behavioral", "lastline", "custom_star"], n),
            "mitre_tactic": rng.choice(
                ["execution", "persistence", "credential_access", "lateral_movement", "discovery"],
                n,
            ),
            "account_id": rng.choice([f"acct_{i}" for i in range(8)], n),
            "process_count": rng.poisson(np.where(is_tp, 12, 5)),
            "network_conn_count": rng.poisson(np.where(is_tp, 8, 3)),
            "file_mod_count": rng.poisson(np.where(is_fp, 1, 6)),
            "risk_score": np.clip(rng.normal(np.where(is_tp, 70, 40), 15), 0, 100),
            "host_alert_count_24h": rng.poisson(np.where(is_fp, 15, 3)),
            "disposition": label,
        }
    )
