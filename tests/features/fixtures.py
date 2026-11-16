"""Synthetic, non-sensitive alerts."""

PS_CMD = (
    "powershell.exe -NoProfile -EncodedCommand "
    "SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQAIABOAGUAdAAuAFcAZQBiAEMAbABpAGUAbgB0ACkA"
)


def user_path(user: str) -> str:
    return rf"C:\Users\{user}\AppData\Local\Temp\update.exe"


def ocsf_alert(
    *,
    user: str = "Alice",
    full_path_key: str | None = "path",
    n_evidence: int = 2,
    **overrides: object,
) -> dict[str, object]:
    file_: dict[str, object] = {"name": "update.exe"}
    if full_path_key is not None:
        file_[full_path_key] = user_path(user)
    alert: dict[str, object] = {
        "analytic": {"category": "Execution", "desc": "Suspicious PowerShell execution"},
        "finding_info": {
            "title": "Suspicious PowerShell",
            "analytic": {"desc": "Suspicious  PowerShell execution"},
        },
        "message": "Host 10.1.2.3 ran a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90",
        "confidence_id": 2,
        "process": {"file": file_, "cmd_line": PS_CMD},
        "evidences": [
            {"process": {"file": {"name": "cmd.exe"}, "cmd_line": "cmd.exe /c whoami"}}
            for _ in range(n_evidence)
        ],
        "status": "resolved",
        "verdict_id": 1,
        "assignee": {"email_addr": "analyst@example.com", "full_name": "A Analyst"},
        "unknown_extra": "ignored",
    }
    alert.update(overrides)
    return alert
