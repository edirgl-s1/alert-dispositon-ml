"""Schema-independent canonical alert representation."""

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum

type Scalar = str | int | float | bool


class ProcessOrigin(StrEnum):
    """Where a process object came from in the source alert."""

    PRIMARY = "primary"
    EVIDENCE = "evidence"


@dataclass(frozen=True)
class CanonicalProcess:
    """One observed process.

    Attributes:
        origin: Primary alert process or an evidence process.
        file_name: File basename as reported. Never treated as a full path.
        file_path: Full path, only when taken from a validated source field.
        cmd_line: Raw command line (never logged, never executed).
    """

    origin: ProcessOrigin
    file_name: str | None = None
    file_path: str | None = None
    cmd_line: str | None = None

    @property
    def is_observed(self) -> bool:
        """Whether any process data is present."""
        return any(
            v is not None and v != "" for v in (self.file_name, self.file_path, self.cmd_line)
        )


@dataclass(frozen=True)
class CanonicalAlert:
    """Adapter output consumed by feature extraction.

    ``None`` always means "not observed", never a benign default.

    Attributes:
        analytic_category: Detector category.
        confidence: Source confidence value as a string.
        title: Alert title.
        descriptions: Distinct analytic descriptions, in source order.
        message: Alert message.
        processes: Primary then evidence processes (bounded).
        evidence_count: Total evidence entries before truncation; None if
            the source had no evidence list.
        attacks: Raw scalar attack values, one dict per attack entry.
        metadata: Allowlisted scalar metadata.
        source_format: Name of the adapter that produced this alert.
        as_of: Moment the alert snapshot is claimed to represent.
        provenance: Canonical field to source path, for audits.
        truncated: Names of inputs truncated by bounds.
    """

    analytic_category: str | None = None
    confidence: str | None = None
    title: str | None = None
    descriptions: tuple[str, ...] = ()
    message: str | None = None
    processes: tuple[CanonicalProcess, ...] = ()
    evidence_count: int | None = None
    attacks: tuple[dict[str, Scalar], ...] = ()
    metadata: dict[str, Scalar] = field(default_factory=dict)
    source_format: str = ""
    as_of: datetime | None = None
    provenance: dict[str, str] = field(default_factory=dict)
    truncated: tuple[str, ...] = ()

    def signals(self) -> "CanonicalAlert":
        """Copy with source-specific bookkeeping removed, for equivalence checks."""
        return replace(self, source_format="", as_of=None, provenance={})
