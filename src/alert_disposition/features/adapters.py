"""Adapters from source payloads to ``CanonicalAlert``.

Every mapping is an explicit path. Fields not named here (``status``,
``verdict_id``, ``assignee``, other ``extra="allow"`` keys) are never read.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from alert_disposition.features.canonical import (
    CanonicalAlert,
    CanonicalProcess,
    ProcessOrigin,
    Scalar,
)
from alert_disposition.features.config import FeatureConfig

type Json = str | int | float | bool | None | list[Json] | dict[str, Json]


@runtime_checkable
class SupportsModelDump(Protocol):
    """Anything shaped like a Pydantic model (no Pydantic import needed)."""

    def model_dump(self, *, mode: str) -> dict[str, Json]:
        """Return a JSON-compatible dict."""
        ...


class Adapter(Protocol):
    """Maps one source payload to a canonical alert."""

    def to_canonical(self, payload: object, *, as_of: datetime | None) -> CanonicalAlert:
        """Convert ``payload``."""
        ...


class UnsupportedPayloadError(TypeError):
    """Payload is neither a mapping nor a model-like object."""


class UnknownSourceFormatError(KeyError):
    """No adapter is registered for the requested source format."""


def as_dict(payload: object) -> dict[str, Json]:
    """Coerce a mapping or Pydantic-like object into a plain dict.

    Raises:
        UnsupportedPayloadError: For any other type.
    """
    if isinstance(payload, SupportsModelDump):
        return payload.model_dump(mode="json")
    if isinstance(payload, Mapping):
        return {str(k): v for k, v in payload.items()}
    raise UnsupportedPayloadError(type(payload).__name__)


def get_path(data: Json, path: str | None) -> Json:
    """Follow a dotted dict path; None on any miss."""
    if path is None:
        return None
    node: Json = data
    for key in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _text(value: Json) -> str | None:
    return value if isinstance(value, str) and value.strip() != "" else None


def _scalar(value: Json) -> Scalar | None:
    return value if isinstance(value, (str, int, float, bool)) else None


def _confidence(value: Json) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    return str(value) if isinstance(value, (str, int)) else None


@dataclass(frozen=True)
class RawJsonMapping:
    """Explicit canonical-field to source-path mapping.

    Paths are dotted. ``evidences`` names a list; the ``evidence_*`` paths
    are relative to each list entry.
    """

    analytic_category: tuple[str, ...] = ()
    confidence: str | None = None
    title: str | None = None
    descriptions: tuple[str, ...] = ()
    message: str | None = None
    primary_file_name: str | None = None
    primary_file_path: str | None = None
    primary_cmd_line: str | None = None
    evidences: str | None = None
    evidence_file_name: str | None = None
    evidence_file_path: str | None = None
    evidence_cmd_line: str | None = None
    attacks: str | None = None
    metadata: str | None = None
    extra_provenance: dict[str, str] = field(default_factory=dict)


def ocsf_mapping(config: FeatureConfig) -> RawJsonMapping:
    """Mapping for the proposed ``AlertOCSFData`` shape."""
    key = config.ocsf_full_path_key
    return RawJsonMapping(
        analytic_category=("analytic.category", "finding_info.analytic.category"),
        confidence="confidence_id",
        title="finding_info.title",
        descriptions=("analytic.desc", "finding_info.analytic.desc"),
        message="message",
        primary_file_name="process.file.name",
        primary_file_path=None if key is None else f"process.file.{key}",
        primary_cmd_line="process.cmd_line",
        evidences="evidences",
        evidence_file_name="process.file.name",
        evidence_file_path=None if key is None else f"process.file.{key}",
        evidence_cmd_line="process.cmd_line",
        attacks="attacks",
        metadata="metadata",
    )


class RawJsonAdapter:
    """Mapping-driven adapter; also the base of the OCSF adapter."""

    def __init__(
        self,
        mapping: RawJsonMapping,
        config: FeatureConfig,
        source_format: str,
    ) -> None:
        self._m = mapping
        self._config = config
        self._source_format = source_format

    def to_canonical(self, payload: object, *, as_of: datetime | None = None) -> CanonicalAlert:
        """Build a canonical alert; absent fields stay None."""
        data = as_dict(payload)
        m = self._m
        prov: dict[str, str] = {}
        truncated: list[str] = []

        category = None
        for path in m.analytic_category:
            category = _text(get_path(data, path))
            if category is not None:
                prov["analytic_category"] = path
                break

        descriptions: list[str] = []
        seen: set[str] = set()
        for path in m.descriptions:
            desc = _text(get_path(data, path))
            if desc is not None and " ".join(desc.lower().split()) not in seen:
                seen.add(" ".join(desc.lower().split()))
                descriptions.append(desc)
                prov.setdefault("descriptions", path)

        title = _text(get_path(data, m.title))
        message = _text(get_path(data, m.message))
        confidence = _confidence(get_path(data, m.confidence))
        for name, tvalue, tpath in (
            ("title", title, m.title),
            ("message", message, m.message),
            ("confidence", confidence, m.confidence),
        ):
            if tvalue is not None and tpath is not None:
                prov[name] = tpath

        processes = [
            CanonicalProcess(
                origin=ProcessOrigin.PRIMARY,
                file_name=_text(get_path(data, m.primary_file_name)),
                file_path=_text(get_path(data, m.primary_file_path)),
                cmd_line=_text(get_path(data, m.primary_cmd_line)),
            )
        ]
        evidence_count: int | None = None
        raw_evidences = get_path(data, m.evidences)
        if isinstance(raw_evidences, list):
            evidence_count = len(raw_evidences)
            limit = self._config.limits.max_evidences
            if evidence_count > limit:
                truncated.append("evidences")
            for entry in raw_evidences[:limit]:
                processes.append(
                    CanonicalProcess(
                        origin=ProcessOrigin.EVIDENCE,
                        file_name=_text(get_path(entry, m.evidence_file_name)),
                        file_path=_text(get_path(entry, m.evidence_file_path)),
                        cmd_line=_text(get_path(entry, m.evidence_cmd_line)),
                    )
                )
            prov["evidences"] = m.evidences or ""

        attacks: list[dict[str, Scalar]] = []
        raw_attacks = get_path(data, m.attacks)
        if isinstance(raw_attacks, list) and len(self._config.attack_fields) > 0:
            for entry in raw_attacks[: self._config.limits.max_evidences]:
                picked = {
                    k: v
                    for k in self._config.attack_fields
                    if isinstance(entry, dict) and (v := _scalar(entry.get(k))) is not None
                }
                if len(picked) > 0:
                    attacks.append(picked)
            prov["attacks"] = m.attacks or ""

        metadata: dict[str, Scalar] = {}
        raw_meta = get_path(data, m.metadata)
        if isinstance(raw_meta, dict):
            for k in self._config.metadata_allowlist:
                mvalue = _scalar(raw_meta.get(k))
                if mvalue is not None:
                    metadata[k] = mvalue
                    prov[f"metadata.{k}"] = f"{m.metadata}.{k}"

        return CanonicalAlert(
            analytic_category=category,
            confidence=confidence,
            title=title,
            descriptions=tuple(descriptions),
            message=message,
            processes=tuple(processes),
            evidence_count=evidence_count,
            attacks=tuple(attacks),
            metadata=metadata,
            source_format=self._source_format,
            as_of=as_of,
            provenance={**prov, **m.extra_provenance},
            truncated=tuple(truncated),
        )


class OCSFAdapter(RawJsonAdapter):
    """Adapter for the proposed OCSF/Pydantic alert shape."""

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(ocsf_mapping(config), config, source_format="ocsf")


type AdapterFactory = Callable[[FeatureConfig], Adapter]

_REGISTRY: dict[str, AdapterFactory] = {"ocsf": OCSFAdapter}


def register_adapter(source_format: str, factory: AdapterFactory) -> None:
    """Register an adapter factory for a new source format."""
    _REGISTRY[source_format] = factory


def normalize_alert(
    payload: object,
    source_format: str,
    *,
    as_of: datetime | None = None,
    config: FeatureConfig | None = None,
) -> CanonicalAlert:
    """Convert a source payload into a canonical alert.

    Raises:
        UnknownSourceFormatError: If ``source_format`` has no adapter.
        UnsupportedPayloadError: If the payload type cannot be read.
    """
    factory = _REGISTRY.get(source_format)
    if factory is None:
        raise UnknownSourceFormatError(source_format)
    return factory(config or FeatureConfig()).to_canonical(payload, as_of=as_of)
