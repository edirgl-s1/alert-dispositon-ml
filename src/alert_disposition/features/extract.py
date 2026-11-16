"""Canonical alert to schema-independent feature document."""

import re
from dataclasses import dataclass, field

from alert_disposition.features.canonical import CanonicalAlert, ProcessOrigin, Scalar
from alert_disposition.features.config import Family, FeatureConfig
from alert_disposition.features.normalize import (
    REDACTED_PREFIX,
    command_info,
    normalize_text,
    path_info,
    redact,
)

MISSING = "missing"
_MAX_CATEGORY_CHARS = 64


@dataclass(frozen=True)
class FeatureDocument:
    """One alert as model-ready channels.

    Attributes:
        structured: Scalar features; absent keys mean not observed.
        path_tokens: Space-separated atomic path tokens.
        command_tokens: Space-separated atomic command tokens.
        attack_tokens: Space-separated validated attack tokens.
        texts: Normalized text per channel.
        truncated: Inputs that hit a bound.
        provenance: Canonical field to source path.
    """

    structured: dict[str, Scalar] = field(default_factory=dict)
    path_tokens: str = ""
    command_tokens: str = ""
    attack_tokens: str = ""
    texts: dict[str, str] = field(default_factory=dict)
    truncated: tuple[str, ...] = ()
    provenance: dict[str, str] = field(default_factory=dict)


def _category(value: str | None) -> str:
    if value is None:
        return MISSING
    cleaned = " ".join(redact(value[:_MAX_CATEGORY_CHARS]).lower().split())
    return "redacted" if REDACTED_PREFIX in cleaned else cleaned


def _token_value(value: Scalar) -> str | None:
    cleaned = re.sub(r"\s+", "_", redact(str(value)[:_MAX_CATEGORY_CHARS]).strip().lower())
    return None if cleaned == "" or REDACTED_PREFIX in cleaned else cleaned


def extract_feature_document(
    alert: CanonicalAlert, config: FeatureConfig | None = None
) -> FeatureDocument:
    """Derive the feature document for one alert.

    Args:
        alert: Canonical alert.
        config: Feature configuration.
    """
    config = config or FeatureConfig()
    limits = config.limits
    truncated: list[str] = list(alert.truncated)
    path_tokens: set[str] = set()
    command_tokens: set[str] = set()
    distinct: set[tuple[str | None, str | None, str | None]] = set()
    evidence_processes = evidence_with_cmd = 0
    has_full_path = False
    has_extension = False
    depth_max: int | None = None
    cmd_len = cmd_flags = cmd_args = 0
    qualified = config.source_qualified_tokens

    for proc in alert.processes:
        if not proc.is_observed:
            continue
        distinct.add((proc.file_name, proc.file_path, proc.cmd_line))
        is_evidence = proc.origin is ProcessOrigin.EVIDENCE
        evidence_processes += is_evidence
        evidence_with_cmd += is_evidence and proc.cmd_line is not None
        infos = [
            path_info(proc.file_name, is_full_path=False, limits=limits),
            path_info(proc.file_path, is_full_path=True, limits=limits),
        ]
        cmd = command_info(proc.cmd_line, limits)
        if cmd.truncated:
            truncated.append("cmd_line")
        cmd_len, cmd_flags, cmd_args = (
            max(cmd_len, cmd.length),
            max(cmd_flags, cmd.flag_count),
            max(cmd_args, cmd.arg_count),
        )
        for info in infos:
            has_full_path |= info.has_full_path
            if info.depth is not None:
                depth_max = max(depth_max or 0, info.depth)
            has_extension |= any(t.startswith("ext:") for t in info.tokens)
            path_tokens.update(info.tokens)
            if qualified:
                path_tokens.update(f"{proc.origin}.{t}" for t in info.tokens)
        command_tokens.update(cmd.tokens)
        if qualified:
            command_tokens.update(f"{proc.origin}.{t}" for t in cmd.tokens)

    structured: dict[str, Scalar] = {
        "analytic_category": _category(alert.analytic_category),
        "process_present": len(distinct) > 0,
        "has_command_line": cmd_len > 0,
        "has_full_path": has_full_path,
        "has_file_extension": has_extension,
        "evidence_list_present": alert.evidence_count is not None,
        "evidence_count": alert.evidence_count or 0,
        "evidence_process_count": evidence_processes,
        "evidence_with_command_count": evidence_with_cmd,
        "distinct_process_count": len(distinct),
        "cmd_len_max": cmd_len,
        "cmd_flag_count_max": cmd_flags,
        "cmd_arg_count_max": cmd_args,
    }
    if depth_max is not None:
        structured["path_depth_max"] = depth_max
    if config.include_confidence:
        structured["confidence"] = _category(alert.confidence)
    for key, value in alert.metadata.items():
        if isinstance(value, str):
            cleaned = _token_value(value)
            if cleaned is not None:
                structured[f"meta_{key}"] = cleaned
        else:
            structured[f"meta_{key}"] = value

    texts = _texts(alert, config, truncated) if Family.TEXT in config.families else {}
    attack_tokens = sorted(
        {
            f"attack_{k}:{v}"
            for attack in alert.attacks
            for k, raw in attack.items()
            if (v := _token_value(raw)) is not None
        }
    )

    return FeatureDocument(
        structured=structured,
        path_tokens=" ".join(sorted(path_tokens)) if Family.PATH in config.families else "",
        command_tokens=(
            " ".join(sorted(command_tokens)) if Family.COMMAND in config.families else ""
        ),
        attack_tokens=" ".join(attack_tokens) if Family.ATTACK in config.families else "",
        texts=texts,
        truncated=tuple(dict.fromkeys(truncated)),
        provenance=dict(alert.provenance),
    )


def _texts(alert: CanonicalAlert, config: FeatureConfig, truncated: list[str]) -> dict[str, str]:
    sources: dict[str, tuple[str | None, ...]] = {
        "title": (alert.title,),
        "description": alert.descriptions,
        "message": (alert.message,),
    }
    out: dict[str, str] = {}
    for channel, fields in config.text_channels.items():
        seen: list[str] = []
        for name in fields:
            for raw in sources.get(name, ()):
                text, was_truncated = normalize_text(raw, config.limits)
                if was_truncated:
                    truncated.append(name)
                if text != "" and text not in seen:
                    seen.append(text)
        out[channel] = " ".join(seen)
    return out
