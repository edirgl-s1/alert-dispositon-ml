"""Diagnostics. Reports contain counts and normalized tokens only, never raw alert content."""

import time
import tracemalloc
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from alert_disposition.features.canonical import CanonicalAlert
from alert_disposition.features.extract import FeatureDocument
from alert_disposition.features.pipeline import FeaturePipeline

_TOP_TOKENS = 20
_CANONICAL_FIELDS = ("analytic_category", "confidence", "title", "message")


@dataclass(frozen=True)
class LatencyReport:
    """Batch-size-1 preprocessing cost."""

    p50_ms: float
    p95_ms: float
    peak_kib: float


def _rate(count: int, total: int) -> float:
    return count / total if total > 0 else 0.0


def _token_docs(docs: Sequence[FeatureDocument]) -> dict[str, list[str]]:
    return {
        "path": [d.path_tokens for d in docs],
        "command": [d.command_tokens for d in docs],
        "attack": [d.attack_tokens for d in docs],
    }


def coverage_report(
    pipeline: FeaturePipeline,
    train_docs: Sequence[FeatureDocument],
    alerts: Sequence[CanonicalAlert],
    *,
    held_out_docs: Sequence[FeatureDocument] = (),
    groups: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Summarize feature coverage.

    Args:
        pipeline: Fitted pipeline.
        train_docs: Documents the pipeline was fitted on.
        alerts: Canonical alerts for field presence and source coverage.
        held_out_docs: Optional held-out documents for unseen-token rates.
        groups: Diagnostic-only labels per alert (for example ``{"product":
            [...], "tenant": [...]}``), aligned with ``alerts``. Counted, never
            used as model features. Callers decide whether labels are safe to show.

    Returns:
        A JSON-serializable report.
    """
    matrix = pipeline.transform(train_docs)
    n_rows, n_cols = matrix.shape
    names = pipeline.get_feature_names_out()

    families: Counter[str] = Counter(n.split(":", 1)[0] for n in names)
    presence = {
        f: _rate(sum(getattr(a, f) not in (None, "") for a in alerts), len(alerts))
        for f in _CANONICAL_FIELDS
    }
    presence["has_primary_process_cmd"] = _rate(
        sum(any(p.cmd_line for p in a.processes[:1]) for a in alerts), len(alerts)
    )
    presence["has_full_path"] = _rate(
        sum(any(p.file_path for p in a.processes) for a in alerts), len(alerts)
    )

    empty_rates = {
        channel: _rate(sum(t.strip() == "" for t in texts), len(texts))
        for channel, texts in _token_docs(train_docs).items()
    }
    for channel in {c for d in train_docs for c in d.texts}:
        empty_rates[f"text:{channel}"] = _rate(
            sum(d.texts.get(channel, "").strip() == "" for d in train_docs), len(train_docs)
        )

    vocab: dict[str, dict[str, object]] = {}
    for channel, texts in _token_docs(train_docs).items():
        vec = CountVectorizer(analyzer=str.split, binary=True, min_df=1)
        try:
            counts = vec.fit_transform(texts)
        except ValueError:
            continue
        doc_freq = np.asarray(counts.sum(axis=0)).ravel()
        terms = vec.get_feature_names_out()
        kept = sum(n.startswith(f"{channel}:") for n in names)
        order = np.argsort(-doc_freq, kind="stable")[:_TOP_TOKENS]
        vocab[channel] = {
            "unique_before_min_df": len(terms),
            "unique_after_min_df": kept,
            "top_tokens": [(str(terms[i]), int(doc_freq[i])) for i in order],
        }

    unseen: dict[str, float] = {}
    if len(held_out_docs) > 0:
        for channel, texts in _token_docs(held_out_docs).items():
            fitted = {n.split(":", 1)[1] for n in names if n.startswith(f"{channel}:")}
            seen_tokens = [t for text in texts for t in set(text.split())]
            unseen[channel] = _rate(sum(t not in fitted for t in seen_tokens), len(seen_tokens))

    coverage: dict[str, dict[str, int]] = {
        "source_format": dict(Counter(a.source_format for a in alerts))
    }
    for key, labels in (groups or {}).items():
        coverage[key] = dict(Counter(labels))

    return {
        "shape": [n_rows, n_cols],
        "density": _rate(matrix.nnz, n_rows * n_cols),
        "features_by_family": dict(families),
        "features_by_block": pipeline.feature_counts_by_block(),
        "field_presence": presence,
        "empty_channel_rate": empty_rates,
        "vocabulary": vocab,
        "unseen_token_rate": unseen,
        "coverage": coverage,
        "truncation_counts": dict(Counter(t for d in train_docs for t in d.truncated)),
    }


def benchmark_latency(
    pipeline: FeaturePipeline,
    extract_one: Callable[[object], FeatureDocument],
    payloads: Sequence[object],
) -> LatencyReport:
    """Time batch-size-1 extraction plus transform, and peak traced memory.

    Args:
        pipeline: Fitted pipeline.
        extract_one: Payload to feature document (normalize + extract).
        payloads: Representative payloads.
    """
    timings: list[float] = []
    tracemalloc.start()
    try:
        for payload in payloads:
            start = time.perf_counter()
            pipeline.transform([extract_one(payload)])
            timings.append((time.perf_counter() - start) * 1000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return LatencyReport(
        p50_ms=float(np.percentile(timings, 50)),
        p95_ms=float(np.percentile(timings, 95)),
        peak_kib=peak / 1024,
    )
