"""Fitted, serializable feature pipeline producing one sparse CSR matrix."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from alert_disposition.features.canonical import Scalar
from alert_disposition.features.config import (
    ChannelConfig,
    EmptyVocabularyStrategy,
    Family,
    FeatureConfig,
)
from alert_disposition.features.extract import FeatureDocument

SCHEMA_VERSION = "1"
_EMPTY_VOCAB_MESSAGES = ("empty vocabulary", "no terms remain")


class EmptyVocabularyError(ValueError):
    """A channel has no vocabulary after ``min_df``/``max_df`` filtering."""


class ArtifactMismatchError(ValueError):
    """A loaded artifact does not match the expected schema."""


@dataclass(frozen=True)
class _Channel:
    """One vectorizer block: how to select its input and name its features."""

    name: str
    family: Family
    prefix: str


type Vectorizer = DictVectorizer | CountVectorizer | TfidfVectorizer


def _token_vectorizer(cfg: ChannelConfig) -> CountVectorizer:
    return CountVectorizer(
        analyzer=str.split,
        binary=True,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        max_features=cfg.max_features,
        dtype=np.float64,
    )


def _text_vectorizer(cfg: ChannelConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        max_features=cfg.max_features,
        dtype=np.float64,
    )


class FeaturePipeline:
    """Fit on training documents, transform everywhere else.

    Blocks, in order: ``structured``, ``path``, ``command``, ``attack``, then
    one ``<channel>`` block per text channel. Feature names are prefixed with
    their block (``structured:``, ``path:``, ``command:``, ``attack:``,
    ``title:`` ...).
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self._channels = self._plan()
        self._vectorizers: dict[str, Vectorizer | None] = {}
        self._names: list[str] = []
        self._widths: dict[str, int] = {}
        self._fitted = False

    def _plan(self) -> list[_Channel]:
        fams = self.config.families
        plan = [_Channel("structured", Family.STRUCTURED, "structured")]
        if Family.PATH in fams:
            plan.append(_Channel("path", Family.PATH, "path"))
        if Family.COMMAND in fams:
            plan.append(_Channel("command", Family.COMMAND, "command"))
        if Family.ATTACK in fams:
            plan.append(_Channel("attack", Family.ATTACK, "attack"))
        if Family.TEXT in fams:
            plan.extend(_Channel(f"text:{c}", Family.TEXT, c) for c in self.config.text_channels)
        return plan

    def _new_vectorizer(self, channel: _Channel) -> Vectorizer:
        match channel.family:
            case Family.STRUCTURED:
                return DictVectorizer(sparse=True, dtype=np.float64)
            case Family.PATH:
                return _token_vectorizer(self.config.path)
            case Family.COMMAND:
                return _token_vectorizer(self.config.command)
            case Family.ATTACK:
                return _token_vectorizer(self.config.attack)
            case Family.TEXT:
                return _text_vectorizer(self.config.text)

    @staticmethod
    def _inputs(
        channel: _Channel, docs: Sequence[FeatureDocument]
    ) -> list[dict[str, Scalar]] | list[str]:
        match channel.family:
            case Family.STRUCTURED:
                return [d.structured for d in docs]
            case Family.PATH:
                return [d.path_tokens for d in docs]
            case Family.COMMAND:
                return [d.command_tokens for d in docs]
            case Family.ATTACK:
                return [d.attack_tokens for d in docs]
            case Family.TEXT:
                return [d.texts.get(channel.prefix, "") for d in docs]

    def fit(self, docs: Sequence[FeatureDocument]) -> "FeaturePipeline":
        """Learn vocabularies, IDF weights and categories from training docs only.

        Raises:
            ValueError: If ``docs`` is empty.
            EmptyVocabularyError: If a channel is empty and the strategy is RAISE.
        """
        if len(docs) == 0:
            raise ValueError("cannot fit on zero documents")
        names: list[str] = []
        self._vectorizers = {}
        self._widths = {}
        for channel in self._channels:
            vec = self._new_vectorizer(channel)
            data = self._inputs(channel, docs)
            try:
                vec.fit(data)  # type: ignore[arg-type]  # sklearn stubs take one input type per vectorizer; _inputs matches it by family
            except ValueError as exc:
                if not any(m in str(exc) for m in _EMPTY_VOCAB_MESSAGES):
                    raise
                if self.config.on_empty_vocabulary is EmptyVocabularyStrategy.RAISE:
                    raise EmptyVocabularyError(
                        f"channel {channel.name!r} has no vocabulary after min_df/max_df"
                    ) from exc
                self._vectorizers[channel.name] = None
                self._widths[channel.name] = 0
                continue
            self._vectorizers[channel.name] = vec
            block_names = [f"{channel.prefix}:{n}" for n in vec.get_feature_names_out()]
            self._widths[channel.name] = len(block_names)
            names.extend(block_names)
        self._names = names
        self._fitted = True
        return self

    def transform(self, docs: Sequence[FeatureDocument]) -> sp.csr_matrix:
        """Vectorize documents with the fitted vocabulary; never refits.

        Raises:
            NotFittedError: If called before ``fit``.
        """
        if not self._fitted:
            raise NotFittedError("FeaturePipeline.fit must be called first")
        blocks: list[sp.spmatrix] = []
        for channel in self._channels:
            vec = self._vectorizers[channel.name]
            if vec is None:
                blocks.append(sp.csr_matrix((len(docs), 0), dtype=np.float64))
            else:
                blocks.append(vec.transform(self._inputs(channel, docs)))  # type: ignore[arg-type]  # see fit
        return sp.hstack(blocks, format="csr", dtype=np.float64)

    def fit_transform(self, docs: Sequence[FeatureDocument]) -> sp.csr_matrix:
        """Fit then transform the training set (training split only)."""
        return self.fit(docs).transform(docs)

    def get_feature_names_out(self) -> list[str]:
        """Deterministic, prefixed feature names in column order."""
        if not self._fitted:
            raise NotFittedError("FeaturePipeline.fit must be called first")
        return list(self._names)

    def feature_counts_by_block(self) -> dict[str, int]:
        """Number of columns per block."""
        return dict(self._widths)

    def save(self, path: Path, *, model_version_ref: str | None = None) -> None:
        """Serialize pipeline, config, schema version and model reference."""
        if not self._fitted:
            raise NotFittedError("FeaturePipeline.fit must be called first")
        joblib.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "model_version_ref": model_version_ref,
                "feature_names": self._names,
                "pipeline": self,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> tuple["FeaturePipeline", str | None]:
        """Load a trusted artifact. Never load files from untrusted sources.

        Returns:
            The pipeline and its model version reference.

        Raises:
            ArtifactMismatchError: On schema version or feature-order mismatch.
        """
        bundle = joblib.load(path)
        if bundle["schema_version"] != SCHEMA_VERSION:
            raise ArtifactMismatchError(f"schema {bundle['schema_version']} != {SCHEMA_VERSION}")
        pipeline = bundle["pipeline"]
        if (
            not isinstance(pipeline, cls)
            or pipeline.get_feature_names_out() != bundle["feature_names"]
        ):
            raise ArtifactMismatchError("feature names do not match stored schema")
        return pipeline, bundle["model_version_ref"]
