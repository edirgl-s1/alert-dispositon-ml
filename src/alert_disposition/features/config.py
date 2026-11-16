"""Configuration for feature extraction and vectorization."""

from dataclasses import dataclass, field
from enum import StrEnum


class Family(StrEnum):
    """Feature families that can be enabled independently."""

    STRUCTURED = "structured"
    PATH = "path"
    COMMAND = "command"
    TEXT = "text"
    ATTACK = "attack"


class EmptyVocabularyStrategy(StrEnum):
    """What fitting does when ``min_df`` leaves a channel with no vocabulary."""

    RAISE = "raise"
    SKIP = "skip"


@dataclass(frozen=True)
class ChannelConfig:
    """Vectorizer settings for one token or text channel.

    ``min_df``/``max_df`` follow scikit-learn: ints are document counts,
    floats are proportions. One alert is one document.
    """

    min_df: int | float = 10
    max_df: int | float = 1.0
    max_features: int | None = None
    ngram_range: tuple[int, int] = (1, 2)


@dataclass(frozen=True)
class Limits:
    """Bounds on per-alert work."""

    max_evidences: int = 50
    max_text_chars: int = 4096
    max_cmd_chars: int = 4096
    max_cmd_args: int = 256
    max_path_chars: int = 1024
    max_path_parts: int = 32


EXPERIMENT_FAMILIES: dict[str, frozenset[Family]] = {
    "E0": frozenset({Family.STRUCTURED}),
    "E1": frozenset({Family.STRUCTURED, Family.PATH}),
    "E2": frozenset({Family.STRUCTURED, Family.PATH, Family.COMMAND}),
    "E3": frozenset({Family.STRUCTURED, Family.PATH, Family.COMMAND, Family.TEXT}),
    "E4": frozenset(Family),
}

DEFAULT_TEXT_CHANNELS: dict[str, tuple[str, ...]] = {
    "title": ("title",),
    "description": ("description", "message"),
}
COMBINED_TEXT_CHANNELS: dict[str, tuple[str, ...]] = {
    "text": ("title", "description", "message"),
}


@dataclass(frozen=True)
class FeatureConfig:
    """Top-level feature configuration.

    Attributes:
        families: Enabled feature families.
        path: Path token channel settings.
        command: Command token channel settings.
        attack: Attack token channel settings.
        text: TF-IDF settings shared by all text channels.
        text_channels: Channel name to canonical text fields
            (``title``, ``description``, ``message``).
        limits: Work bounds.
        include_confidence: Emit source confidence as a category. Off until
            it is confirmed to be available before investigation.
        source_qualified_tokens: Also emit ``primary.``/``evidence.`` prefixed
            copies of path and command tokens.
        attack_fields: Keys read from each ``attacks[]`` dict. Empty until
            real key shapes are inspected.
        metadata_allowlist: Top-level ``metadata`` keys allowed through.
        ocsf_full_path_key: Key under ``process.file`` holding a validated
            full path (for example ``"path"``). None means no full path is
            ever assumed.
        on_empty_vocabulary: Behaviour when a channel has no vocabulary.
    """

    families: frozenset[Family] = EXPERIMENT_FAMILIES["E3"]
    path: ChannelConfig = field(default_factory=ChannelConfig)
    command: ChannelConfig = field(default_factory=ChannelConfig)
    attack: ChannelConfig = field(default_factory=ChannelConfig)
    text: ChannelConfig = field(
        default_factory=lambda: ChannelConfig(max_features=10000, ngram_range=(1, 2))
    )
    text_channels: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_TEXT_CHANNELS)
    )
    limits: Limits = field(default_factory=Limits)
    include_confidence: bool = False
    source_qualified_tokens: bool = False
    attack_fields: tuple[str, ...] = ()
    metadata_allowlist: tuple[str, ...] = ()
    ocsf_full_path_key: str | None = None
    on_empty_vocabulary: EmptyVocabularyStrategy = EmptyVocabularyStrategy.RAISE

    @classmethod
    def for_experiment(cls, name: str) -> "FeatureConfig":
        """Build a config for ablation experiment E0..E4.

        Use ``dataclasses.replace`` to override individual settings.

        Raises:
            KeyError: If the experiment name is unknown.
        """
        return cls(families=EXPERIMENT_FAMILIES[name])
