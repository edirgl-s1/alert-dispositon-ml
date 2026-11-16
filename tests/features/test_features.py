import dataclasses
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from alert_disposition.features.adapters import (
    RawJsonAdapter,
    RawJsonMapping,
    UnknownSourceFormatError,
    UnsupportedPayloadError,
    normalize_alert,
)
from alert_disposition.features.canonical import ProcessOrigin
from alert_disposition.features.config import (
    ChannelConfig,
    EmptyVocabularyStrategy,
    FeatureConfig,
    Limits,
)
from alert_disposition.features.extract import FeatureDocument, extract_feature_document
from alert_disposition.features.normalize import command_info, path_info, redact
from alert_disposition.features.pipeline import EmptyVocabularyError, FeaturePipeline
from alert_disposition.features.report import benchmark_latency, coverage_report
from tests.features.fixtures import PS_CMD, ocsf_alert, user_path

LIMITS = Limits()
CFG = FeatureConfig(
    ocsf_full_path_key="path",
    path=ChannelConfig(min_df=1),
    command=ChannelConfig(min_df=1),
    text=ChannelConfig(min_df=1),
)


def doc(payload: object, config: FeatureConfig = CFG) -> FeatureDocument:
    return extract_feature_document(normalize_alert(payload, "ocsf", config=config), config)


def test_minimal_alert_with_all_optional_fields_absent() -> None:
    d = doc({})
    assert d.structured["analytic_category"] == "missing"
    assert d.structured["process_present"] is False
    assert d.structured["evidence_list_present"] is False
    assert d.path_tokens == "" and d.command_tokens == ""


def test_unknown_extra_fields_ignored_and_known_mapped() -> None:
    alert = normalize_alert(ocsf_alert(), "ocsf", config=CFG)
    assert alert.analytic_category == "Execution"
    assert alert.title == "Suspicious PowerShell"
    assert len(alert.descriptions) == 1
    assert alert.processes[0].origin is ProcessOrigin.PRIMARY
    assert alert.evidence_count == 2
    assert "unknown_extra" not in repr(alert)


def test_non_ocsf_mapping_gives_equivalent_canonical_signals() -> None:
    raw = {
        "detector": {"kind": "Execution"},
        "name": "Suspicious PowerShell",
        "texts": {"a": "Suspicious PowerShell execution"},
        "note": ocsf_alert()["message"],
        "exe": {"base": "update.exe", "full": user_path("Alice"), "args": PS_CMD},
        "related": [{"proc": {"base": "cmd.exe", "args": "cmd.exe /c whoami"}} for _ in range(2)],
        "conf": 2,
    }
    mapping = RawJsonMapping(
        analytic_category=("detector.kind",),
        confidence="conf",
        title="name",
        descriptions=("texts.a",),
        message="note",
        primary_file_name="exe.base",
        primary_file_path="exe.full",
        primary_cmd_line="exe.args",
        evidences="related",
        evidence_file_name="proc.base",
        evidence_file_path="proc.full",
        evidence_cmd_line="proc.args",
    )
    other = RawJsonAdapter(mapping, CFG, "raw").to_canonical(raw, as_of=None)
    ocsf = normalize_alert(ocsf_alert(), "ocsf", config=CFG)
    assert other.signals() == ocsf.signals()


def test_user_paths_share_signals_without_usernames() -> None:
    a = path_info(user_path("Alice"), is_full_path=True, limits=LIMITS)
    b = path_info(user_path("Bob"), is_full_path=True, limits=LIMITS)
    assert a.tokens == b.tokens
    assert {"dir:temp", "dir:local", "dir:appdata", "ext:exe", "loc:user_profile"} <= set(a.tokens)
    assert not any("alice" in t or "bob" in t for t in a.tokens)
    assert a.has_full_path is True


def test_posix_and_env_var_paths() -> None:
    posix = path_info("/home/carol/.cache/tmp/run.sh", is_full_path=True, limits=LIMITS)
    assert "loc:user_profile" in posix.tokens and "ext:sh" in posix.tokens
    assert not any("carol" in t for t in posix.tokens)
    env = path_info(r"%TEMP%\x.exe", is_full_path=True, limits=LIMITS)
    assert "dir:temp" in env.tokens and "loc:user_profile" in env.tokens


def test_bare_file_name_has_no_directory_or_full_path() -> None:
    info = path_info("update.exe", is_full_path=False, limits=LIMITS)
    assert set(info.tokens) == {"basename:update.exe", "ext:exe"}
    assert info.has_full_path is False and info.depth is None
    d = doc(ocsf_alert(full_path_key=None), dataclasses.replace(CFG, ocsf_full_path_key=None))
    assert d.structured["has_full_path"] is False
    assert not any(t.startswith("dir:") for t in d.path_tokens.split())


def test_powershell_command_is_atomic_and_drops_payload() -> None:
    info = command_info(PS_CMD, LIMITS)
    assert {
        "exe:powershell.exe",
        "flag:noprofile",
        "flag:encodedcommand",
        "has:opaque_arg",
    } <= set(info.tokens)
    assert all("SQBF" not in t and "sqbf" not in t.lower() for t in info.tokens)
    assert info.flag_count == 2


@pytest.mark.parametrize(
    ("cmd", "expected"),
    [
        (r'"C:\Program Files\app\x.exe" /c "echo hi"', {"exe:x.exe", "flag:c"}),
        ("bash -c 'curl http://evil.example/x?a=1'", {"exe:bash", "flag:c"}),
        ('cmd.exe /c "unterminated', {"exe:cmd.exe", "flag:c"}),
        ("python3 /tmp/work/run.py --verbose", {"exe:python3", "flag:verbose", "cmd_dir:tmp"}),
    ],
)
def test_command_quoting_modes(cmd: str, expected: set[str]) -> None:
    assert expected <= set(command_info(cmd, LIMITS).tokens)


def test_url_in_command_is_flagged_not_preserved() -> None:
    tokens = command_info("curl https://h.example/p?token=abc123", LIMITS).tokens
    assert "has:url" in tokens
    assert not any("example" in t or "abc123" in t for t in tokens)


@pytest.mark.parametrize(
    "secret",
    [
        "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90",
        "123e4567-e89b-12d3-a456-426614174000",
        "bob@example.com",
        "10.20.30.40",
        "host01.corp.example.com",
        "S-1-5-21-111-222-333-1001",
    ],
)
def test_identifiers_redacted_in_text_and_tokens(secret: str) -> None:
    assert secret not in redact(f"saw {secret} here")
    d = doc(ocsf_alert(message=f"saw {secret} here", metadata={"x": secret}))
    assert secret not in " ".join(d.texts.values())
    cfg = dataclasses.replace(CFG, metadata_allowlist=("k",))
    out = doc(ocsf_alert(metadata={"k": secret}), cfg)
    assert "meta_k" not in out.structured


def test_evidence_aggregates_to_one_row_with_union_and_origin() -> None:
    cfg = dataclasses.replace(CFG, source_qualified_tokens=True)
    d = doc(ocsf_alert(n_evidence=3), cfg)
    assert d.structured["evidence_count"] == 3
    assert d.structured["evidence_process_count"] == 3
    assert d.structured["distinct_process_count"] == 2
    tokens = d.path_tokens.split()
    assert tokens.count("basename:cmd.exe") == 1
    assert "evidence.basename:cmd.exe" in tokens and "primary.dir:temp" in tokens


def test_oversized_inputs_are_bounded_and_flagged() -> None:
    cfg = dataclasses.replace(
        CFG, limits=Limits(max_evidences=2, max_text_chars=20, max_cmd_chars=30)
    )
    d = doc(ocsf_alert(n_evidence=5, message="x" * 500), cfg)
    assert {"evidences", "message", "cmd_line"} <= set(d.truncated)
    assert d.structured["evidence_count"] == 5
    assert d.structured["evidence_process_count"] == 2


def _corpus(n: int) -> list[FeatureDocument]:
    return [doc(ocsf_alert(user=f"user{i}", n_evidence=1 + i % 3)) for i in range(n)]


def test_min_df_counts_documents_and_binary_ignores_repeats() -> None:
    cfg = dataclasses.replace(CFG, path=ChannelConfig(min_df=10))
    with_token = [doc(ocsf_alert(n_evidence=30), cfg) for _ in range(9)]
    without = [doc({"process": {"file": {"name": "other.dll"}}}, cfg) for _ in range(10)]
    token = "path:basename:cmd.exe"
    assert token not in FeaturePipeline(cfg).fit(with_token + without).get_feature_names_out()
    docs = with_token + [doc(ocsf_alert(), cfg)] + without
    pipe = FeaturePipeline(cfg)
    matrix = pipe.fit_transform(docs)
    col = pipe.get_feature_names_out().index(token)
    assert set(matrix[:10, col].toarray().ravel()) == {1.0}


def test_excluded_fields_cannot_change_features() -> None:
    base = doc(ocsf_alert())
    changed = doc(
        ocsf_alert(
            status="new",
            verdict_id=2,
            assignee={"email_addr": "z@example.com"},
            unknown_extra="different",
            tenant_id="t-1",
        )
    )
    assert base == changed


def test_transform_is_stable_for_unseen_tokens_and_categories() -> None:
    pipe = FeaturePipeline(CFG).fit(_corpus(12))
    names = pipe.get_feature_names_out()
    unseen = doc(
        ocsf_alert(
            analytic={"category": "ZZZ-new", "desc": "never seen words"},
            process={
                "file": {"name": "novel.xyz", "path": r"D:\weird\novel.xyz"},
                "cmd_line": "novel --zzz",
            },
        )
    )
    matrix = pipe.transform([unseen, doc({})])
    assert matrix.shape == (2, len(names))
    assert pipe.get_feature_names_out() == names


def test_save_reload_preserves_order_and_values(tmp_path: Path) -> None:
    docs = _corpus(12)
    pipe = FeaturePipeline(CFG).fit(docs)
    pipe.save(tmp_path / "p.joblib", model_version_ref="m-1")
    loaded, ref = FeaturePipeline.load(tmp_path / "p.joblib")
    assert ref == "m-1"
    assert loaded.get_feature_names_out() == pipe.get_feature_names_out()
    assert (loaded.transform(docs) != pipe.transform(docs)).nnz == 0


def test_not_fitted_and_empty_corpus_fail_clearly() -> None:
    with pytest.raises(NotFittedError):
        FeaturePipeline(CFG).transform([doc({})])
    with pytest.raises(ValueError, match="zero documents"):
        FeaturePipeline(CFG).fit([])


def test_empty_vocabulary_strategies() -> None:
    docs = [doc({}) for _ in range(5)]
    with pytest.raises(EmptyVocabularyError, match="path"):
        FeaturePipeline(CFG).fit(docs)
    skip = dataclasses.replace(CFG, on_empty_vocabulary=EmptyVocabularyStrategy.SKIP)
    pipe = FeaturePipeline(skip).fit(docs)
    assert pipe.transform(docs).shape[0] == 5
    assert pipe.feature_counts_by_block()["path"] == 0


@pytest.mark.parametrize("payload", [None, 5, "str"])
def test_bad_payload_types_rejected(payload: object) -> None:
    with pytest.raises(UnsupportedPayloadError):
        normalize_alert(payload, "ocsf")


def test_unknown_source_format() -> None:
    with pytest.raises(UnknownSourceFormatError):
        normalize_alert({}, "nope")


def test_malformed_nested_values_do_not_crash() -> None:
    d = doc({"process": "oops", "evidences": [None, 3, {"process": []}], "finding_info": 7})
    assert d.structured["evidence_count"] == 3
    assert d.structured["process_present"] is False


def test_ablation_experiments_select_families() -> None:
    docs = _corpus(12)
    widths = []
    for name in ("E0", "E1", "E2", "E3"):
        cfg = dataclasses.replace(
            FeatureConfig.for_experiment(name),
            path=ChannelConfig(min_df=1),
            command=ChannelConfig(min_df=1),
            text=ChannelConfig(min_df=1),
            ocsf_full_path_key="path",
        )
        pipe = FeaturePipeline(cfg).fit([doc(ocsf_alert(user=f"u{i}"), cfg) for i in range(12)])
        widths.append(len(pipe.get_feature_names_out()))
    assert widths == sorted(widths) and len(set(widths)) == 4
    assert len(docs) == 12


def test_attack_and_metadata_only_through_allowlist() -> None:
    payload = ocsf_alert(
        attacks=[{"tactic": "Execution", "uid": "abc", "technique": {"x": 1}}],
        metadata={"product": "edr", "secret": "x"},
    )
    assert doc(payload).attack_tokens == ""
    cfg = dataclasses.replace(
        FeatureConfig.for_experiment("E4"),
        attack_fields=("tactic",),
        metadata_allowlist=("product",),
    )
    d = doc(payload, cfg)
    assert d.attack_tokens == "attack_tactic:execution"
    assert d.structured["meta_product"] == "edr"
    assert "meta_secret" not in d.structured


def test_report_has_no_raw_values_and_benchmark_runs() -> None:
    payloads = [ocsf_alert(user=f"user{i}") for i in range(12)]
    alerts = [normalize_alert(p, "ocsf", config=CFG) for p in payloads]
    docs = [extract_feature_document(a, CFG) for a in alerts]
    pipe = FeaturePipeline(CFG).fit(docs)
    report = coverage_report(
        pipe, docs, alerts, held_out_docs=docs[:3], groups={"product": ["edr"] * 12}
    )
    text = repr(report)
    assert report["shape"][1] == len(pipe.get_feature_names_out())  # type: ignore[index]
    assert "user3" not in text and "Alice" not in text and "10.1.2.3" not in text
    lat = benchmark_latency(
        pipe,
        lambda p: extract_feature_document(normalize_alert(p, "ocsf", config=CFG), CFG),
        payloads,
    )
    assert lat.p95_ms >= lat.p50_ms >= 0
    assert np.isfinite(lat.peak_kib)
