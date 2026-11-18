# Alert Disposition feature extraction

Subpackage `src/alert_disposition/features/` that turns alert objects into a stable,
sparse CSR matrix for a low-cost "needs investigation" classifier
(`y=1` should have been investigated, `y=0` safely auto-disposable; not TP/FP).
Task brief: `docs/feature_extraction_task.md`. Runtime deps:
scikit-learn, SciPy, NumPy, joblib. LightGBM only for the example.

Training uses it through `features/transformer.py` (`AlertFeatureTransformer`: a DataFrame
column of raw payloads, JSON string or dict, to CSR), configured by `features:` in the YAML
config (`FeaturesConfig` in `config.py`). `alertml train` scores `P(needs_investigation)`: the
auto-dispose threshold is chosen on val (max missed-investigation rate) and applied unchanged
to test, with per-group breakdowns logged as `group_metrics.json`.

## Layout

| Module | Role |
| --- | --- |
| `canonical.py` | `CanonicalAlert` / `CanonicalProcess`; independent of any source schema. `None` = not observed. |
| `adapters.py` | `OCSFAdapter`, `RawJsonAdapter` + `RawJsonMapping` (explicit paths), registry, `normalize_alert`. |
| `normalize.py` | Redaction, path tokens, command tokenizer. Pure and bounded; never executes or decodes input. |
| `extract.py` | `extract_feature_document` -> `FeatureDocument` (structured, path, command, attack, text channels). |
| `pipeline.py` | `FeaturePipeline`: fit/transform/names/save/load, one CSR matrix. |
| `report.py` | `coverage_report`, `benchmark_latency`. Counts and normalized tokens only. |
| `config.py` | `FeatureConfig`: families, per-channel `min_df`/`max_df`/`max_features`, limits, allowlists, E0-E4 presets. |

The adapters accept a dict or any object with `model_dump(mode="json")`, so the
repo's `AlertOCSFData` can be passed directly with no import coupling.

## Usage

```python
from alert_disposition.features.adapters import normalize_alert
from alert_disposition.features.config import FeatureConfig
from alert_disposition.features.extract import extract_feature_document
from alert_disposition.features.pipeline import FeaturePipeline

cfg = FeatureConfig.for_experiment("E3")  # E0..E4 ablations
docs = [extract_feature_document(normalize_alert(p, "ocsf", config=cfg), cfg) for p in train]
pipe = FeaturePipeline(cfg)
X = pipe.fit_transform(docs)  # train split only
X_new = pipe.transform(other_docs)  # never fit_transform
pipe.get_feature_names_out()  # structured:... path:... command:... title:...
pipe.save(
    path, model_version_ref="..."
)  # FeaturePipeline.load(path) -> (pipe, ref); trusted files only
```

Run tests / example from the repo root:

```
uv run pytest tests/features
uv run python -m scripts.train_lightgbm_example
```

The example's LightGBM step needs OpenMP on macOS (`brew install libomp`). The pipeline and tests do not need LightGBM.

## Field mappings (OCSF)

| Canonical | Source |
| --- | --- |
| analytic category | `analytic.category`, else `finding_info.analytic.category` |
| descriptions | `analytic.desc`, `finding_info.analytic.desc` (deduplicated on whitespace/case) |
| title / message | `finding_info.title` / `message` |
| confidence | `confidence_id` (extracted; emitted only if `include_confidence=True`) |
| primary process | `process.file.name`, `process.cmd_line`, full path only from `process.file.<ocsf_full_path_key>` |
| evidence processes | same fields under `evidences[].process`, bounded by `limits.max_evidences` |
| attacks | `attacks[]` keys listed in `attack_fields` (default none) |
| metadata | `metadata` keys listed in `metadata_allowlist` (default none) |

Never read: `status`, `verdict_id`, `assignee.*`, `time`, any other `extra` key.

## Features

- **structured** (`DictVectorizer`): category one-hot (`missing` when absent), presence flags, evidence counts, distinct process count, command length/flag/arg maxima, `path_depth_max` (only with a full path).
- **path** (binary tokens): `dir:`, `ext:`, `basename:`, `loc:user_profile`, `path:unc`. Windows paths lowercased, `%TEMP%`-style variables expanded, user directories collapsed to `loc:user_profile`. A bare `file.name` yields only `basename:`/`ext:`.
- **command** (binary tokens): `exe:`, `flag:`, `has:url`, `has:opaque_arg`, `cmd_dir:`/`cmd_ext:` for absolute-path arguments. Values, URLs, and opaque blobs are dropped.
- **text** (TF-IDF): channels `title` and `description` (description + message); `COMBINED_TEXT_CHANNELS` gives a single channel. Redacted before vectorization.
- **attack** (binary tokens): `attack_<key>:<value>`, empty until `attack_fields` is set.

Redaction covers URLs, emails, GUIDs, SIDs, IPv4, 32/40/64/128-char hashes, long hex, base64-like blobs, user-home components, and FQDN hostnames.

## Missingness and leakage

- Unknown is never fabricated. `has_full_path=False` means not observed. `evidence_list_present` separates "no list" from "empty list".
- Sparse zero is also how absent numerics appear to LightGBM; use the presence flags to disambiguate.
- Excluded fields cannot reach the model (tested). Source, product, and tenant labels appear only in `coverage_report(groups=...)`.
- Reports hold counts and normalized tokens, never raw commands or alert bodies.

## Bounds

`Limits`: evidences (50), text chars (4096), command chars (4096), command args (256), path chars (1024), path parts (32). Hits are listed in `FeatureDocument.truncated` and counted in the report.

## Assumptions / open questions

- Input is OCSF or raw JSON; non-OCSF needs an explicit `RawJsonMapping` (see the equivalence test). Register with `register_adapter`.
- No full-path field is validated in `AlertOCSFData`. `ocsf_full_path_key` is None until one is confirmed; tokens from `file.name` never imply a directory.
- `attacks[]` and `metadata` key shapes are uninspected, so both are allowlist-only.
- `confidence_id` availability and mutability before the ML stage is unconfirmed; off by default.
- `time` unit/zone unverified; no time features. `as_of` is recorded on `CanonicalAlert` but not enforced.
- Command tokenizer picks Windows vs POSIX quoting heuristically (backslash or `.exe`-style head). PowerShell abbreviations (`-enc`) are kept as written; no alias table.
- FQDN redaction is heuristic (two or more dots, short alphabetic TLD).
- **Known issue (deferred):** IPv4 redaction also matches four-part version strings whose
  parts are all 0-255 (`Chrome 120.0.0.1`, S1 agent versions like `23.4.2.14`); they become
  `redacted_ip`. In text this only adds a false "IP present" token. In allowlisted metadata it
  is worse: `_token_value` drops any value containing a redaction, so a version field would
  vanish. Options when real data arrives: measure how often matches sit next to version cues;
  context rule in text (`v`/`version`/`build`/`Product/` -> `redacted_version`); per-field
  types for allowlisted metadata (version field -> major version, no IP redaction); optionally
  coarse `redacted_ip_private`/`_public`/`_loopback` tokens. Keep IP redaction the default.
- `min_df=10`, TF-IDF `(1,2)`/10000 are untested starting points. Defaults make small corpora raise `EmptyVocabularyError`; set `on_empty_vocabulary=SKIP` to drop the channel.
- Training data must be the residual population after deterministic rules, with independently adjudicated investigation-necessity labels. Out of scope: orchestrator, thresholds, calibration, embeddings, prevalence features.
