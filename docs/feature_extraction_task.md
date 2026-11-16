# Alert Disposition — Feature Extraction Pipeline

## Coding-agent implementation brief

### 1. Objective

Implement a reusable, configurable Python feature-extraction pipeline that converts security-alert objects into a **stable, sparse, model-ready feature matrix** for a candidate low-cost Alert Disposition classifier.

The first intended classifier is **LightGBM** (`LGBMClassifier`); logistic regression and CatBoost may be used for comparison. **The primary deliverable is the feature pipeline, not a fully deployed classifier or service.** Include a minimal LightGBM integration example, but prioritize extraction, reproducibility, tests, and diagnostics.

The pipeline must work on alerts that survive earlier deterministic disposition rules. It must support the currently proposed OCSF/Pydantic input while **not assuming OCSF will always be present**. It must also be extensible to other alert representations when their mappings become known.

### 2. Product and ML context (important)

The broader Alert Disposition/auto-triage design evaluates eligible UAM alerts in an ordered chain:

1. Customer-defined deterministic rules.
2. SentinelOne-managed deterministic rules.
3. **Candidate ML classification stage** (this work supports that stage).
4. LLM-based evaluation if earlier stages return no verdict.

A stage can return a disposition verdict with supporting evidence, or **no verdict** to continue the chain. The ML stage is an optional, low-cost filter before expensive LLM evaluation. The initial POC focuses on EDR/CWS alerts. Do **not** implement the orchestrator, messaging, LLM, or disposition processor as part of this task.

**Crucial label definition:**

- `y = 1`: The alert **should have been investigated** / requires review.
- `y = 0`: The alert **could safely have been automatically disposed of** without investigation.
- This is **not** a true-positive/false-positive classifier. TP/FP describes the outcome of an investigation conducted by a human or Purple AI, and is orthogonal to whether an investigation was warranted. A historical TP/FP verdict must **not** be blindly mapped to the new target.
- The model should eventually return a score such as `P(needs_investigation)` via `predict_proba`. A separately selected, conservative threshold may allow high-confidence auto-disposition; uncertain alerts should return **no ML verdict** and continue to the LLM stage. Thresholding/policy is outside this feature-extraction task.

**Training-distribution requirement:** Learn and evaluate on the **residual alert population after the deterministic stages**, not indiscriminately on all historical alerts. The rules have already removed some alerts that the ML stage would never see. The extraction code itself should be usable on any input, but training/evaluation datasets must be constructed accordingly.

### 3. Proposed inputs

The tentative typed input is `AlertOCSFData` (a Pydantic `BaseModel`, with `extra="allow"`). It is an intentionally **incomplete** view of the underlying OCSF payload. Relevant current fields include:

| Source field | Current type / shape | Planned treatment |
| --- | --- | --- |
| `analytic.category` | Optional string | Categorical feature |
| `analytic.desc` | Optional string | Alert-text vectorization |
| `finding_info.title` | Optional string | Alert-text vectorization |
| `finding_info.analytic.category`, `.desc` | Optional nested analytic | Preserve source; deduplicate equivalent values cautiously |
| `message` | Optional string | Alert-text vectorization |
| `confidence_id` | Optional enum/int | Source confidence as category, **only if known to be available pre-investigation** |
| `process.cmd_line` | Optional string | Command tokenizer + structural properties |
| `process.file.name` | Optional string | File basename / extension; use directory tokens **only if a full path actually exists** |
| `evidences[].process` | Optional repeated process objects | Aggregated counts + process/command/path tokens |
| `attacks[]` | Optional list of untyped dictionaries | Add recognized tactic/technique tokens **only after inspecting actual keys** |
| `metadata` | Optional untyped dictionary | Strict allowlist, after investigating actual contents |
| `time` | Optional int | Optional derived numeric/time features after verifying units and semantics |
| `status` | Optional string | **Exclude initially: workflow/lifecycle leakage risk** |
| `verdict_id` | Optional enum/int | **Exclude initially: downstream outcome leakage risk** |
| `assignee.email_addr`, `.full_name` | Optional strings | **Exclude: identity and workflow leakage risk** |

`ProcessOCSFData` contains `file` and `cmd_line`. `FileOCSFData` currently defines only `name`. `EvidencesOCSFData` contains an optional `process`. These classes and the other nested OCSF classes allow additional fields; **do not flatten or model all `extra` fields automatically**.

**Open input-format question:** The runtime payload may be OCSF or some other JSON shape. Implement an adapter boundary:

- `OCSFAdapter` maps explicitly known Pydantic/OCSF fields into a canonical intermediate representation.
- A `RawJsonAdapter`/mapping-based adapter is extensible, but must use **explicit, tested mappings** for any non-OCSF schema. Do not guess that similarly named keys have identical meaning.
- When a field is unavailable, preserve missingness instead of fabricating a default value or treating missing as benign.
- Keep field/source provenance in diagnostics so future investigations can identify which original field produced a feature.
- Keep the canonical representation independent of `AlertOCSFData` so that adding a new adapter need not change vectorizers or model feature names.

Use the repository's existing Pydantic definitions where available; do not duplicate or rewrite the upstream schema unnecessarily.

### 4. Feature extraction philosophy

Prefer **simple, atomic, reusable features**. LightGBM can discover nonlinear interactions between them. For example, generate `dir:temp` and `ext:exe` independently, rather than explicitly generating a hardcoded `exe_from_temp` feature. Avoid writing a large rule engine that decides whether a command or path is suspicious.

Combine four core feature families, and optionally extend to attack metadata:

#### 4.1 Structured / categorical / numerical features

Use explicit, stable scalar values. Suggested initial candidates:

- Analytic category; eligible source confidence; approved detector metadata if available.
- Process present, command line present, usable path present, evidence present.
- Number of evidence entries and number of distinct observed processes, where reliably derivable.
- Command-line length, argument/flag counts, path depth, and filename extension presence where parseable.
- Missingness indicators that distinguish **unknown** from a known negative (`has_full_path`, for example).
- Event hour/day only if meaningful and time representation is verified; never derive features using information from after the alert's classification point.

Use `sklearn.feature_extraction.DictVectorizer` or an equivalent named, sparse-compatible structured representation. Avoid high-cardinality IDs as categorical features. Review any feature for data leakage before allowing it.

#### 4.2 Path and filename features — binary tokens

Use `CountVectorizer(binary=True)` with a **custom analyzer** or with already-tokenized strings. Generate atomic tokens for **actual observed data**, such as:

- `dir:users`, `dir:appdata`, `dir:local`, `dir:temp`, `dir:system32`.
- `ext:exe`, `ext:ps1`, `ext:dll`.
- `basename:powershell.exe` or `basename:cmd.exe` where useful and sufficiently common.
- Optional generic structure tokens (e.g., UNC/network path, user-profile location, path depth) if derivable without guessing.

Use OS-aware normalization (`PureWindowsPath`, `PurePosixPath`, or equivalent), handling separators, case rules, Windows environment-variable forms where resolvable, and common path prefixes. Do not assume all strings are Windows paths. If the payload has only `file.name="update.exe"`, emit filename/extension features but **no directory-derived feature**.

Normalize or suppress unstable high-cardinality components (usernames, user home directory names, hostnames, GUIDs, random identifiers, long hex strings, etc.). For example, paths under `C:\Users\Alice\...` and `C:\Users\Bob\...` should expose the same reusable directory signals without learning `Alice` or `Bob` as predictive identities.

**Never pass SHA256 values, other hash values, or raw unique identifiers into the classifier.** The feature extractor should not accidentally reintroduce them through generic tokenization, metadata flattening, or text fields.

#### 4.3 Command-line features — binary tokens plus a few scalars

Treat commands separately from file paths and general prose. Build a conservative tokenizer/normalizer that exposes signals like:

- Executable/interpreter name (`exe:powershell.exe`, `exe:cmd.exe`, `exe:bash`).
- Command switches and option names (`flag:noprofile`, `flag:encodedcommand`) when reliably tokenized.
- Generic structure indicators such as command length, number of switches, whether a URL is present, or presence of a long opaque/encoded-looking argument.
- Reusable path tokens from file paths embedded in command arguments **only when parsing is reliable**.

Handle Windows and POSIX quoting/escaping deliberately; do not assume one `shlex` mode correctly parses both. Keep tokenization bounded for very long inputs. Normalize/strip secrets, literal tokens, raw URLs with query strings, IPs, user-specific paths, long base64 blobs, and similar unstable or sensitive values. **Do not execute, evaluate, fetch, decode payloads for execution, or otherwise trust command-line content.**

Do not implement an extensive security-detection rule list. Preserve useful atomic flags and patterns for LightGBM to combine.

#### 4.4 Alert textual fields — TF-IDF

Use `TfidfVectorizer` for meaningful natural-language portions of the alert:

- `finding_info.title`.
- `analytic.desc` and `finding_info.analytic.desc` when present.
- `message`.

Normalize/redact volatile identifiers before vectorization, including hash-like values. Consider separate text channels (at least title vs description/message), or a combined text channel with field labels. Pick a clear default and make it configurable. Compare word unigrams/bigrams against character n-grams as a later experiment. Avoid silently duplicating the same description across multiple nested fields.

#### 4.5 Evidence and attack aggregation

`evidences` is a list. Produce a single feature row per alert by combining:

- Aggregate counts (evidence count, count with processes, count with commands, etc.).
- **Union** of binary path/command tokens across evidence processes.
- Optional source-qualified tokens distinguishing `primary` process from `evidence` process, if useful; do not lose provenance silently.

Apply reasonable input-size and evidence-count bounds so a very large alert cannot produce unbounded work. Preserve a diagnostic showing when inputs were truncated. Do not assume `attacks[]` has MITRE IDs until actual dictionary shapes have been checked; add technique/tactic tokens only for validated fields.

### 5. Proposed intermediate data and interfaces

An illustrative, schema-independent intermediate feature document:

```python
{
    "structured": {
        "analytic_category": "execution",
        "confidence_id": "medium",
        "evidence_count": 2,
        "has_command_line": True,
        "has_full_path": True,
    },
    "path_tokens": "dir:users dir:appdata dir:local dir:temp ext:exe",
    "command_tokens": "exe:powershell.exe flag:noprofile flag:encodedcommand",
    "title_text": "Suspicious PowerShell execution",
    "description_text": "...",
}
```

This is an **example of the internal feature document, not a claim that all those fields are available in the proposed OCSF class**. In particular, a full executable path must come from a validated full-path field, not be inferred from `file.name`.

Suggested conceptual APIs (adapt names to the repository):

```python
normalize_alert(payload, source_format, *, as_of=None) -> CanonicalAlert
extract_feature_document(canonical_alert) -> FeatureDocument
feature_pipeline.fit(training_feature_documents) -> FeaturePipeline
X = feature_pipeline.transform(feature_documents)  # scipy.sparse.csr_matrix
feature_pipeline.get_feature_names_out() -> list[str]
```

The pipeline may be implemented with `ColumnTransformer`, `FeatureUnion`, `DictVectorizer`, `CountVectorizer`, `TfidfVectorizer`, and small custom sklearn-compatible transformers, or an equivalent clear composition. Avoid introducing a large framework. **One alert corresponds to one document/row** for `min_df` purposes, even when it contains multiple evidence processes.

For path and command channels, initially try:

```python
CountVectorizer(analyzer=str.split, binary=True, min_df=10)
```

For free text, initially try a configurable `TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_features=10000)`. These values are **initial experimental defaults, not validated production settings**. Set `min_df` independently per channel and allow either counts or proportions. `min_df` counts **training documents containing a token**, not raw repetitions in a single alert. Consider `max_df` and `max_features` to bound dimensionality. Do not rely on `min_df` alone to remove usernames/identifiers: normalize them explicitly.

Keep all feature blocks sparse and concatenate into one CSR matrix for LightGBM. Preserve deterministic, prefixed feature names such as `structured:...`, `path:...`, `command:...`, `title:...` to support audits and importance inspection. Avoid expensive dense conversion.

### 6. Training vs. inference invariants

- Fit vocabulary, IDF weights, category mappings, and all learned preprocessing **on the training split only**.
- Validation/test/production use the exact same fitted pipeline via `.transform()`, never `.fit_transform()`.
- Unknown tokens at inference are handled gracefully; missing fields do not crash the service.
- Serialize the fitted pipeline together with a **feature schema/version, config, and model version reference**, and load only trusted artifacts.
- Ensure feature order and dimensions are identical after save/reload.
- Keep normalization deterministic and side-effect-free. No network lookups or costly external enrichments in the initial extractor.
- Feature computation at inference should be bounded and practical for individual alerts (batch size 1). Benchmark actual preprocessing latency and memory.
- Guard against empty inputs and corpora where `min_df` leaves no vocabulary; fail clearly during fitting or use an explicitly documented empty-channel strategy.

### 7. Leakage, privacy, and quality safeguards

**Hard exclusions in the initial implementation:** `status`, `verdict_id`, assignee identity, post-investigation results, analyst notes and actions, subsequent resolution state, human/Purple AI verdicts, and any field created or updated after the classifier's decision time. Do not make these available through generic `extra="allow"` flattening.

`confidence_id` and metadata fields are **conditional**: include only when we have evidence that their values reflect information available at model inference time and are not downstream decisions. Audit all input timestamps and provenance where possible.

Further safeguards:

- No exact SHA256 or other hash, user identity, hostname, tenant ID, alert ID, or other unique/high-cardinality identifier as a model feature.
- Do not log raw command lines or sensitive alert bodies in debug output, reports, or failed-test dumps by default.
- Treat alert text/commands as untrusted data, not instructions.
- Explicit missingness is necessary. `has_full_path = False` means **not observed**, not that the process is known to have avoided risky directories.
- Detect and report input/source coverage by EDR vs CWS, OCSF vs other, detector family, and tenant **in diagnostics only**; do not automatically feed these identifiers or source-availability shortcuts into the model.
- Preserve the exact "as-of" moment used to build training examples. Historical enriched/final-state objects must not silently stand in for pre-investigation alerts.

### 8. Diagnostics and experimentation

Make the feature extractor inspectable. Provide a report or utility that shows:

- Feature names and counts by family; sparse matrix shape/density.
- Per-field presence, null/empty rates, and how often each feature channel yields no tokens.
- Number of unique tokens before/after `min_df`, most frequent tokens, and examples of normalized non-sensitive tokens.
- Unknown/unseen token rates on held-out data, without disclosing raw identities.
- OCSF vs non-OCSF and EDR vs CWS coverage; metrics by tenant may be computed in reports without using IDs as model features.
- Preprocessing p50/p95 latency and memory on representative batch-size-1 alerts, plus truncation counts.

The implementation should make ablation experiments easy:

1. **E0** — structured features only.
2. **E1** — E0 + binary path features.
3. **E2** — E1 + binary command features.
4. **E3** — E2 + TF-IDF alert text.
5. **E4** — E3 + validated attack/metadata features.
6. **Later** — contextual/prevalence features or transformer embeddings, only if justified; these are **out of scope for this first implementation**.

Provide a small example of training `LGBMClassifier` on the extracted matrix and retrieving `predict_proba()` as `P(needs_investigation)`. Full model selection, calibration, threshold optimization, and production deployment are separate tasks.

For eventual evaluation, prioritize **percentage of residual alerts safely auto-disposed at a constrained missed-investigation rate**, plus calibration, precision/recall, cost, and latency, rather than accuracy alone. Use temporal holdouts and inspect performance by alert source/detector/tenant. Labels must be independently defined for investigation necessity, **not simply TP/FP**.

### 9. Tests and acceptance criteria

Implement unit tests with synthetic, non-sensitive alerts (plus integration tests using existing repository fixtures when available):

- [ ] A minimal OCSF alert works with every optional field absent.
- [ ] Known fields map correctly into the canonical object; unknown `extra="allow"` fields are ignored unless explicitly allowlisted.
- [ ] Equivalent OCSF and non-OCSF inputs (using a **tested explicit mapping**) yield equivalent canonical signals.
- [ ] `C:\Users\Alice\AppData\Local\Temp\update.exe` and the analogous Bob path yield shared `dir:...`/`ext:exe` signals **without** exposing usernames.
- [ ] A bare `file.name="update.exe"` yields extension/name features but **not** inferred directory or `has_full_path=True` features.
- [ ] `powershell.exe -NoProfile -EncodedCommand ...` yields expected normalized atomic executable/flag features, without preserving opaque command payloads.
- [ ] Long hashes, GUIDs, emails, hostnames, and other protected identifiers are excluded even when they appear in raw metadata or text.
- [ ] Repeated process evidence aggregates to **one alert row** with deterministic counts and binary token union; primary vs evidence origin is preserved if enabled.
- [ ] With `binary=True`, repeated token occurrences within one alert yield a value of 1. With `min_df=10`, a token must occur in 10 **different training alerts** to be included.
- [ ] Changing excluded fields (`status`, `verdict_id`, `assignee`, downstream outcomes) cannot change the generated model features.
- [ ] `.transform()` never changes the fitted vocabulary or feature count, including for unseen tokens and categories.
- [ ] Save/reload preserves exact feature order and equivalent transformed values.
- [ ] Null, empty, malformed, cross-platform, oversized, and mixed-format inputs have deterministic, documented behavior.
- [ ] Feature extraction uses no network calls, never executes command strings, and has bounded work/memory.
- [ ] Feature names are discoverable and diagnostic reports avoid raw sensitive values.

### 10. Deliverables

1. **Feature-extraction package**: canonical representation, OCSF adapter, an extensible non-OCSF adapter interface, normalization/tokenization, structured/path/command/text extractors, and a fitted pipeline composer.
2. **Configuration** for which feature families are enabled, per-channel `min_df`, `max_df`, `max_features`, truncation limits, and approved metadata mappings.
3. **Tests** covering the acceptance criteria above, with concise synthetic fixtures.
4. **Example script/notebook**: build features from sample alerts, inspect names/matrix, fit/transform, serialize/reload, and optionally train a minimal LightGBM model.
5. **Feature coverage report utility**, including input availability and sparse dimensionality.
6. **README**: architecture, field mappings, usage, feature definitions, missingness behavior, security/leakage rules, and documented assumptions/open questions.

Follow the repository's package layout, coding conventions, testing framework, and supported Python/dependency versions. Prefer standard Python and scikit-learn/SciPy rather than adding unnecessary dependencies. Make the modules easy to integrate into both offline training and single-alert service inference.

### 11. Open questions to document, not invent answers for

- Whether the classifier receives OCSF, raw UAM JSON, or both; exact mapping for a non-OCSF schema.
- Where complete process/executable paths reside (the typed `file.name` field alone does not guarantee one).
- Actual keys and semantics in `attacks[]` and `metadata`; safe allowlist candidates.
- Field presence and freshness at the precise point **after rules but before ML**.
- Whether `confidence_id` is immutable analytic confidence or may be modified during triage.
- Timestamp units/time zone, and whether time-based features are valuable.
- How representative training examples and **independently adjudicated investigation-necessity labels** will be obtained for rule-surviving alerts.
- Whether different source-specific adapters need explicit features or model variants after inspecting empirical coverage.

**Implementation instruction:** Start by examining the existing repository models and sample alert fixtures. Build the core adapter + normalization + vectorizers using only validated fields. Where real schema details are unknown, provide clear extension points and report the limitation; **do not fabricate mappings, downstream labels, security semantics, or production thresholds**.
