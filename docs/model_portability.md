# Model portability and serving notes

## What is portable
- LightGBM's learned model is platform-independent: a model trained on macOS gives the same
  predictions on Linux/Windows and x86/ARM (up to tiny floating-point differences).
- `libomp` (macOS: `brew install libomp`) is only a *runtime* dependency of LightGBM on the
  machine running it. Linux uses `libgomp`, usually present; slim Docker images may need
  `apt install libgomp1`.

## What is not
`models/<run>.joblib` is a pickle of the whole scikit-learn pipeline (features + classifier).
It loads reliably only when all of these match:
- Same scikit-learn, LightGBM, NumPy, SciPy versions and the same Python minor version.
  `uv.lock` pins them, so deploy from the same lock file.
- The `alert_disposition` package is installed at a compatible version (the pickle references
  our classes by import path).
- Trust: only load pickles we produced ourselves. Loading a pickle can execute code.

## Feature schema
Every training run writes `models/<run>.schema.json` next to the pickle (both are also logged
to MLflow). It is plain JSON: schema version, ordered feature names plus a SHA-256 of them,
per-block counts, the `features:` config that produced them, target definition, input format,
the val-selected auto-dispose threshold, and library versions.
`alert_disposition.models.artifact.load_model(path)` loads the pickle and refuses it if the
feature names or schema version do not match; it warns if library versions differ.
The schema fixes *which* columns exist and in what order. It does not yet carry IDF weights or
normalization rules, so it is not enough on its own to rebuild features outside Python.

## Guidance
- Experiments: joblib is fine.
- Serving (not yet decided): export the LightGBM part with `booster_.save_model("model.txt")`
  (portable text format, stable across versions/platforms). The feature pipeline is harder to
  export; simplest is to serve it from the same locked package in a container.
- Revisit once the serving environment is known.
