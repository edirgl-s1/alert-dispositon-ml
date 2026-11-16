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

## Guidance
- Experiments: joblib is fine.
- Serving (not yet decided): export the LightGBM part with `booster_.save_model("model.txt")`
  (portable text format, stable across versions/platforms). The feature pipeline is harder to
  export; simplest is to serve it from the same locked package in a container.
- Revisit once the serving environment is known.
