# Experiments
- Tracking: MLflow, SQLite store at `experiments/mlflow.db` (gitignored). UI: `make mlflow`.
- Each experiment = a config file in `configs/` layered over `default.yaml`.
- Record conclusions (what was tried, result, decision) in `experiments/LOG.md` — MLflow holds numbers, this holds reasoning.
