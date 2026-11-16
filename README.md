# alert-disposition-ml

ML for automated alert disposition (true positive / false positive / benign) on SentinelOne alerts.

## Quickstart
macOS: LightGBM needs OpenMP — `brew install libomp` (logreg works without it).

```bash
make setup        # uv sync + pre-commit hooks
make train        # trains on synthetic data by default -> logs to MLflow
make mlflow       # browse runs at http://localhost:5000
make test
```

## Layout
| Path | Purpose |
|---|---|
| `configs/` | YAML experiment configs; layered (`-c default.yaml -c model_logreg.yaml`) with `-o a.b=c` overrides |
| `data/{raw,interim,processed,external}` | Gitignored. Real alert exports go in `raw/` as parquet |
| `src/alert_disposition/data` | loading, splitting (chronological by default), synthetic generator |
| `src/alert_disposition/features` | sklearn feature pipelines (fit on train only, saved with the model) |
| `src/alert_disposition/models` | model factory — register new models here |
| `src/alert_disposition/training` | train loop + MLflow logging |
| `src/alert_disposition/evaluation` | metrics, incl. auto-close rate at fixed TP recall |
| `notebooks/` | exploration only |
| `experiments/` | MLflow store + `LOG.md` decision log |
| `models/`, `reports/` | gitignored artifacts / figures |
| `tests/` | pytest |

## Workflow
1. Put real data in `data/raw/alerts.parquet`, set `data.source: parquet`, update `features:` columns in config.
2. New idea → new config file (or `-o` override) → `alertml train` → compare in MLflow → note it in `experiments/LOG.md`.
3. Promote reusable notebook code into `src/`.

## Data handling
Alert data is customer-sensitive: it never gets committed (gitignore + pre-commit guard). Use `.env` (see `.env.example`) for credentials.
