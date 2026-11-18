# alert-disposition-ml

Low-cost ML stage for automated alert disposition on SentinelOne alerts. It runs after the
deterministic rules and before LLM evaluation, and predicts `P(needs_investigation)`:
`y=1` means the alert should have been investigated, `y=0` means it could safely be auto-disposed.
This is **not** a TP/FP classifier; a historical TP/FP verdict is not the label. Alerts the model
is unsure about get no ML verdict and continue to the LLM stage.

## Quickstart
macOS: LightGBM needs OpenMP — `brew install libomp` (logreg works without it).

```bash
make setup        # uv sync + pre-commit hooks
make train        # synthetic data (default.yaml + synthetic.yaml overlay) -> logs to MLflow
make mlflow       # browse runs at http://localhost:5000
make test
```

## Layout
| Path | Purpose |
|---|---|
| `configs/` | YAML experiment configs; layered (`-c default.yaml -c model_logreg.yaml`) with `-o a.b=c` overrides. `synthetic.yaml` is for the synthetic generator only |
| `data/{raw,interim,processed,external}` | Gitignored. Real alert exports go in `raw/` as parquet |
| `src/alert_disposition/data` | loading, splitting (chronological by default), synthetic generator |
| `src/alert_disposition/features` | alert adapters (OCSF / raw JSON) -> canonical alert -> sparse structured/path/command/text features (fit on train only, saved with the model) |
| `src/alert_disposition/models` | model factory (register new models here) and artifact save/load with a JSON feature schema |
| `src/alert_disposition/training` | train loop + MLflow logging |
| `src/alert_disposition/evaluation` | metrics, incl. auto-dispose rate at a capped missed-investigation rate |
| `notebooks/` | exploration only |
| `experiments/` | MLflow store + `LOG.md` decision log |
| `models/`, `reports/` | gitignored artifacts / figures |
| `tests/` | pytest |

## Workflow
1. Put real data in `data/raw/alerts.parquet` (columns: `alert` raw payload, `needs_investigation` 0/1, `created_at`, `alert_id`, plus diagnostic `source`/`tenant_id`) and set `data.source: parquet`. Training data must be the population left *after* deterministic rules, with independently adjudicated labels.
2. New idea → new config file (or `-o` override) → `alertml train` → compare in MLflow → note it in `experiments/LOG.md`.
3. Promote reusable notebook code into `src/`.

## Further docs
- `docs/feature_extraction.md`: feature pipeline design and field mappings
- `docs/model_portability.md`: cross-platform notes for saved models
- `docs/open_issues.md`: known gaps deferred until real data (threshold guarantee, bad rows, ...)

## Data handling
Alert data is customer-sensitive: it never gets committed (gitignore + pre-commit guard). Use `.env` (see `.env.example`) for credentials.
