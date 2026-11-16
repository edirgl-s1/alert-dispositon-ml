# Project notes
- Package manager: uv only (`uv add <pkg>`, `uv add --group dev <pkg>`, `uv run ...`). Never pip install.
- Entry point: `uv run alertml {train,prepare-data}`; configs in `configs/`, pydantic schema in `src/alert_disposition/config.py` (extra keys are rejected — add new config fields there).
- Splits are chronological by default; do not shuffle across time without noting it in `experiments/LOG.md`.
- Never commit anything under `data/` or `models/`. Alert data is sensitive.
- Before finishing: `make lint test`.
