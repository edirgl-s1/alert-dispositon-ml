from __future__ import annotations

from pathlib import Path

import pandas as pd

from alert_disposition.config import ROOT, Config
from alert_disposition.data.synthetic import make_synthetic_alerts


def load_raw(cfg: Config) -> pd.DataFrame:
    """Load raw alerts. Real data goes in data/raw/ (gitignored) as parquet."""
    if cfg.data.source == "synthetic":
        return make_synthetic_alerts(cfg.data.n_synthetic, cfg.seed)
    path = Path(cfg.data.raw_path)
    path = path if path.is_absolute() else ROOT / path
    df = pd.read_parquet(path)
    missing = {cfg.data.label_col, cfg.data.payload_col, cfg.data.time_col, cfg.data.id_col} - set(
        df.columns
    )
    if missing:
        raise ValueError(f"Raw data missing required columns: {sorted(missing)}")
    return df
