from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from alert_disposition.config import ROOT, load_config

app = typer.Typer(help="Alert disposition ML tooling", no_args_is_help=True)

ConfigOpt = Annotated[
    list[Path] | None, typer.Option("--config", "-c", help="YAML config(s); later override earlier")
]
OverrideOpt = Annotated[list[str] | None, typer.Option("--override", "-o", help="dotted.key=value")]


def _cfg(config, override):
    return load_config(config or [ROOT / "configs/default.yaml"], override)


@app.command()
def train(config: ConfigOpt = None, override: OverrideOpt = None) -> None:
    """Train a model and log the run to MLflow."""
    from alert_disposition.training.train import train as _train

    for k, v in sorted(_train(_cfg(config, override)).items()):
        typer.echo(f"{k:35s} {v:.4f}")


@app.command()
def prepare_data(config: ConfigOpt = None, override: OverrideOpt = None) -> None:
    """Materialize raw data + splits to data/processed (parquet)."""
    from alert_disposition.data.loader import load_raw
    from alert_disposition.data.split import split_data

    cfg = _cfg(config, override)
    out = ROOT / cfg.data.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, df in split_data(load_raw(cfg), cfg).items():
        df.to_parquet(out / f"{name}.parquet", index=False)
        typer.echo(f"{name}: {len(df)} rows -> {out / f'{name}.parquet'}")


if __name__ == "__main__":
    app()
