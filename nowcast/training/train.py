"""Command-line training entry point.

Example::

    python -m nowcast.training.train \\
        --config nowcast/training/configs/convlstm_small.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import lightning as L
import torch
import yaml

from nowcast.data.datamodule import NowcastDataModule
from nowcast.training.lit_module import LitNowcast


def load_config(path: str | Path) -> dict:
    """Parse a YAML training config into a plain dict."""
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def build(config: dict) -> tuple[NowcastDataModule, LitNowcast]:
    """Instantiate the datamodule and LightningModule from a config dict."""
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    trainer_cfg = config.get("trainer", {})

    if data_cfg.get("source", "synthetic") == "synthetic":
        datamodule = NowcastDataModule.from_synthetic(
            n_hours=int(data_cfg.get("n_hours", 56)),
            batch_size=int(data_cfg.get("batch_size", 4)),
        )
    else:
        datamodule = NowcastDataModule(
            features=data_cfg["features"],
            labels=data_cfg["labels"],
            batch_size=int(data_cfg.get("batch_size", 4)),
        )

    module = LitNowcast(
        backbone_kwargs=model_cfg.get("backbone"),
        hazard_weights=model_cfg.get("hazard_weights"),
        lr=float(model_cfg.get("lr", 3e-4)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
        max_epochs=int(trainer_cfg.get("max_epochs", 20)),
        bce_weight=float(model_cfg.get("bce_weight", 1.0)),
        dice_weight=float(model_cfg.get("dice_weight", 0.5)),
    )
    return datamodule, module


def resolve_logger(run_dir: Path):
    """MLflow logger when MLflow is installed, else a CSV logger, else ``None``."""
    try:
        import mlflow  # noqa: F401

        has_mlflow = True
    except ModuleNotFoundError:
        has_mlflow = False
    try:
        from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
    except ModuleNotFoundError:  # pragma: no cover - logger extras absent
        return None
    if has_mlflow:
        return MLFlowLogger(
            experiment_name="nowcast",
            tracking_uri=f"file:{(run_dir / 'mlruns').as_posix()}",
        )
    return CSVLogger(save_dir=str(run_dir))


def make_trainer(
    config: dict, run_dir: Path, *, fast_dev_run: bool = False
) -> L.Trainer:
    """Build the Lightning ``Trainer`` (mixed precision only when CUDA is present)."""
    if fast_dev_run:
        return L.Trainer(fast_dev_run=True, accelerator="cpu", logger=False)
    trainer_cfg = config.get("trainer", {})
    precision = "16-mixed" if torch.cuda.is_available() else 32
    return L.Trainer(
        max_epochs=int(trainer_cfg.get("max_epochs", 20)),
        accelerator="auto",
        precision=precision,
        default_root_dir=str(run_dir),
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 10)),
        logger=resolve_logger(run_dir),
    )


def main(argv: list[str] | None = None) -> str:
    """Parse args, train, and return the saved checkpoint path."""
    parser = argparse.ArgumentParser(description="Train the SIH nowcasting model.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path, default=Path("runs") / "latest")
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / Path(args.config).name)

    datamodule, module = build(config)
    trainer = make_trainer(config, run_dir, fast_dev_run=args.fast_dev_run)
    trainer.fit(module, datamodule=datamodule)

    checkpoint = run_dir / "last.ckpt"
    trainer.save_checkpoint(str(checkpoint))
    return str(checkpoint)


if __name__ == "__main__":  # pragma: no cover
    main()
