"""W&B sweep helpers."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import lightning as L
import wandb
from lightning.pytorch.loggers import WandbLogger

from .data import GLUEDataModule
from .experiments import _log_and_return_val_metric
from .model import GLUETransformer


def build_default_sweep_config() -> Dict[str, object]:
    return {
        "method": "bayes",
        "metric": {"name": "val_metric", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-4},
            "weight_decay": {"distribution": "uniform", "min": 0.0, "max": 0.1},
            "train_batch_size": {"values": [16, 32, 64]},
            "warmup_steps": {"distribution": "int_uniform", "min": 0, "max": 300},
            "seed": {"value": 42},
        },
    }


def _make_sweep_train(model_name: str) -> Callable[[], None]:
    def sweep_train() -> None:
        with wandb.init() as run:
            config = wandb.config
            run.name = (
                f"sweep_lr-{config.learning_rate:.1e}_"
                f"bs-{config.train_batch_size}_"
                f"wd-{config.weight_decay:.3f}_"
                f"warmup-{config.warmup_steps}"
            )

            L.seed_everything(config.seed)
            datamodule = GLUEDataModule(
                model_name_or_path=model_name,
                task_name="mrpc",
                train_batch_size=config.train_batch_size,
            )
            datamodule.setup("fit")

            model = GLUETransformer(
                model_name_or_path=model_name,
                num_labels=datamodule.num_labels,
                eval_splits=datamodule.eval_splits,
                train_batch_size=config.train_batch_size,
                task_name=datamodule.task_name,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                warmup_steps=config.warmup_steps,
            )

            trainer = L.Trainer(
                max_epochs=3,
                accelerator="auto",
                devices=1,
                logger=WandbLogger(),
                accumulate_grad_batches=1,
                enable_progress_bar=False,
            )
            trainer.fit(model, datamodule=datamodule)

            val_metric = _log_and_return_val_metric(trainer, model, datamodule)
            print(f"✅ Sweep Run beendet | val_metric={val_metric:.4f}\n")

    return sweep_train


def run_bayesian_sweep(
    project: str,
    entity: str,
    count: int = 8,
    sweep_config: Optional[Dict[str, object]] = None,
    model_name: str = "distilbert-base-uncased",
) -> str:
    config = sweep_config or build_default_sweep_config()
    sweep_id = wandb.sweep(config, project=project, entity=entity)
    print(f"📍 Sweep ID: {sweep_id}")
    print(
        "🔗 Verfolge den Sweep unter: "
        f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}"
    )
    wandb.agent(sweep_id, function=_make_sweep_train(model_name), count=count)
    return sweep_id
