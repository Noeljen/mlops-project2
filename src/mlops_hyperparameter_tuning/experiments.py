"""Experiment utilities that were originally implemented inside the notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import lightning as L
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .data import GLUEDataModule
from .model import GLUETransformer


@dataclass
class ExperimentConfig:
    run_name: str
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    train_batch_size: int = 32
    warmup_steps: int = 0
    accumulate_grad_batches: int = 1
    seed: int = 42
    project_name: str = "mlops_project2"
    entity_name: str = "noel-jensen-hochschule-luzern"
    model_name: str = "distilbert-base-uncased"
    checkpoint_dir: Optional[str] = None

    def to_kwargs(self) -> Dict[str, object]:
        return self.__dict__.copy()


def _log_and_return_val_metric(trainer, model, datamodule) -> float:
    """Validate the model and log a common WandB-friendly metric."""

    try:
        metrics_list = trainer.validate(model=model, dataloaders=None, datamodule=datamodule, verbose=False)
        metrics = metrics_list[0] if metrics_list else {}
    except Exception:  # pragma: no cover - Lightning swallows meaningful info here
        metrics = {}

    serializable_metrics: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            serializable_metrics[key] = float(value)
        else:
            try:
                serializable_metrics[key] = float(value)
            except Exception:  # noqa: BLE001 - fallback for tensors/arrays
                continue

    candidate_keys = [
        "val_accuracy",
        "eval_accuracy",
        "validation_accuracy",
        "mrpc/accuracy",
        "acc",
        "val_acc",
        "val_f1",
        "eval_f1",
        "f1",
    ]
    val = next((float(metrics[k]) for k in candidate_keys if k in metrics), None)

    if val is None:
        for _, value in metrics.items():
            if isinstance(value, (int, float)):
                val = float(value)
                break

    val = val or 0.0
    log_payload = {
        **serializable_metrics,
        "val_metric": val,
        "epoch": getattr(trainer, "current_epoch", 0),
        "trainer/global_step": getattr(trainer, "global_step", 0),
    }
    wandb.log(log_payload)
    return val


def run_experiment(config: ExperimentConfig) -> float:
    """Run a single training loop with the provided parameters."""

    print("\n" + "=" * 50)
    print(f"--- Starte Lauf: {config.run_name} ---")
    print(
        "Parameter: LR={learning_rate}, BS={train_batch_size}, WD={weight_decay}, "
        "Warmup={warmup_steps}, Accum={accumulate_grad_batches}, Seed={seed}".format(**config.to_kwargs())
    )
    print("=" * 50 + "\n")

    logger = WandbLogger(project=config.project_name, entity=config.entity_name, name=config.run_name)
    L.seed_everything(config.seed)

    datamodule = GLUEDataModule(
        model_name_or_path=config.model_name,
        task_name="mrpc",
        train_batch_size=config.train_batch_size,
    )
    datamodule.setup("fit")

    model = GLUETransformer(
        model_name_or_path=config.model_name,
        num_labels=datamodule.num_labels,
        eval_splits=datamodule.eval_splits,
        train_batch_size=config.train_batch_size,
        task_name=datamodule.task_name,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
    )

    callbacks = []
    if config.checkpoint_dir:
        ckpt_dir = Path(config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename=f"{config.run_name}" + "-{epoch:02d}-{val_loss:.3f}",
            save_top_k=1,
            mode="min",
            monitor="val_loss",
        )
        callbacks.append(checkpoint)

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1,
        logger=logger,
        accumulate_grad_batches=config.accumulate_grad_batches,
        enable_progress_bar=False,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)

    val_metric = _log_and_return_val_metric(trainer, model, datamodule)
    wandb.finish()

    print(f"\n--- Lauf beendet: {config.run_name} | val_metric={val_metric:.4f} ---\n")
    return val_metric


EXPERIMENT_GROUPS: Dict[str, List[ExperimentConfig]] = {
    "baseline": [ExperimentConfig(run_name="00_baseline", learning_rate=2e-5)],
    "learning_rate": [
        ExperimentConfig(run_name="01_lr-2e-2", learning_rate=2e-2),
        ExperimentConfig(run_name="02_lr-5e-3", learning_rate=5e-3),
        ExperimentConfig(run_name="03_lr-2e-3", learning_rate=2e-3),
        ExperimentConfig(run_name="04_lr-5e-4", learning_rate=5e-4),
        ExperimentConfig(run_name="05_lr-5e-5", learning_rate=5e-5),
        ExperimentConfig(run_name="06_lr-3e-5", learning_rate=3e-5),
    ],
    "weight_decay": [
        ExperimentConfig(run_name="07_wd-0.001", weight_decay=0.001),
        ExperimentConfig(run_name="08_wd-0.01", weight_decay=0.01),
        ExperimentConfig(run_name="09_wd-0.1", weight_decay=0.1),
    ],
    "batch_size": [
        ExperimentConfig(run_name="10_bs-16", train_batch_size=16),
        ExperimentConfig(run_name="11_bs-64", train_batch_size=64),
    ],
    "learning_rate_schedule": [
        ExperimentConfig(run_name="12_warmup-100", warmup_steps=100),
        ExperimentConfig(run_name="13_warmup-250", warmup_steps=250),
    ],
    "interaction_2way": [
        ExperimentConfig(run_name="14_lr-3e-5_wd-0.01", learning_rate=3e-5, weight_decay=0.01),
        ExperimentConfig(run_name="15_lr-3e-5_bs-16", learning_rate=3e-5, train_batch_size=16),
        ExperimentConfig(run_name="16_lr-5e-5_warmup-250", learning_rate=5e-5, warmup_steps=250),
    ],
    "interaction_multiway": [
        ExperimentConfig(
            run_name="17_lr-5e-5_bs-16_wd-0.01_warmup-250",
            learning_rate=5e-5,
            train_batch_size=16,
            weight_decay=0.01,
            warmup_steps=250,
        ),
        ExperimentConfig(
            run_name="18_lr-3e-5_bs-32_wd-0.001_warmup-100",
            learning_rate=3e-5,
            train_batch_size=32,
            weight_decay=0.001,
            warmup_steps=100,
        ),
    ],
    "final_optimization": [
        ExperimentConfig(
            run_name="19_optimal_lr-5e-5_bs-16_wd-0.01_warmup-100",
            learning_rate=5e-5,
            train_batch_size=16,
            weight_decay=0.01,
            warmup_steps=100,
        ),
    ],
}


def collect_experiments(groups: Optional[Iterable[str]] = None) -> List[ExperimentConfig]:
    """Return ordered experiment configs filtered by group name."""

    if groups is None:
        groups = EXPERIMENT_GROUPS.keys()

    collected: List[ExperimentConfig] = []
    for name in groups:
        collected.extend(EXPERIMENT_GROUPS[name])
    return collected


def run_experiment_plan(groups: Optional[Iterable[str]] = None) -> List[float]:
    """Execute a sequence of runs referenced by group name."""

    experiments = collect_experiments(groups)
    print("\n" + "=" * 70)
    print("🔬 SYSTEMATIC HYPERPARAMETER OPTIMIZATION PLAN")
    print("=" * 70)
    print(f"\n📊 Total Experiments: {len(experiments)}")

    metrics: List[float] = []
    for cfg in experiments:
        metrics.append(run_experiment(cfg))

    return metrics
