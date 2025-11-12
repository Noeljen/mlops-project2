"""Simple entry point to launch a single training run via CLI arguments."""

from __future__ import annotations

import argparse
from typing import Optional

from mlops_hyperparameter_tuning.experiments import ExperimentConfig, run_experiment


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DistilBERT fine-tuning with custom hyperparameters")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project-name", default="mlops_project2")
    parser.add_argument("--entity-name", default="noel-jensen-hochschule-luzern")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--wandb-login", action="store_true", help="Call wandb.login() before running")
    parser.add_argument("--wandb-api-key", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if args.wandb_login:
        import wandb

        wandb.login(key=args.wandb_api_key)

    config = ExperimentConfig(
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_batch_size=args.train_batch_size,
        warmup_steps=args.warmup_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        seed=args.seed,
        project_name=args.project_name,
        entity_name=args.entity_name,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
