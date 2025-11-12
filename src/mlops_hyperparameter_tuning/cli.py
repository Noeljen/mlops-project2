"""Command line interface for running experiments or sweeps."""

from __future__ import annotations

import argparse
from typing import List, Optional

import wandb

from .experiments import ExperimentConfig, collect_experiments, run_experiment, run_experiment_plan
from .sweeps import run_bayesian_sweep


def _add_shared_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project-name", default="mlops_project2")
    parser.add_argument("--entity-name", default="noel-jensen-hochschule-luzern")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--checkpoint-dir", default=None)


def _maybe_login(login: bool, api_key: Optional[str]) -> None:
    if login:
        wandb.login(key=api_key)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MLOps hyperparameter tuning utilities")
    parser.add_argument("--wandb-login", action="store_true", help="Call wandb.login() before running")
    parser.add_argument("--wandb-api-key", default=None, help="Optional WandB API key")

    subparsers = parser.add_subparsers(dest="command", required=True)

    single_parser = subparsers.add_parser("run-single", help="Run a single experiment")
    _add_shared_experiment_args(single_parser)

    plan_parser = subparsers.add_parser("run-plan", help="Run a list of predefined experiments")
    plan_parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Subset of experiment groups to run (default: all groups)",
    )

    sweep_parser = subparsers.add_parser("run-sweep", help="Launch a WandB Bayesian sweep")
    sweep_parser.add_argument("--project", default="mlops_project2")
    sweep_parser.add_argument("--entity", default="noel-jensen-hochschule-luzern")
    sweep_parser.add_argument("--count", type=int, default=8)
    sweep_parser.add_argument("--model-name", default="distilbert-base-uncased")

    args = parser.parse_args(argv)
    _maybe_login(args.wandb_login, args.wandb_api_key)

    if args.command == "run-single":
        cfg = ExperimentConfig(
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
        run_experiment(cfg)
    elif args.command == "run-plan":
        run_experiment_plan(groups=args.groups)
    elif args.command == "run-sweep":
        run_bayesian_sweep(
            project=args.project,
            entity=args.entity,
            count=args.count,
            model_name=args.model_name,
        )
    else:  # pragma: no cover - defensive
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
