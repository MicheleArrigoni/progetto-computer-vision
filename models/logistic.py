"""Train logistic-regression baselines on processed HGCN-ready datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

try:
    from .mlp import (
        CLASS_NAMES,
        DEFAULT_DATASET_DIRS,
        METHODS,
        accuracy,
        evaluate_model,
        external_test_data,
        load_samples,
        make_split_tensors,
        stratified_train_val_test,
        train_model,
    )
except ImportError:
    from mlp import (
        CLASS_NAMES,
        DEFAULT_DATASET_DIRS,
        METHODS,
        accuracy,
        evaluate_model,
        external_test_data,
        load_samples,
        make_split_tensors,
        stratified_train_val_test,
        train_model,
    )


class LogisticRegression(nn.Module):
    """Multiclass logistic regression implemented as one linear layer."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def train_and_evaluate(
    dataset_dirs=None,
    *,
    train_dataset_dirs=None,
    test_dataset_dirs=None,
    seed=32,
    epochs=500,
    patience=50,
    lr=1e-2,
    weight_decay=1e-4,
):
    """Train one logistic-regression model per causal method."""
    external_test = train_dataset_dirs is not None or test_dataset_dirs is not None
    results = {}

    for method in METHODS:
        if external_test:
            samples, labels, subject_ids, train, val, test = external_test_data(
                train_dataset_dirs,
                test_dataset_dirs,
                method,
                seed,
            )
            heading = f"{method.upper()} EXTERNAL TEST"
        else:
            samples, labels, subject_ids = load_samples(dataset_dirs, method)
            train, val, test = stratified_train_val_test(labels, seed)
            heading = method.upper()

        dataset = make_split_tensors(samples, labels, train, val, test)
        print(f"\n{heading}")
        print(f"Train / validation / test: {dataset['sizes']}")

        torch.manual_seed(seed)
        model = LogisticRegression(samples.shape[1], len(CLASS_NAMES))
        training = train_model(
            model,
            dataset["train"],
            dataset["val"],
            class_weights=dataset["class_weights"],
            epochs=epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
        )
        balanced, confusion, report = evaluate_model(model, dataset["test"])
        results[method] = {
            "balanced_accuracy": balanced,
            "accuracy": accuracy(model, dataset["test"]),
            "training": training,
            "confusion_matrix": confusion.tolist(),
            "classification_report": report,
            "train_subjects": subject_ids[train].tolist(),
            "validation_subjects": subject_ids[val].tolist(),
            "test_subjects": subject_ids[test].tolist(),
        }
        print(f"Test balanced accuracy: {balanced:.3f}")
        print(report)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Train logistic-regression baselines.")
    parser.add_argument("--dataset-dir", action="append", type=Path)
    parser.add_argument("--train-dataset-dir", action="append", type=Path)
    parser.add_argument("--test-dataset-dir", action="append", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dirs = args.dataset_dir or DEFAULT_DATASET_DIRS
    results = train_and_evaluate(
        dataset_dirs=dataset_dirs,
        train_dataset_dirs=args.train_dataset_dir,
        test_dataset_dirs=args.test_dataset_dir,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_path = Path("results/logistic/logistic_results.json")
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
