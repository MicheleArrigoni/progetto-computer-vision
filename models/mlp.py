"""Train MLP baselines on one or more processed HGCN-ready datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data_hgcn_ready"
DEFAULT_DATASET_DIRS = (
    DEFAULT_DATASET_ROOT / "openneuro",
    DEFAULT_DATASET_ROOT / "brainlat",
)
METHODS = ("granger", "lingam", "pcmci")
CLASS_NAMES = ("CN", "AD", "FTD")
N_CHANNELS = 19
N_NODE_FEATURES = 8
DEFAULT_HIDDEN_SIZE = 64


def dataset_dirs_or_default(paths=None) -> list[Path]:
    """Return explicit dataset paths, or the two default processed datasets."""
    if paths is None:
        paths = DEFAULT_DATASET_DIRS
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return [Path(path) for path in paths]


def load_samples(
    dataset_dirs=None,
    method="granger",
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.str_]]:
    """Load flattened node features plus directed adjacency values."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")

    samples = []
    labels = []
    subject_ids = []

    for dataset_dir in dataset_dirs_or_default(dataset_dirs):
        subjects_dir = dataset_dir / "subjects"
        if not subjects_dir.exists():
            print(f"Warning: missing subjects folder, skipping {subjects_dir}")
            continue

        for subject_dir in sorted(subjects_dir.glob("sub-*")):
            sample_id = f"{dataset_dir.name}:{subject_dir.name}"
            features = np.load(subject_dir / "X.npy").astype(np.float32)
            adjacency = np.load(subject_dir / f"A_{method}_normal.npy").astype(np.float32)

            with open(subject_dir / "y.json", encoding="utf-8") as file:
                label = json.load(file)["label"]

            validate_sample(sample_id, features, adjacency, label)
            directed_edges = adjacency[~np.eye(N_CHANNELS, dtype=bool)]
            samples.append(np.concatenate((features.ravel(), directed_edges)))
            labels.append(label)
            subject_ids.append(sample_id)

    if not samples:
        raise FileNotFoundError("No processed subject samples found")

    return (
        np.stack(samples).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(subject_ids, dtype=np.str_),
    )


def validate_sample(sample_id, features, adjacency, label) -> None:
    """Check that a subject sample matches the model contract."""
    if features.shape != (N_CHANNELS, N_NODE_FEATURES):
        raise ValueError(f"{sample_id}: expected X shape (19, 8), got {features.shape}")
    if adjacency.shape != (N_CHANNELS, N_CHANNELS):
        raise ValueError(
            f"{sample_id}: expected adjacency shape (19, 19), got {adjacency.shape}"
        )
    if not np.isfinite(features).all() or not np.isfinite(adjacency).all():
        raise ValueError(f"{sample_id}: features and adjacency must be finite")
    if type(label) is not int or label not in range(len(CLASS_NAMES)):
        raise ValueError(f"{sample_id}: expected integer label 0..2")


def stratified_train_val_test(labels, seed=42, fractions=(0.70, 0.10, 0.20)):
    """Return stratified train/validation/test indices."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    split_indices = [[], [], []]

    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        counts = class_split_counts(len(class_indices), fractions)

        start = 0
        for split_id, count in enumerate(counts):
            split_indices[split_id].extend(class_indices[start:start + count])
            start += count

    result = []
    for indices in split_indices:
        rng.shuffle(indices)
        result.append(np.asarray(indices, dtype=np.int64))
    return result


def stratified_train_val(labels, seed=42, validation_fraction=0.15):
    """Return stratified train/validation indices for external-test mode."""
    train, val, _ = stratified_train_val_test(
        labels,
        seed=seed,
        fractions=(1.0 - validation_fraction, validation_fraction, 0.0),
    )
    return train, val


def class_split_counts(n_items, fractions):
    """Convert split fractions to integer per-class counts."""
    raw = np.asarray(fractions, dtype=float) * n_items
    counts = np.floor(raw).astype(int)
    remainder = n_items - counts.sum()
    if remainder:
        counts[np.argsort(-(raw - counts))[:remainder]] += 1
    return counts


def make_split_tensors(samples, labels, train, val, test):
    """Normalize with training statistics and return tensors for each split."""
    if any(len(indices) == 0 for indices in (train, val, test)):
        raise ValueError("Train, validation, and test splits must be non-empty")

    train_mean = samples[train].mean(axis=0)
    train_std = samples[train].std(axis=0)
    train_std[train_std < 1e-6] = 1.0

    def tensor_pair(indices):
        x = (samples[indices] - train_mean) / train_std
        y = labels[indices].astype(np.int64)
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y)

    class_weights = compute_class_weights(labels[train])
    return {
        "train": tensor_pair(train),
        "val": tensor_pair(val),
        "test": tensor_pair(test),
        "class_weights": class_weights,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
    }


def compute_class_weights(labels):
    """Return inverse-frequency class weights for weighted cross-entropy."""
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    weights = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    present = counts > 0
    weights[present] = len(labels) / (present.sum() * counts[present])
    return weights


class SimpleMLP(nn.Module):
    """Compact one-hidden-layer classifier."""

    def __init__(self, input_size, output_size, hidden_size=DEFAULT_HIDDEN_SIZE):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)


def train_model(
    model,
    train_data,
    val_data,
    *,
    class_weights=None,
    epochs=1000,
    patience=500,
    lr=1e-3,
    weight_decay=1e-4,
):
    """Train a model with validation-loss early stopping."""
    train_x, train_y = train_data
    val_x, val_y = val_data
    weight = None if class_weights is None else torch.as_tensor(
        class_weights,
        dtype=train_x.dtype,
        device=train_x.device,
    )
    criterion = nn.CrossEntropyLoss(weight=weight)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = {"loss": float("inf"), "epoch": 0, "state": model.state_dict()}
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = val_criterion(model(val_x), val_y).item()

        if val_loss < best["loss"]:
            best = {
                "loss": val_loss,
                "epoch": epoch,
                "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch % 50 == 0:
            print(
                f"Epoch [{epoch}/{epochs}], train loss: {loss.item():.4f}, "
                f"val loss: {val_loss:.4f}"
            )
        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best["state"])
    model.eval()
    print(f"Best validation loss: {best['loss']:.4f} at epoch {best['epoch']}")
    return {
        "epochs_run": epoch,
        "best_epoch": best["epoch"],
        "best_val_loss": best["loss"],
    }


def predict_labels(model, features):
    """Return predicted class IDs."""
    model.eval()
    with torch.no_grad():
        return model(features).argmax(dim=1)


def accuracy(model, data):
    """Return ordinary accuracy."""
    features, labels = data
    predictions = predict_labels(model, features)
    return (predictions == labels).float().mean().item()


def evaluate_model(model, data, class_names=CLASS_NAMES):
    """Return balanced accuracy, confusion matrix, and text report."""
    features, labels = data
    pred = predict_labels(model, features).cpu().numpy()
    true = labels.cpu().numpy()
    names = list(class_names)

    confusion = np.zeros((len(names), len(names)), dtype=np.int64)
    np.add.at(confusion, (true, pred), 1)

    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = confusion.diagonal()
    precision = np.divide(true_positive, predicted, out=np.zeros(len(names)), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros(len(names)), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(names)), where=(precision + recall) > 0)
    balanced = float(recall[support > 0].mean())

    lines = [f"{'class':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}"]
    for idx, name in enumerate(names):
        lines.append(
            f"{name:>10} {precision[idx]:10.2f} {recall[idx]:10.2f} "
            f"{f1[idx]:10.2f} {support[idx]:10d}"
        )
    lines.append(f"Accuracy: {true_positive.sum() / support.sum():.3f}")
    return balanced, confusion, "\n".join(lines)


def train_and_evaluate(
    dataset_dirs=None,
    *,
    train_dataset_dirs=None,
    test_dataset_dirs=None,
    seed=42,
    epochs=1000,
    patience=500,
):
    """Train one MLP per causal method."""
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
        model = SimpleMLP(samples.shape[1], len(CLASS_NAMES))
        training = train_model(
            model,
            dataset["train"],
            dataset["val"],
            class_weights=dataset["class_weights"],
            epochs=epochs,
            patience=patience,
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


def external_test_data(train_dirs, test_dirs, method, seed):
    """Load train dirs and full external test dirs into one indexed array."""
    if not train_dirs or not test_dirs:
        raise ValueError("External testing requires train and test dataset dirs")

    train_x, train_y, train_ids = load_samples(train_dirs, method)
    test_x, test_y, test_ids = load_samples(test_dirs, method)

    train, val = stratified_train_val(train_y, seed=seed)
    test = np.arange(len(train_x), len(train_x) + len(test_x), dtype=np.int64)
    return (
        np.concatenate((train_x, test_x), axis=0),
        np.concatenate((train_y, test_y), axis=0),
        np.concatenate((train_ids, test_ids)),
        train,
        val,
        test,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP baselines.")
    parser.add_argument("--dataset-dir", action="append", type=Path)
    parser.add_argument("--train-dataset-dir", action="append", type=Path)
    parser.add_argument("--test-dataset-dir", action="append", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=500)
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
    )

    output_path = Path("results") / "mlp" / "classifier_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
