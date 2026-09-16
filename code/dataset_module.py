"""Assemble, split, and serialize graph samples for HGCN training.

The functions in this module convert raw causal matrices into model-ready graph
representations, attach subject metadata and labels, create subject-level data
splits, and write both consolidated and per-subject artifacts to disk.
"""

import csv
import json
import pickle
from pathlib import Path

import numpy as np


def prepare_adjacency_for_hgcn(A, config):
    """Clean and normalize a causal adjacency matrix for an HGCN.

    Parameters
    ----------
    A:
        Square directed adjacency matrix. Rows represent targets and columns
        represent sources.
    config:
        Pipeline configuration. ``add_self_loops`` controls whether the
        symmetric representation receives a unit diagonal.

    Returns
    -------
    dict
        Two ``float32`` matrices: ``normal`` retains edge direction, whereas
        ``symmetric`` averages both directions. Both use the maximum value of
        the cleaned symmetric matrix as a common normalization factor.

    Notes
    -----
    NaN and infinite values are removed only from the symmetric copy, matching
    the pipeline's existing behavior. Self-loops are added only to that copy.
    """
    A = np.asarray(A, dtype=np.float32)
    A_s = np.asarray(A, dtype=np.float32)
    A_s = np.nan_to_num(A_s, nan=0.0, posinf=0.0, neginf=0.0)

    np.fill_diagonal(A_s, 0.0)

    # An undirected view is required by HGCN variants that assume symmetric
    # message-passing neighborhoods.
    A_s = 0.5 * (A_s + A_s.T)

    max_value = float(A_s.max())
    if max_value > 0:
        A_s = A_s / max_value
        A = A / max_value


    if config["add_self_loops"]:
        np.fill_diagonal(A_s, 1.0)

    return {
        "symmetric": A_s.astype(np.float32),
        "normal": A.astype(np.float32),
    }

def build_subject_graph(
    subject_id,
    label_name,
    X,
    adjacencies,
    preprocessed_epochs,
    quality_report,
    stationarity_report,
    fs,
    config,
):
    """Build the complete graph sample for one subject.

    Parameters
    ----------
    subject_id:
        Unique BIDS-style subject identifier.
    label_name:
        Clinical class name; it must exist in ``config["labels"]``.
    X:
        Node-feature matrix with shape ``(n_channels, n_features)``.
    adjacencies:
        Mapping from causal method to its ``normal`` and ``symmetric`` matrix.
    preprocessed_epochs:
        EEG tensor with shape ``(n_epochs, n_channels, n_timepoints)``.
    quality_report:
        Per-subject preprocessing and rejection statistics.
    stationarity_report:
        Per-subject stationarity statistics.
    fs:
        Sampling frequency in hertz.
    config:
        Shared pipeline configuration.

    Returns
    -------
    dict
        Self-contained subject sample containing features, graph structures,
        encoded label, channel order, metadata, and preprocessed epochs.
    """
    label = config["labels"][label_name]
    X = np.asarray(X, dtype=np.float32)
    preprocessed_epochs = np.asarray(preprocessed_epochs, dtype=np.float32)

    metadata = {
        "fs": float(fs),
        "target_fs": config["target_fs"],
        "n_epochs": int(preprocessed_epochs.shape[0]),
        "n_channels": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "quality_report": quality_report,
        "stationarity_report": stationarity_report,
    }

    return {
        "subject_id": subject_id,
        "x": X,
        "adjacencies": {
            method: {
                matrix_type: matrix.astype(np.float32)
                for matrix_type, matrix in matrices.items()
            }
            for method, matrices in adjacencies.items()
        },
        "label": label,
        "label_name": label_name,
        "channels": config["channel_order"],
        "metadata": metadata,
        "stationarity_report": stationarity_report,
        "preprocessed_epochs": preprocessed_epochs,
    }


def _json_safe(value):
    """Recursively convert NumPy and pathlib values to JSON-compatible types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def make_subject_splits(graphs, test_size=0.10, val_size=0.20, random_state=42):
    """Create reproducible train, validation, and test subject splits.

    Stratified splitting is used when there are at least ten samples and two
    classes. Smaller datasets use a deterministic fallback because stratified
    splitting may be mathematically impossible with very small class counts.

    Parameters
    ----------
    graphs:
        Sequence of subject graph dictionaries.
    test_size, val_size:
        Fractions of the full dataset assigned to test and validation sets.
    random_state:
        Seed passed to scikit-learn for reproducible splits.

    Returns
    -------
    dict
        Subject-ID lists under the keys ``train``, ``val``, and ``test``.
    """
    subject_ids = [graph["subject_id"] for graph in graphs]
    labels = [graph["label"] for graph in graphs]

    if len(graphs) < 10 or len(set(labels)) < 2:
        return {
            "train": subject_ids[:2],
            "val": subject_ids[2:3],
            "test": subject_ids[3:],
        }

    from sklearn.model_selection import train_test_split

    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        subject_ids,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    relative_val_size = val_size / (1.0 - test_size)
    train_ids, val_ids, _, _ = train_test_split(
        train_val_ids,
        train_val_labels,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val_labels,
    )

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def save_hgcn_ready_dataset(graphs, config):
    """Persist the complete HGCN-ready dataset and its metadata.

    The output includes a consolidated ``graphs.pkl``, JSON configuration and
    split files, a tabular summary, and a directory for each subject containing
    features, epochs, adjacency variants, label, and metadata.

    Parameters
    ----------
    graphs:
        Iterable of graph dictionaries returned by :func:`build_subject_graph`.
    config:
        Pipeline configuration containing at least ``output_dir``.

    Returns
    -------
    pathlib.Path
        Directory to which the dataset was written.
    """
    output_dir = Path(config["output_dir"])
    subjects_dir = output_dir / "subjects"

    output_dir.mkdir(parents=True, exist_ok=True)
    subjects_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "graphs.pkl", "wb") as f:
        pickle.dump(graphs, f)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(config), f, indent=2)

    splits = make_subject_splits(graphs)
    with open(output_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id",
                "label",
                "label_name",
                "n_epochs",
                "n_channels",
                "n_features",
            ],
        )
        writer.writeheader()

        for graph in graphs:
            metadata = graph["metadata"]
            writer.writerow({
                "subject_id": graph["subject_id"],
                "label": graph["label"],
                "label_name": graph["label_name"],
                "n_epochs": metadata["n_epochs"],
                "n_channels": metadata["n_channels"],
                "n_features": metadata["n_features"],
            })

    for graph in graphs:
        subject_dir = subjects_dir / graph["subject_id"]
        subject_dir.mkdir(parents=True, exist_ok=True)

        np.save(subject_dir / "X.npy", graph["x"])
        np.save(subject_dir / "preprocessed_epochs.npy", graph["preprocessed_epochs"])

        for method, matrices in graph["adjacencies"].items():
            np.save(
                subject_dir / f"A_{method}_symmetric.npy",
                matrices["symmetric"],
            )
            np.save(
                subject_dir / f"A_{method}_normal.npy",
                matrices["normal"],
            )

        with open(subject_dir / "y.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "label": graph["label"],
                    "label_name": graph["label_name"],
                },
                f,
                indent=2,
            )

        with open(subject_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(graph["metadata"]), f, indent=2)


    return output_dir
