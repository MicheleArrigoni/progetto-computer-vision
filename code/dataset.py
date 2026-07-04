import csv
import json
import pickle
from pathlib import Path

import numpy as np


def prepare_adjacency_for_hgcn(A, config):
    """
    Pulisce e normalizza una matrice di adiacenza prima di passarla alla HGCN.
    """
    A = np.asarray(A, dtype=np.float32)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    np.fill_diagonal(A, 0.0)

    if config["symmetrize_adjacency"]:
        A = 0.5 * (A + A.T)

    max_value = float(A.max())
    if max_value > 0:
        A = A / max_value

    if config["add_self_loops"]:
        np.fill_diagonal(A, 1.0)

    return A.astype(np.float32)


def combine_adjacencies(adjacencies, weights):
    A_combined = None
    total_weight = 0.0

    for method, weight in weights.items():
        if method not in adjacencies:
            continue

        A = np.asarray(adjacencies[method], dtype=np.float32)
        weight = float(weight)

        if A_combined is None:
            A_combined = weight * A
        else:
            A_combined += weight * A

        total_weight += weight

    if A_combined is None or total_weight == 0.0:
        raise ValueError("No adjacency matrices available for combination")

    return (A_combined / total_weight).astype(np.float32)


def build_subject_graph(
    subject_id,
    label_name,
    X,
    adjacencies,
    preprocessed_epochs,
    stationarity_report,
    quality_report,
    fs,
    config,
):
    """
    Costruisce il dizionario finale del grafo associato a un soggetto.
    """
    main_method = config["adjacency_method"]
    if main_method == "combined":
        adj = combine_adjacencies(
            adjacencies,
            config["adjacency_weights"],
        )
    else:
        if main_method not in adjacencies:
            raise ValueError(f"Missing main adjacency method: {main_method}")
        adj = adjacencies[main_method]

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
    }

    return {
        "subject_id": subject_id,
        "x": X,
        "adj": adj.astype(np.float32),
        "adjacencies": {
            method: A.astype(np.float32)
            for method, A in adjacencies.items()
        },
        "label": label,
        "label_name": label_name,
        "channels": config["channel_order"],
        "metadata": metadata,
        "stationarity_report": stationarity_report,
        "preprocessed_epochs": preprocessed_epochs,
    }


def _json_safe(value):
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


def make_subject_splits(graphs, test_size=0.15, val_size=0.15, random_state=42):
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
    """
    Salva su disco il dataset finale pronto per HGCN.
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
        np.save(subject_dir / "A.npy", graph["adj"])
        np.save(subject_dir / "preprocessed_epochs.npy", graph["preprocessed_epochs"])

        for method, A in graph["adjacencies"].items():
            np.save(subject_dir / f"A_{method}.npy", A)

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

        with open(subject_dir / "stationarity_report.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(graph["stationarity_report"]), f, indent=2)

    return output_dir
