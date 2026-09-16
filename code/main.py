"""Run the end-to-end EEG preprocessing and causal-graph export pipeline.

The pipeline first preprocesses every subject to determine a common number of
valid epochs. It then extracts channel features, estimates configured causal
graphs, and serializes HGCN-ready samples. Subject-level failures are isolated
and reported without stopping the remaining workload.
"""

import numpy as np

from config import CONFIG
from dataset_module import (
    build_subject_graph,
    prepare_adjacency_for_hgcn,
    save_hgcn_ready_dataset,
)
from preprocessing import (
    build_node_features,
    filter_and_resample,
    get_subject_label,
    load_subject_raw,
    order_channels,
    preprocess_epochs_for_subject,
    split_in_epochs,
)


def compute_graphs(processed_epochs: np.ndarray) -> dict[str, np.ndarray]:
    """Run every configured causal estimator on one subject's epochs.

    Parameters
    ----------
    processed_epochs:
        Stationary EEG tensor with shape
        ``(n_epochs, n_channels, n_timepoints)``.

    Returns
    -------
    dict
        Mapping from enabled method name to a directed adjacency matrix.

    Notes
    -----
    Imports are local so a causal-analysis dependency is loaded only when its
    method appears in ``CONFIG["causality_methods"]``.
    """
    graphs = {}

    if "granger" in CONFIG["causality_methods"]:
        from granger_causality import compute_granger_adjacency

        graphs["granger"] = compute_granger_adjacency(
            processed_epochs,
            CONFIG["maxlag"],
            CONFIG["alpha"],
        )

    if "lingam" in CONFIG["causality_methods"]:
        from lingam_causality import compute_lingam_adjacency

        graphs["lingam"] = compute_lingam_adjacency(processed_epochs)

    if "pcmci" in CONFIG["causality_methods"]:
        from pcmci_causality import compute_pcmci_adjacency

        graphs["pcmci"] = compute_pcmci_adjacency(
            processed_epochs,
            tau_max=CONFIG["maxlag"],
            alpha_level=CONFIG["alpha"],
        )

    return graphs

def get_dataset_key(derivatives_dir) -> str:
    """Infer which configured dataset is being processed from its path."""
    name = derivatives_dir.name.lower()
    if "brainlat" in name:
        return "brainlat"
    return "openneuro"

def main() -> None:
    """Preprocess all subjects and save their HGCN graph samples.

    Processing is divided into two passes. The first pass loads and cleans the
    EEG recordings. After the minimum valid epoch count is known, the second
    pass crops every subject consistently and computes features and graphs.
    """
    dataset_key = get_dataset_key(CONFIG["derivatives_dir"])
    excluded_config = CONFIG.get("excluded_subjects", {})
    excluded_subjects = set(excluded_config.get(dataset_key, []))
    subjects = [
        subject_dir
        for subject_dir in sorted(CONFIG["derivatives_dir"].glob("sub-*"))
        if subject_dir.name not in excluded_subjects
    ]
    if excluded_subjects:
        print(f"Excluded {dataset_key} subjects: {sorted(excluded_subjects)}")
    print(f"Starting preprocessing for {len(subjects)} subjects...")

    processed_subjects = []
    failed_subjects = []

    # First pass: preprocess all recordings before selecting a shared duration.
    for subject_dir in subjects:
        subject_id = subject_dir.name
        print(f"\nPreprocessing {subject_id}...")

        try:
            label_name = get_subject_label(subject_id)
            label = CONFIG["labels"][label_name]

            raw = load_subject_raw(subject_dir)
            raw = order_channels(raw, CONFIG["channel_order"])
            raw = filter_and_resample(raw)

            eeg = raw.get_data()
            fs = float(raw.info["sfreq"])
            epochs = split_in_epochs(eeg, fs, CONFIG["epoch_length_sec"])
            processed_epochs, raw_good_epochs, quality_report, stationarity_report = ( preprocess_epochs_for_subject(epochs))

            if processed_epochs is None:
                print(f"  {subject_id}: no valid epochs, skipping")
                continue

            print(
                f"  kept {quality_report['kept_epochs']} / "
                f"{quality_report['total_epochs']} epochs"
            )

            processed_subjects.append({
                "subject_id": subject_id,
                "label_name": label_name,
                "label": label,
                "fs": fs,
                "processed_epochs": processed_epochs,
                "raw_good_epochs": raw_good_epochs,
                "quality_report": quality_report,
                "stationarity_report": stationarity_report,
            })

        except Exception as e:
            failed_subjects.append((subject_id, str(e)))
            print(f"  {subject_id}: preprocessing failed: {e}")

    if not processed_subjects:
        print("\nNo valid subjects to process.")
        return

    # Using the same epoch count gives all subjects a consistent observation
    # duration and prevents duration from becoming an unintended model signal.
    min_epochs = min(
        subject["processed_epochs"].shape[0]
        for subject in processed_subjects
    )
    print(f"\nCropping all valid subjects to {min_epochs} evenly spaced epochs.")
    print("Starting feature and graph computation...")

    subject_graphs = []

    # Second pass: compute model inputs after aligning subject dimensions.
    for subject in processed_subjects:
        subject_id = subject["subject_id"]
        print(f"\nBuilding sample for {subject_id}...")

        try:
            processed_epochs = subject["processed_epochs"][:min_epochs]
            raw_good_epochs = np.asarray(subject["raw_good_epochs"])[:min_epochs]

            print("  computing node features...")
            node_features = build_node_features(
                raw_good_epochs,
                fs=subject["fs"],
            )
            print("  node features done")

            adjacencies = compute_graphs(processed_epochs)
            # Preserve both directed and symmetric graph variants for models
            # with different message-passing assumptions.
            adjacencies = {
                method: prepare_adjacency_for_hgcn(matrix, CONFIG)
                for method, matrix in adjacencies.items()
            }

            graph = build_subject_graph(
                subject_id=subject_id,
                label_name=subject["label_name"],
                X=node_features,
                adjacencies=adjacencies,
                preprocessed_epochs=processed_epochs,
                quality_report=subject["quality_report"],
                stationarity_report=subject["stationarity_report"],
                fs=subject["fs"],
                config=CONFIG,
            )

            subject_graphs.append(graph)
            print("graph ready")

        except Exception as e:
            failed_subjects.append((subject_id, str(e)))
            print(f"  {subject_id}: feature/graph computation failed: {e}")

    if subject_graphs:
        output_dir = save_hgcn_ready_dataset(subject_graphs, CONFIG)
        print(f"\nSaved HGCN-ready dataset to {output_dir}")

    print(
        f"\nPipeline completed. Saved {len(subject_graphs)} / "
        f"{len(processed_subjects)} valid subjects."
    )
    if failed_subjects:
        print("Failed subjects:")
        for subject_id, error in failed_subjects:
            print(f"  {subject_id}: {error}")


if __name__ == "__main__":
    main()
