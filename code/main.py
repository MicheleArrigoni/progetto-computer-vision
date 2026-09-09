import numpy as np

from config import CONFIG

from preprocessing import (
    preprocess_epochs_for_subject,
    load_subject_raw,
    order_channels,
    filter_and_resample,
    split_in_epochs,
    get_subject_label,
    build_node_features,
)

from dataset_module import(
    prepare_adjacency_for_hgcn,
    build_subject_graph,
    save_hgcn_ready_dataset,
)

def compute_graphs(processed_epochs):
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

def main():
    subjects = sorted(CONFIG["derivatives_dir"].glob("sub-*"))
    print(f"Starting preprocessing for {len(subjects)} subjects...")

    processed_subjects = []
    failed_subjects = []

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
            raw_good_epochs = []

            epochs = split_in_epochs(eeg, fs, CONFIG["epoch_length_sec"])
            processed_epochs, raw_good_epochs, quality_report = preprocess_epochs_for_subject(epochs)

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
            })

        except Exception as e:
            failed_subjects.append((subject_id, str(e)))
            print(f"  {subject_id}: preprocessing failed: {e}")

    if not processed_subjects:
        print("\nNo valid subjects to process.")
        return

    min_epochs = min(
        subject["processed_epochs"].shape[0]
        for subject in processed_subjects
    )
    print(f"\nCropping all valid subjects to {min_epochs} evenly spaced epochs.")
    print("Starting feature and graph computation...")

    subject_graphs = []

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
            adjacencies = {
                method: prepare_adjacency_for_hgcn(A, CONFIG)
                for method, A in adjacencies.items()
            }

            graph = build_subject_graph(
                subject_id=subject_id,
                label_name=subject["label_name"],
                X=node_features,
                adjacencies=adjacencies,
                preprocessed_epochs=processed_epochs,
                quality_report=subject["quality_report"],
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

    print(f"\nPipeline completed. Saved {len(subject_graphs)} / {len(processed_subjects)} valid subjects.")
    if failed_subjects:
        print("Failed subjects:")
        for subject_id, error in failed_subjects:
            print(f"  {subject_id}: {error}")


if __name__ == "__main__":
    main()
