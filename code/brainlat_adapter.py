"""Convert BrainLat EEG files into the layout expected by the pipeline.

The pipeline reads one EEGLAB file from each folder:

    dataset/brainlat_derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set

Labels are encoded in the final three digits of the subject id, because
preprocessing.py reads labels from those digits. If BrainLat has more subjects
than an original range, the adapter keeps all of them by creating longer unique
ids that still end with a valid class number.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil

import mne
import numpy as np

from config import CONFIG


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHANNEL_ORDER = CONFIG["channel_order"]
ID_SLOTS = {
    "AD": range(1, 37),
    "CN": range(37, 66),
    "FTD": range(66, 89),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BrainLat EEG for the pipeline.")
    parser.add_argument(
        "--brainlat-root",
        type=Path,
        default=PROJECT_ROOT / "brainlat_dataset" / "EEG data",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "brainlat_derivatives",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    subjects = assign_pipeline_ids(discover_subjects(args.brainlat_root))
    print_counts(subjects)

    if args.dry_run:
        return

    convert_subjects(subjects, args.target, overwrite=args.overwrite)
    print(f"\nReady: {args.target.resolve()}")


def discover_subjects(eeg_root: Path) -> list[dict]:
    """Find BrainLat .set files and attach AD/CN/FTD labels."""
    eeg_root = eeg_root.resolve()
    labels = labels_from_records(eeg_root)
    subjects = []

    for eeg_path in sorted(eeg_root.rglob("*.set")):
        subject_dir = subject_dir_from_eeg_path(eeg_path)
        if not subject_dir.name.startswith(("sub-", "sub_", "suj_")):
            continue

        label = labels.get(subject_dir.name) or label_from_path(eeg_root, eeg_path)
        if label not in ID_SLOTS:
            continue

        subjects.append({
            "original_id": subject_dir.name,
            "label": label,
            "eeg_path": eeg_path,
        })

    return subjects


def labels_from_records(eeg_root: Path) -> dict[str, str]:
    """Read BrainLat record CSVs when available."""
    labels = {}
    for records_path in eeg_root.rglob("records*_eeg_data.csv"):
        with open(records_path, newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                subject_id = row_value(row, "id_EEG", "id EEG", "id")
                label = normalize_label(row_value(row, "diagnosis"))
                if subject_id and label:
                    labels[subject_id] = label
    return labels


def assign_pipeline_ids(subjects: list[dict]) -> list[dict]:
    """Give every subject a unique id whose final digits encode the label."""
    prepared = []

    for label, slots in ID_SLOTS.items():
        label_subjects = sorted(
            (subject for subject in subjects if subject["label"] == label),
            key=lambda subject: subject["original_id"],
        )
        slots = list(slots)

        for index, subject in enumerate(label_subjects):
            slot = slots[index % len(slots)]
            subject = subject.copy()
            subject["subject_id"] = (
                f"sub-{slot:03d}"
                if index < len(slots)
                else f"sub-brainlat-{label.lower()}-{index + 1:03d}-{slot:03d}"
            )
            prepared.append(subject)

    return sorted(prepared, key=lambda subject: subject["subject_id"])


def convert_subjects(subjects: list[dict], target_root: Path, overwrite: bool) -> None:
    """Export converted EEGLAB files only."""
    target_root.mkdir(parents=True, exist_ok=True)
    failures = 0

    for index, subject in enumerate(subjects, start=1):
        subject_id = subject["subject_id"]
        out_dir = target_root / subject_id / "eeg"
        out_set = out_dir / f"{subject_id}_task-eyesclosed_eeg.set"

        if out_set.exists() and not overwrite:
            print(f"[{index:03d}] exists, skipping {subject_id}")
            continue

        print(f"[{index:03d}] {subject['original_id']} ({subject['label']}) -> {subject_id}")
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            raw = load_and_adapt_raw(subject["eeg_path"])
            mne.export.export_raw(out_set, raw, fmt="eeglab", overwrite=True, verbose=False)
        except Exception as exc:
            failures += 1
            if out_dir.parent.exists():
                shutil.rmtree(out_dir.parent)
            print(f"      failed: {exc}")

    if failures:
        print(f"\nConversion failures: {failures}")


def load_and_adapt_raw(eeg_path: Path):
    """Select the nearest BrainLat electrodes and rename them to 10-20 labels."""
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=False)
    mapping = map_brainlat_channels_to_1020(raw)

    raw = raw.copy().pick(picks=[mapping[channel] for channel in CHANNEL_ORDER])
    raw.rename_channels({source: target for target, source in mapping.items()})
    return raw.pick(picks=CHANNEL_ORDER)


def map_brainlat_channels_to_1020(raw) -> dict[str, str]:
    standard = mne.channels.make_standard_montage("standard_1020")
    positions = standard.get_positions()["ch_pos"]
    used = set()
    mapping = {}

    for target in CHANNEL_ORDER:
        target_xyz = unit_vector(positions[target])
        candidates = []

        for channel in raw.info["chs"]:
            source = channel["ch_name"]
            source_xyz = unit_vector(channel["loc"][:3])
            if source not in used and source_xyz is not None:
                candidates.append((np.linalg.norm(source_xyz - target_xyz), source))

        if not candidates:
            raise RuntimeError(f"No BrainLat electrode found for {target}")

        _, source = min(candidates)
        mapping[target] = source
        used.add(source)

    return mapping


def print_counts(subjects: list[dict]) -> None:
    counts = {}
    for subject in subjects:
        counts[subject["label"]] = counts.get(subject["label"], 0) + 1
    print(f"BrainLat subjects to convert: {len(subjects)} {dict(sorted(counts.items()))}")


def row_value(row: dict[str, str], *names: str) -> str:
    normalized = {
        key.lower().replace("_", "").replace(" ", ""): value
        for key, value in row.items()
        if key and value
    }
    for name in names:
        value = normalized.get(name.lower().replace("_", "").replace(" ", ""))
        if value:
            return value.strip()
    return ""


def normalize_label(label: str) -> str:
    label = label.strip()
    if label in {"HC", "Control", "CONTROL"}:
        return "CN"
    if label in {"bvFTD", "bVFTD"}:
        return "FTD"
    return label


def label_from_path(eeg_root: Path, eeg_path: Path) -> str:
    folder = eeg_path.relative_to(eeg_root).parts[0]
    return normalize_label(folder.split("_", 1)[1] if "_" in folder else folder)


def subject_dir_from_eeg_path(eeg_path: Path) -> Path:
    return eeg_path.parent.parent if eeg_path.parent.name == "eeg" else eeg_path.parent


def unit_vector(vector) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        return None
    return vector / norm


if __name__ == "__main__":
    main()
