"""Prepare BrainLat EEG files for the unchanged project pipeline.

The main pipeline expects this layout:

    target/sub-001/eeg/sub-001_task-eyesclosed_eeg.set

and it infers labels from the numeric subject id:

    sub-001..sub-036 -> AD
    sub-037..sub-065 -> CN
    sub-066..sub-088 -> FTD

BrainLat stores subjects below diagnosis/site folders and uses a 128-channel
montage with labels such as A1, B22, C29. This adapter converts BrainLat into a
new target folder that follows the pipeline contract, selecting the BrainLat
electrodes nearest to the configured 19-channel 10-20 montage.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import shutil

import mne
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHANNEL_ORDER = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4",
    "P3", "P4", "O1", "O2", "F7", "F8",
    "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
]

PIPELINE_ID_SLOTS = {
    "AD": range(1, 37),
    "CN": range(37, 66),
    "FTD": range(66, 89),
}


@dataclass(frozen=True)
class BrainLatSubject:
    """One BrainLat EEG recording."""

    original_id: str
    label_name: str
    eeg_path: Path
    source_path: str


@dataclass(frozen=True)
class PreparedSubject:
    """A BrainLat subject assigned to a pipeline-compatible subject id."""

    original: BrainLatSubject
    pipeline_id: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BrainLat EEG into the layout expected by main.py."
    )
    parser.add_argument(
        "--brainlat-root",
        type=Path,
        default=PROJECT_ROOT / "brainlat_dataset" / "EEG data",
        help="Path to the BrainLat 'EEG data' folder.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "brainlat_derivatives",
        help="Output folder to use later as CONFIG['derivatives_dir'].",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print counts and planned subject assignment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted .set files.",
    )
    args = parser.parse_args()

    subjects = discover_subjects(args.brainlat_root)
    prepared, skipped = assign_pipeline_ids(subjects)

    print_counts(subjects, prepared, skipped)

    if args.dry_run:
        return

    prepare_target_dataset(prepared, args.target, overwrite=args.overwrite)
    print(f"\nPrepared BrainLat target folder: {args.target.resolve()}")
    print("Set CONFIG['derivatives_dir'] to this folder, then run code/main.py.")


def discover_subjects(eeg_root: Path) -> list[BrainLatSubject]:
    """Discover BrainLat .set files and infer labels from records or folders."""
    eeg_root = eeg_root.resolve()
    labels_by_subject = labels_from_records(eeg_root)
    subjects = []

    for eeg_path in sorted(eeg_root.rglob("*.set")):
        subject_dir = subject_dir_from_eeg_path(eeg_path)
        if not is_subject_dir_name(subject_dir.name):
            continue

        label_name = labels_by_subject.get(subject_dir.name)
        if label_name is None:
            label_name = label_from_path(eeg_root, eeg_path)

        subjects.append(
            BrainLatSubject(
                original_id=subject_dir.name,
                label_name=label_name,
                eeg_path=eeg_path,
                source_path=source_path(eeg_root, eeg_path),
            )
        )

    return subjects


def labels_from_records(eeg_root: Path) -> dict[str, str]:
    """Read optional BrainLat records CSV files as a label source."""
    labels = {}

    for records_path in sorted(eeg_root.rglob("records*_eeg_data.csv")):
        try:
            with open(records_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    subject_id = row_value(row, "id_EEG", "id EEG", "id")
                    diagnosis = normalize_label(row_value(row, "diagnosis"))
                    has_eeg = row_value(row, "eeg")

                    if not subject_id or not diagnosis:
                        continue
                    if has_eeg and has_eeg.strip() not in {"1", "true", "True"}:
                        continue

                    labels[subject_id] = diagnosis
        except UnicodeDecodeError:
            continue

    return labels


def assign_pipeline_ids(
    subjects: list[BrainLatSubject],
) -> tuple[list[PreparedSubject], list[BrainLatSubject]]:
    """Assign subjects to ids compatible with preprocessing.get_subject_label."""
    prepared = []
    skipped = []

    by_label: dict[str, list[BrainLatSubject]] = {}
    for subject in subjects:
        by_label.setdefault(subject.label_name, []).append(subject)

    for label, slots in PIPELINE_ID_SLOTS.items():
        slots = list(slots)
        label_subjects = sorted(
            by_label.get(label, []),
            key=lambda subject: subject.original_id,
        )

        for index, subject in enumerate(label_subjects):
            slot = slots[index % len(slots)]
            pipeline_id = (
                f"sub-{slot:03d}"
                if index < len(slots)
                else f"sub-brainlat-{label.lower()}-{index + 1:03d}-{slot:03d}"
            )
            prepared.append(
                PreparedSubject(
                    original=subject,
                    pipeline_id=pipeline_id,
                )
            )

    for label, label_subjects in by_label.items():
        if label not in PIPELINE_ID_SLOTS:
            skipped.extend(label_subjects)

    return sorted(prepared, key=lambda item: item.pipeline_id), skipped


def prepare_target_dataset(
    prepared_subjects: list[PreparedSubject],
    target_root: Path,
    overwrite: bool,
) -> None:
    """Export selected and renamed channels into a pipeline-ready target."""
    ensure_eeglabio_available()
    target_root.mkdir(parents=True, exist_ok=True)
    failures = []

    for index, prepared in enumerate(prepared_subjects, start=1):
        out_dir = target_root / prepared.pipeline_id / "eeg"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_set = out_dir / f"{prepared.pipeline_id}_task-eyesclosed_eeg.set"
        if out_set.exists() and not overwrite:
            print(f"[{index:03d}] exists, skipping {prepared.pipeline_id}")
            continue

        print(
            f"[{index:03d}] converting {prepared.original.original_id} "
            f"({prepared.original.label_name}) -> {prepared.pipeline_id}"
        )

        try:
            raw = load_and_adapt_raw(prepared.original.eeg_path)
            mne.export.export_raw(
                out_set,
                raw,
                fmt="eeglab",
                overwrite=True,
                verbose=False,
            )
            write_sidecars(out_dir, prepared)
        except Exception as exc:
            cleanup_failed_subject_dir(out_dir)
            failures.append({
                "pipeline_id": prepared.pipeline_id,
                "original_id": prepared.original.original_id,
                "label": prepared.original.label_name,
                "eeg_path": str(prepared.original.eeg_path),
                "error": str(exc),
            })
            print(f"      failed: {exc}")

    if failures:
        write_failures_csv(target_root, failures)
        print(f"\nConversion failures: {len(failures)}")
        print(f"Details written to: {(target_root / 'conversion_failures.csv').resolve()}")
    else:
        failures_path = target_root / "conversion_failures.csv"
        if failures_path.exists():
            failures_path.unlink()


def ensure_eeglabio_available() -> None:
    """Fail early with an actionable message when EEGLAB export is unavailable."""
    try:
        import eeglabio  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing optional dependency 'eeglabio'. Install it with:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install eeglabio\n"
            "or, if you use uv:\n"
            "  uv sync"
        ) from exc


def cleanup_failed_subject_dir(out_dir: Path) -> None:
    """Remove an incomplete subject export after a failed conversion."""
    if out_dir.exists():
        shutil.rmtree(out_dir.parent)


def write_failures_csv(target_root: Path, failures: list[dict[str, str]]) -> None:
    failures_path = target_root / "conversion_failures.csv"
    with open(failures_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pipeline_id", "original_id", "label", "eeg_path", "error"],
        )
        writer.writeheader()
        writer.writerows(failures)


def load_and_adapt_raw(eeg_path: Path):
    """Load BrainLat, select nearest 10-20 electrodes, and rename channels."""
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose=False)
    mapping = map_brainlat_channels_to_1020(raw)
    source_order = [mapping[target] for target in CHANNEL_ORDER]

    raw = raw.copy().pick(picks=source_order)
    raw.rename_channels({source: target for target, source in mapping.items()})
    raw = raw.pick(picks=CHANNEL_ORDER)
    return raw


def map_brainlat_channels_to_1020(raw) -> dict[str, str]:
    """Map every target channel to the nearest BrainLat electrode coordinate."""
    standard = mne.channels.make_standard_montage("standard_1020")
    standard_positions = standard.get_positions()["ch_pos"]
    used_sources = set()
    mapping = {}

    for target in CHANNEL_ORDER:
        target_xyz = unit_vector(np.asarray(standard_positions[target]))
        best_source = None
        best_distance = float("inf")

        for channel in raw.info["chs"]:
            source = channel["ch_name"]
            if source in used_sources:
                continue

            source_xyz = unit_vector(channel["loc"][:3])
            if source_xyz is None:
                continue

            distance = float(np.linalg.norm(source_xyz - target_xyz))
            if distance < best_distance:
                best_distance = distance
                best_source = source

        if best_source is None:
            raise RuntimeError(f"No BrainLat electrode found for {target}")

        mapping[target] = best_source
        used_sources.add(best_source)

    return mapping


def write_sidecars(out_dir: Path, prepared: PreparedSubject) -> None:
    """Write lightweight metadata files next to the converted .set file."""
    channels_path = out_dir / f"{prepared.pipeline_id}_task-eyesclosed_channels.tsv"
    json_path = out_dir / f"{prepared.pipeline_id}_task-eyesclosed_eeg.json"

    with open(channels_path, "w", encoding="utf-8", newline="") as f:
        f.write("name\ttype\n")
        for channel in CHANNEL_ORDER:
            f.write(f"{channel}\tEEG\n")

    with open(json_path, "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f'  "OriginalBrainLatID": "{prepared.original.original_id}",\n')
        f.write(f'  "Diagnosis": "{prepared.original.label_name}",\n')
        f.write(f'  "SourcePath": "{prepared.original.source_path}"\n')
        f.write("}\n")


def print_counts(
    subjects: list[BrainLatSubject],
    prepared: list[PreparedSubject],
    skipped: list[BrainLatSubject],
) -> None:
    all_counts = counts_by_label(subjects)
    prepared_counts = counts_by_label([item.original for item in prepared])
    skipped_counts = counts_by_label(skipped)

    print(f"BrainLat EEG subjects found: {len(subjects)} {all_counts}")
    print(f"Pipeline-compatible subjects prepared: {len(prepared)} {prepared_counts}")
    print(f"Skipped subjects: {len(skipped)} {skipped_counts}")


def counts_by_label(subjects: list[BrainLatSubject]) -> dict[str, int]:
    counts = {}
    for subject in subjects:
        counts[subject.label_name] = counts.get(subject.label_name, 0) + 1
    return dict(sorted(counts.items()))


def row_value(row: dict[str, str], *names: str) -> str:
    normalized = {
        key.lower().replace("_", "").replace(" ", ""): value
        for key, value in row.items()
        if key is not None and value is not None
    }

    for name in names:
        value = normalized.get(name.lower().replace("_", "").replace(" ", ""))
        if value is not None:
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
    parts = eeg_path.relative_to(eeg_root).parts
    if not parts:
        return ""
    folder = parts[0]
    return normalize_label(folder.split("_", 1)[1] if "_" in folder else folder)


def source_path(eeg_root: Path, eeg_path: Path) -> str:
    parts = eeg_path.relative_to(eeg_root).parts
    return "/".join(parts[:2]) if len(parts) >= 2 else ""


def subject_dir_from_eeg_path(eeg_path: Path) -> Path:
    if eeg_path.parent.name == "eeg":
        return eeg_path.parent.parent
    return eeg_path.parent


def is_subject_dir_name(name: str) -> bool:
    return name.startswith(("sub-", "sub_", "suj_"))


def unit_vector(vector) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        return None
    return vector / norm


if __name__ == "__main__":
    main()
