from pathlib import Path

import mne
import pandas as pd
from config import CONFIG

DATASET_DIR = CONFIG["dataset_dir"]


def resolve_annexed_file(file_path: Path) -> Path:
    """Restituisce il file reale se il percorso punta a un oggetto git-annex."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    try:
        pointer = file_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return file_path

    if not (
        pointer.startswith("../../.git/annex/objects/")
        or pointer.startswith("/annex/objects/")
    ):
        return file_path

    annex_objects_dir = file_path.parents[2] / ".git" / "annex" / "objects"
    target_name = Path(pointer).name

    for real_path in annex_objects_dir.rglob(target_name):
        if real_path.is_file():
            return real_path

    raise FileNotFoundError(
        "Il file EEG e' un puntatore git-annex, ma il dato reale non e' disponibile. "
        f"Scaricalo con git-annex o DataLad: {file_path}"
    )


participants_path = DATASET_DIR / "participants.tsv"
eeg_path = DATASET_DIR / "sub-001/eeg/sub-001_task-eyesclosed_eeg.set"

participants = pd.read_csv(participants_path, sep="\t")
print(participants.head())

raw = mne.io.read_raw_eeglab(
    str(resolve_annexed_file(eeg_path)),
    preload=True,
)
print(raw.info)