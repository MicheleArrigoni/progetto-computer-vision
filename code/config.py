"""Central configuration for the EEG-to-HGCN dataset pipeline.

Keeping paths, preprocessing parameters, causal-model settings, channel order,
and class labels in one module ensures that every pipeline stage uses the same
assumptions. Paths are represented by :class:`pathlib.Path` objects and are
resolved relative to the repository, not the caller's working directory.
"""

from pathlib import Path


# ``config.py`` is inside ``code``; ``parents[1]`` is therefore the repository
# root. Deriving all paths from it makes scripts independent of the current
# working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Shared settings used during preprocessing, causal-graph construction, and
# final dataset serialization.
CONFIG = {
    # Main BIDS input and HGCN-ready output locations.
    "project_root": PROJECT_ROOT,
    "dataset_dir": PROJECT_ROOT / "dataset",
    "derivatives_dir": PROJECT_ROOT / "dataset" / "openneuro_derivatives",
    "output_dir": PROJECT_ROOT / "data_hgcn_ready"/ "openneuro",

    # Subjects excluded before computing the shared minimum epoch count.
    # Exclusions are dataset-specific because the same numeric subject id can
    # exist in both OpenNeuro and BrainLat.
    "excluded_subjects": {
        "openneuro": ["sub-086"],
        "brainlat": ["sub-078"],
    },

    # Target sampling frequency after resampling. Lower rates reduce memory and
    # computation while retaining the configured frequency range.
    "target_fs": 100,

    # EEG passband in hertz. It removes slow baseline drift and frequencies
    # above the range used to calculate spectral node features.
    "low_freq": 0.5,
    "high_freq": 40.0,

    # Power-line frequency suppressed by the notch filter (50 Hz in Europe).
    "notch_freq": 50.0,

    # Fixed-length epoch settings. An overlap of zero means adjacent windows do
    # not share samples.
    "epoch_length_sec": 5,
    "overlap": 0.0,

    # Shared causal-analysis settings. ``maxlag`` is measured in samples;
    # ``alpha`` is the significance threshold; ``max_diffs`` limits the number
    # of first differences used to obtain stationary time series.
    "maxlag": 4,
    "alpha": 0.05,
    "max_diffs": 2,

    # VARLiNGAM model selection and pruning. Instantaneous B(0) effects are
    # excluded from the aggregated graph by default.
    "varlingam_criterion": "bic",
    "varlingam_prune": True,
    "varlingam_include_instantaneous": False,

    # Graph estimators executed for every subject. For a quick smoke test this
    # can temporarily be reduced to ["granger"].
    "causality_methods": ["granger", "lingam", "pcmci"],
    "adjacency_method": "granger",
    # Canonical node order. Keeping it fixed prevents graph rows and columns
    # from referring to different channels across subjects.
    "channel_order": [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4",
        "P3", "P4", "O1", "O2", "F7", "F8",
        "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
    ],

    # Clinical label encoding: Alzheimer's disease (AD), healthy control (CN),
    # and frontotemporal dementia (FTD).
    "labels": {
        "AD": 1,
        "CN": 0,
        "FTD": 2,
    },
    "add_self_loops": False,
}
