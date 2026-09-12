"""Preprocess EEG recordings and derive channel-level node features.

This module loads EEGLAB recordings, enforces a canonical channel order,
filters and epochs continuous signals, rejects low-quality windows, evaluates
stationarity with complementary ADF and KPSS tests, and extracts spectral and
summary-statistic features for graph nodes.
"""

import os
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.signal import welch
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
import warnings

import mne
from config import CONFIG
import openneuro as on

def get_subject_label(subject_id: str):
    """Map a BIDS subject identifier to its clinical class.

    The mapping is specific to OpenNeuro dataset ``ds004504``: subjects 001-036
    are Alzheimer's disease (AD), 037-065 are healthy controls (CN), and
    066-088 are frontotemporal dementia (FTD).

    Parameters
    ----------
    subject_id:
        Identifier ending in a three-digit subject number, such as ``sub-001``.

    Returns
    -------
    str
        One of ``"AD"``, ``"CN"``, or ``"FTD"``.

    Raises
    ------
    ValueError
        If the numeric suffix falls outside the dataset's known range.
    """

    # uses only the last three digits of the subject ID to determine the label
    n = int(subject_id[-3:])

    # Class intervals are defined by the dataset's participant numbering.

    if 1 <= n <= 36:
        return "AD"
    elif 37 <= n <= 65:
        return "CN"
    elif 66 <= n <= 88:
        return "FTD"
    else:
        raise ValueError(f"Invalid subject ID: {subject_id}")

def get_subjects_list(data_dir: str):
    """Return sorted BIDS subject directories found under ``data_dir``.

    Only directories whose names start with ``sub-`` are included.
    """
 
    # Normalize string-like inputs before directory discovery.
    dataset_dir = Path(data_dir)

    subjects = sorted([
        p for p in dataset_dir.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    ])
    return subjects



# RAW EEG LOADING AND PREPROCESSING

def load_subject_raw(subject_dir):
    """Load a subject's only EEGLAB recording into memory.

    Parameters
    ----------
    subject_dir:
        Path to a BIDS subject directory containing an ``eeg`` subdirectory.

    Returns
    -------
    mne.io.BaseRaw
        Preloaded MNE recording.

    Raises
    ------
    RuntimeError
        If the EEG directory does not contain exactly one ``*_eeg.set`` file.
    """

    # A unique file is required to avoid silently choosing the wrong session.
    eeg_files = list((subject_dir/"eeg").glob("*_eeg.set"))

    if len(eeg_files) != 1:
        raise RuntimeError(f"error: expected one .set file \n, found: {len (eeg_files)}")
    

    # Preloading is required because filtering and resampling modify all samples.
    raw = mne.io.read_raw_eeglab(
        eeg_files[0],
        preload = True,
        verbose = False
    )

    return raw

def order_channels(raw, channel_order):
    """Select required EEG channels in a stable graph-node order.

    The input recording is copied, so the caller's original object is not
    modified. A stable order is essential because channel indices become graph
    node indices in every saved sample.

    Raises
    ------
    RuntimeError
        If one or more required channels are absent.
    """

    missing = [ch for ch in channel_order if ch not in raw.ch_names]

    if missing:
        raise RuntimeError(f"{missing} channels are missing, subject{raw.getName()}")

    # `RawEEGLAB` objects do not expose `pick_channel` (singular).
    # Use `pick_channels` which accepts a list of names and preserves the order.
    raw = raw.copy().pick(picks=channel_order)
    return raw

def filter_and_resample(raw):
    """Band-pass, notch-filter, and resample a copy of an MNE recording.

    Filter limits, line-noise frequency, and target sampling rate are read from
    :data:`config.CONFIG`. The original recording is left unchanged.
    """
    raw = raw.copy()

    raw.filter(
        l_freq = CONFIG["low_freq"],
        h_freq = CONFIG["high_freq"],
        fir_design="firwin",
        verbose = False
    )

    if CONFIG["notch_freq"] is not None:
        raw.notch_filter(
            CONFIG["notch_freq"],
            verbose = False
        )
    
    raw.resample(
        sfreq=CONFIG["target_fs"],
        npad="auto",
        verbose=False
    )


    return raw

def split_in_epochs(eeg, fs , epoch_lenght_sec):
    """Split continuous EEG into fixed-length, non-overlapping epochs.

    Parameters
    ----------
    eeg:
        Continuous signal with shape ``(n_channels, n_samples)``.
    fs:
        Sampling frequency in hertz.
    epoch_lenght_sec:
        Epoch duration in seconds. The misspelled name is retained for API
        compatibility.

    Returns
    -------
    numpy.ndarray
        Tensor with shape ``(n_epochs, n_channels, samples_per_epoch)``.
        Trailing samples that do not fill a complete epoch are discarded.
    """
    samples_per_epoch = int (fs*epoch_lenght_sec)
    n_channels, n_samples = eeg.shape
    n_epochs = n_samples // samples_per_epoch

    epochs = []

    for i in range (n_epochs):
        start = i * samples_per_epoch
        end = (i+1) * samples_per_epoch

        # Select every channel over the current time window.
        epochs.append(eeg[:, start:end])

    # Stacking introduces the leading epoch dimension.
    return np.stack(epochs)

def make_epoch_stationary(epoch, channels, max_diffs):
    """Apply first differences until every channel passes ADF and KPSS.

    A channel is accepted only when ADF rejects a unit root and KPSS does not
    reject level stationarity. Since differencing shortens the time axis by one
    sample, callers later crop all accepted epochs to a common length.

    Returns
    -------
    tuple
        ``(transformed_epoch, report)``. The report records each test result at
        every differencing depth and whether differencing was applied.
    """
    # epoch shape: (n_channels, samples_per_epoch)
    current = epoch.copy()
    report = {
        "n_diffs": 0,
        "diffed": False,
        "channels": {}
    }

    for diff_id in range (max_diffs + 1):
        non_stationary = []
        
        for idx, ch in enumerate(channels):
            x = current [idx]
            adf_ok = passes_adf_test(x)
            kpss_ok = passes_kpss_test(x)

            report["channels"].setdefault(ch, []).append({
                "diff_id": diff_id,
                "adf_ok": adf_ok,
                "kpss_ok": kpss_ok
            })

            if not (adf_ok and kpss_ok):
                non_stationary.append(ch)
        
        if len (non_stationary) == 0:
            return current, report
        
        if diff_id < max_diffs:
            current = np.diff(current, axis = 1)
            report["n_diffs"] += 1
            report["diffed"] = True

    return current, report

def passes_adf_test(x: np.ndarray, alpha: float = 0.05, maxlag: int = 4) -> bool:
    """Return whether a series passes the Augmented Dickey-Fuller test.

    The ADF null hypothesis is the presence of a unit root. A p-value no larger
    than ``alpha`` is therefore interpreted as evidence of stationarity.
    Numerical or statistical failures conservatively return ``False``.
    """
    try:
        # The p-value is the second item in statsmodels' tuple result.
        result = adfuller(x, maxlag=maxlag, result_object=False)
        p_value = result[1]
        
        return p_value <= alpha
    except Exception:
        return False

def passes_kpss_test(x: np.ndarray, alpha: float = 0.05, result_object: bool = False) -> bool:
    """Return whether a series passes the KPSS stationarity test.

    KPSS uses stationarity as its null hypothesis, so a p-value greater than
    ``alpha`` is treated as a pass. Interpolation warnings occur when the true
    p-value lies outside the lookup table and are intentionally suppressed.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            result = kpss(x, regression='c', nlags="auto", result_object=result_object)

        p_value = result[1] if isinstance(result, tuple) else result.pvalue
        return p_value > alpha
    except Exception:
        return False

def is_good_epoch(epoch, amp_treshold = 150e-6):
    """Check an epoch for finite values, extreme amplitudes, and flat channels.

    ``amp_treshold`` is expressed in the same unit as the MNE data (normally
    volts). Its historical spelling is retained to avoid breaking callers.
    """
    if not np.isfinite(epoch).all():
        return False

    if np.max(np.abs(epoch)) > amp_treshold:
        return False

    channel_std = epoch.std(axis = 1) 
    if np.any (channel_std < 1e-8):
        return False
    
    return True

def zscore_epoch(epoch, eps=1e-8):
    """Standardize every channel independently within one epoch.

    Parameters
    ----------
    epoch:
        Array with shape ``(n_channels, samples_per_epoch)``.
    eps:
        Small denominator offset protecting against division by zero.
    """
    mean = epoch.mean(axis=1, keepdims=True)
    std = epoch.std(axis=1, keepdims=True)
    return (epoch - mean) / (std + eps)

def crop_to_same_lenght(epochs):
    """Crop epochs to their shortest time axis and stack them.

    Differencing may leave otherwise valid epochs with different sample counts.
    The historical function-name spelling is retained for API compatibility.
    """
    min_len = min(epoch.shape[1] for epoch in epochs)
    return np.stack([epoch[:, :min_len] for epoch in epochs])

def preprocess_epochs_for_subject(epochs):
    """Quality-control, standardize, and stationarize one subject's epochs.

    Returns
    -------
    tuple
        ``(processed_epochs, raw_good_epochs, quality_report)``. The first item
        is ``None`` if every epoch fails quality control. ``raw_good_epochs``
        contains standardized data before differencing and is used for spectral
        features; ``processed_epochs`` is used for causal estimation.
    """
    good_epochs = []
    raw_good_epochs = []
    stationarity_report = []

    quality_report = {
        "total_epochs": int(len(epochs)),
        "kept_epochs": 0,
        "discarded_epochs": 0
    }

    for epoch_idx, epoch in enumerate(epochs):
        ok = is_good_epoch(epoch)

        if not ok:
            quality_report["discarded_epochs"] += 1
            continue

        epoch = zscore_epoch(epoch)

        raw_good_epochs.append(epoch)
        epoch_stat, report = make_epoch_stationary(
            epoch,
            channels = CONFIG["channel_order"],
            max_diffs = CONFIG["max_diffs"]
        )


        good_epochs.append(epoch_stat)

        quality_report["kept_epochs"] += 1
    # Causal estimation cannot proceed when quality control rejects all epochs.
    if len(good_epochs) == 0:
        return None,[], quality_report

    processed_epochs = crop_to_same_lenght(good_epochs)
    return processed_epochs, raw_good_epochs,  quality_report

def mean_bandpower_epochs(ch_epochs, fs, band, relative=False, nperseg=None):
    """Estimate mean spectral power for one channel across multiple epochs.

    Welch's method estimates the power spectral density of each epoch. Power is
    integrated over the inclusive ``(fmin, fmax)`` interval and then averaged
    across epochs. If ``relative`` is true, each band integral is divided by
    that epoch's total spectral power before averaging.

    Parameters
    ----------
    ch_epochs:
        Single-channel data with shape ``(n_epochs, n_samples)``.
    fs:
        Sampling frequency in hertz.
    band:
        Inclusive lower and upper frequency limits in hertz.
    relative:
        Return a fraction of total power instead of absolute power.
    nperseg:
        Welch segment length; defaults to the smaller of 256 and the epoch
        length.
    """
    fmin, fmax = band
    nperseg = nperseg or min(256, ch_epochs.shape[1])
    bp_list = []
    frac_list = []

    for epoch in ch_epochs:
        f, Pxx = welch(epoch, fs=fs, nperseg=nperseg)
        mask = (f >= fmin) & (f <= fmax)
        bp = np.trapezoid(Pxx[mask], f[mask]) if mask.any() else 0.0
        if relative:
            total = np.trapezoid(Pxx, f) if f.size and Pxx.size else 0.0
            frac_list.append(bp / total if total > 0 else 0.0)
        else:
            bp_list.append(bp)

    return float(np.mean(frac_list if relative else bp_list)) if (relative and frac_list) or (not relative and bp_list) else 0.0

def build_node_features(preprocessed_epochs, fs):
    """Build an eight-dimensional feature vector for every EEG channel.

    Features are absolute delta, theta, alpha, beta, and gamma bandpower,
    followed by the mean, standard deviation, and variance over all accepted
    samples. Spectral power is averaged across epochs.

    Parameters
    ----------
    preprocessed_epochs:
        EEG tensor with shape ``(n_epochs, n_channels, n_samples)``. In the main
        pipeline this is the standardized, pre-differencing representation.
    fs:
        Sampling frequency in hertz.

    Returns
    -------
    numpy.ndarray
        ``float32`` feature matrix with shape ``(n_channels, 8)``.
    """
    _, n_channels, _ = preprocessed_epochs.shape
    features = []

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 24),
        "gamma": (24, 40),
    }

    for ch_idx in range(n_channels):
        ch_epochs = preprocessed_epochs[:, ch_idx, :]

        bandpower = {
            name: mean_bandpower_epochs(ch_epochs, fs, band)
            for name, band in bands.items()
        }

        node_features = [
            bandpower["delta"],
            bandpower["theta"],
            bandpower["alpha"],
            bandpower["beta"],
            bandpower["gamma"],
            float(ch_epochs.mean()),
            float(ch_epochs.std()),
            float(ch_epochs.var()),
        ]

        features.append(node_features)

    return np.array(features, dtype=np.float32)
