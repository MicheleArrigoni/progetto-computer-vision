"""Estimate directed EEG connectivity with the VARLiNGAM algorithm.

VARLiNGAM is fitted separately to every epoch. Absolute coefficients are
averaged over selected lags and then across epochs. Matrix entries follow the
``adjacency[target, source]`` convention.
"""

from typing import cast

import lingam
import numpy as np
from joblib import Parallel, delayed

from config import CONFIG


def _compute_lingam_epoch(epoch: np.ndarray, epoch_idx: int) -> np.ndarray:
    """Fit VARLiNGAM to one EEG epoch and aggregate its lag matrices.

    Parameters
    ----------       
    epoch:
        EEG data with shape ``(n_channels, n_timepoints)``.
    epoch_idx:
        Zero-based epoch index, included in errors for easier diagnosis.

    Returns
    -------
    numpy.ndarray
        Non-negative matrix with shape ``(n_channels, n_channels)`` and
        orientation ``[target, source]``.

    Raises
    ------
    RuntimeError
        If the fitted model does not expose any adjacency matrices.
    """
    n_channels = epoch.shape[0]

    # VARLiNGAM expects observations by variables, while epochs are stored as
    # channels by timepoints.
    data = epoch.T

    model = lingam.VARLiNGAM(
        lags=CONFIG["maxlag"],
        criterion=CONFIG["varlingam_criterion"],
        prune=CONFIG["varlingam_prune"],
    )
    model.fit(data)

    lag_coefficients = model.adjacency_matrices_
    if lag_coefficients is None:
        raise RuntimeError(
            f"VARLiNGAM did not produce adjacency matrices "
            f"for epoch {epoch_idx}."
        )

    # Shape: (n_lags, n_targets, n_sources). Index zero describes
    # instantaneous effects and can be omitted through project configuration.
    lag_coefficients = np.asarray(lag_coefficients, dtype=np.float32)
    lag_matrices = (
        lag_coefficients
        if CONFIG["varlingam_include_instantaneous"]
        else lag_coefficients[1:]
    )

    if len(lag_matrices) == 0:
        adjacency = np.zeros((n_channels, n_channels), dtype=np.float32)
    else:
        # Magnitudes encode edge strength; coefficient signs are not retained.
        adjacency = np.mean(np.abs(lag_matrices), axis=0)

    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def compute_lingam_adjacency(preprocessed_epochs: np.ndarray) -> np.ndarray:
    """Average VARLiNGAM adjacency matrices across one subject's epochs.

    Parameters
    ----------
    preprocessed_epochs:
        EEG tensor with shape ``(n_epochs, n_channels, n_timepoints)``.

    Returns
    -------
    numpy.ndarray
        Epoch-averaged ``float32`` matrix in ``[target, source]`` orientation.
        If no epochs are supplied, an all-zero matrix is returned.
    """
    n_epochs, n_channels, _ = preprocessed_epochs.shape

    if n_epochs == 0:
        return np.zeros((n_channels, n_channels), dtype=np.float32)

    matrices = cast(
        list[np.ndarray],
        Parallel(
            n_jobs=CONFIG.get("lingam_n_jobs", -1),
            verbose=10,
        )(
            delayed(_compute_lingam_epoch)(epoch, epoch_idx)
            for epoch_idx, epoch in enumerate(preprocessed_epochs)
        ),
    )

    return np.mean(matrices, axis=0).astype(np.float32)
