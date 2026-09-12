"""Construct directed EEG graphs using pairwise Granger-causality tests.

Each edge weight is the fraction of tested lags for which the source channel
significantly improves prediction of the target channel. Matrices follow the
``adjacency[target, source]`` convention throughout this module.
"""

from typing import cast

from joblib import Parallel, delayed
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

from config import CONFIG


def _compute_granger_epoch(
    epoch: np.ndarray,
    max_lag: int,
    alpha: float,
) -> np.ndarray:
    """Compute a pairwise Granger adjacency matrix for one EEG epoch.

    Parameters
    ----------
    epoch:
        Stationary EEG data with shape ``(n_channels, n_timepoints)``.
    max_lag:
        Maximum autoregressive lag to test, expressed in samples.
    alpha:
        P-value threshold used to classify a lag as significant.

    Returns
    -------
    numpy.ndarray
        A ``float32`` matrix of shape ``(n_channels, n_channels)``. Entry
        ``[target, source]`` lies in ``[0, 1]`` and equals the proportion of
        significant lags for the hypothesis ``source -> target``.
    """
    n_channels = epoch.shape[0]
    adjacency = np.zeros((n_channels, n_channels), dtype=np.float32)

    for target in range(n_channels):
        for source in range(n_channels):
            # A channel trivially predicts its own past, so omit self-edges.
            if source == target:
                continue

            # statsmodels interprets the second column as the potential cause
            # of the first, hence the [target, source] column order.
            pair_data = np.column_stack((epoch[target], epoch[source]))
            results = grangercausalitytests(pair_data, maxlag=max_lag)

            # Normalize the count so results remain on the same scale when a
            # different maximum lag is configured.
            significant_lags = sum(
                results[lag][0]["ssr_ftest"][1] < alpha
                for lag in range(1, max_lag + 1)
            )
            adjacency[target, source] = significant_lags / max_lag

    return adjacency


def compute_granger_adjacency(
    processed_epochs: np.ndarray,
    max_lag: int,
    alpha: float,
) -> np.ndarray:
    """Average Granger-causality graphs across one subject's epochs.

    Parameters
    ----------
    processed_epochs:
        Stationary EEG tensor with shape
        ``(n_epochs, n_channels, n_timepoints)``.
    max_lag:
        Maximum tested lag in samples.
    alpha:
        Significance threshold applied to every lag-specific F test.

    Returns
    -------
    numpy.ndarray
        Mean ``float32`` adjacency matrix in ``[target, source]`` orientation.
        If the input contains no epochs, an all-zero matrix is returned.
    """
    n_epochs, n_channels, _ = processed_epochs.shape

    if n_epochs == 0:
        return np.zeros((n_channels, n_channels), dtype=np.float32)

    matrices = cast(
        list[np.ndarray],
        Parallel(
            n_jobs=CONFIG.get("granger_n_jobs", -1),
            verbose=10,
        )(
            delayed(_compute_granger_epoch)(epoch, max_lag, alpha)
            for epoch in processed_epochs
        ),
    )

    return np.mean(matrices, axis=0).astype(np.float32)
