"""Construct directed EEG graphs using pairwise Granger-causality tests.

Each edge weight is derived from the strongest lag-level p-value that survives
false-discovery-rate correction within an epoch. Matrices follow the
``adjacency[target, source]`` convention throughout this module.
"""

from typing import cast

from joblib import Parallel, delayed
import numpy as np
from statsmodels.stats.multitest import multipletests
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
        ``[target, source]`` contains ``-log10(p)`` for the best lag of the
        hypothesis ``source -> target`` when it survives FDR correction.
    """
    n_channels = epoch.shape[0]
    adjacency = np.zeros((n_channels, n_channels), dtype=np.float32)
    edge_candidates: list[tuple[int, int, float]] = []

    for target in range(n_channels):
        for source in range(n_channels):
            # A channel trivially predicts its own past, so omit self-edges.
            if source == target:
                continue

            if (
                not np.isfinite(epoch[target]).all()
                or not np.isfinite(epoch[source]).all()
                or np.std(epoch[target]) == 0.0
                or np.std(epoch[source]) == 0.0
            ):
                continue

            # statsmodels interprets the second column as the potential cause
            # of the first, hence the [target, source] column order.
            pair_data = np.column_stack((epoch[target], epoch[source]))
            try:
                results = grangercausalitytests(pair_data, maxlag=max_lag)
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise RuntimeError(
                    f"Test Granger fallito: source={source}, target={target}, "
                    f"max_lag={max_lag}"
                ) from exc

            # Extract the p-values for the F test of each lag and keep only finite values.
            p_values = np.array(
                [
                    results[lag][0]["ssr_ftest"][1]
                    for lag in range(1, max_lag + 1)
                ],
                dtype=np.float64,
            )

            # Remove any non-finite p-values before proceeding. If all p-values are non-finite, skip this pair.
            p_values = p_values[np.isfinite(p_values)]

            # If there are no valid p-values left after filtering, skip this pair.
            if p_values.size == 0:
                continue

            # Find the minimum p-value and add it to the list of edge candidates.
            edge_candidates.append((target, source, float(p_values.min())))

    # If no edge candidates were found, return the empty adjacency matrix.
    if not edge_candidates:
        print("No valid Granger-causality edges found for this epoch.")
        return adjacency

    # Apply FDR correction to the p-values of the edge candidates.
    corrected_significant, _, _, _ = multipletests(
        [candidate[2] for candidate in edge_candidates],
        alpha=alpha,
        method="fdr_bh",
    )

    for is_significant, (target, source, p_value) in zip(
        corrected_significant,
        edge_candidates,
    ):
        if is_significant:
            adjacency[target, source] = -np.log10(max(p_value, 1e-12))

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
        Values are averaged ``-log10(p)`` edge strengths across epochs after
        epoch-level FDR correction.
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
