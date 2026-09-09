from joblib import Parallel, delayed
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from typing import cast
from config import CONFIG


def _compute_granger_epoch(epoch, max_lag, alpha):
    n_channels = epoch.shape[0]
    adjacency = np.zeros((n_channels, n_channels), dtype=np.float32)

    for target in range(n_channels):
        for source in range(n_channels):
            if source == target:
                continue

            pair_data = np.column_stack((epoch[target], epoch[source]))
            results = grangercausalitytests(
                pair_data,
                maxlag=max_lag
            )

            significant_lags = sum(
                results[lag][0]["ssr_ftest"][1] < alpha
                for lag in range(1, max_lag + 1)
            )

            adjacency[target, source] = significant_lags / max_lag

    return adjacency


def compute_granger_adjacency(processed_epochs, max_lag, alpha):
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