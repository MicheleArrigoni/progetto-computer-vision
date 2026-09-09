import lingam
import numpy as np
from joblib import Parallel, delayed
from typing import cast

from config import CONFIG


def _compute_lingam_epoch(epoch, epoch_idx):
    """Calcola la matrice di adiacenza per una singola epoca."""
    n_channels = epoch.shape[0]

    # VARLiNGAM vuole shape [osservazioni, variabili].
    data = epoch.T

    model = lingam.VARLiNGAM(
        lags=CONFIG["maxlag"],
        criterion=CONFIG["varlingam_criterion"],
        prune=CONFIG["varlingam_prune"],
    )
    model.fit(data)

    B = model.adjacency_matrices_
    if B is None:
        raise RuntimeError(
            f"VARLiNGAM did not produce adjacency matrices "
            f"for epoch {epoch_idx}."
        )

    B = np.asarray(B, dtype=np.float32)

    # B[tau, target, source].
    lag_matrices = (
        B
        if CONFIG["varlingam_include_instantaneous"]
        else B[1:]
    )

    if len(lag_matrices) == 0:
        adjacency = np.zeros(
            (n_channels, n_channels),
            dtype=np.float32,
        )
    else:
        adjacency = np.mean(np.abs(lag_matrices), axis=0)

    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def compute_lingam_adjacency(preprocessed_epochs):
    """
    Calcola e aggrega le matrici VARLiNGAM in parallelo.
    """
    n_epochs, n_channels, _ = preprocessed_epochs.shape

    if n_epochs == 0:
        return np.zeros(
            (n_channels, n_channels),
            dtype=np.float32,
        )

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