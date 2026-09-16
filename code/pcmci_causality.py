"""Estimate epoch-averaged causal EEG graphs with PCMCI.

PCMCI is run independently on every preprocessed epoch. Significant lagged
dependencies are converted to weighted directed adjacency matrices and then
averaged across all epochs that completed successfully.

All adjacency matrices in this module use ``adjacency[target, source]``. Thus,
entry ``[i, j]`` represents the directed edge ``channel j -> channel i``.
"""

import warnings

import numpy as np
from joblib import Parallel, delayed

warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r"pygam(\.|$)",
)

from tigramite import data_processing as dp
from tigramite.independence_tests.parcorr import ParCorr
import tigramite.pcmci as pcmci

from config import CONFIG


def _compute_pcmci_epoch(
    epoch: np.ndarray,
    tau_max: int,
    alpha_level: float,
) -> np.ndarray | None:
    """Compute a PCMCI adjacency matrix for one EEG epoch.

    Parameters
    ----------
    epoch:
        EEG data with shape ``(n_channels, n_timepoints)``.
    tau_max:
        Largest positive time lag tested by PCMCI, in samples. Lag zero is
        excluded from the returned graph.
    alpha_level:
        Significance threshold used for PCMCI parent selection and final edge
        filtering.

    Returns
    -------
    numpy.ndarray or None
        A ``float32`` matrix with shape ``(n_channels, n_channels)`` and
        orientation ``[target, source]``. An edge weight is the mean absolute
        partial-correlation value across significant lags. Returns ``None`` if
        the epoch cannot be processed.

    Notes
    -----
    Absolute effect sizes deliberately discard the sign of a dependency, since
    downstream graph construction expects non-negative connection strengths.
    """
    n_channels = epoch.shape[0]

    try:
        # Tigramite expects observations by variables; the pipeline stores EEG
        # epochs as channels by timepoints, so transpose before analysis.
        data = epoch.T
        dataframe = dp.DataFrame(data)
        parcorr = ParCorr(significance="analytic")
        pcmci_instance = pcmci.PCMCI(
            dataframe=dataframe,
            cond_ind_test=parcorr,
        )

        results = pcmci_instance.run_pcmci(
            tau_max=tau_max,
            pc_alpha=alpha_level,
        )

        p_matrix = results["p_matrix"]
        val_matrix = results["val_matrix"]
        # Entries remain zero when no tested lag is significant.
        adjacency = np.zeros(
            (n_channels, n_channels),
            dtype=np.float32,
        )

        for source in range(n_channels):
            for target in range(n_channels):
                # Self-connections are not included in the causal graph.
                if source == target:
                    continue

                # The last result axis indexes lags. Starting from one removes
                # contemporaneous dependencies from the graph.
                strengths = [
                    abs(val_matrix[source, target, tau])
                    for tau in range(1, tau_max + 1)
                    if p_matrix[source, target, tau] < alpha_level
                ]

                if strengths:
                    adjacency[target, source] = np.mean(strengths)

        return adjacency

    except Exception:
        # A failed worker must not abort processing of the remaining epochs.
        print("Error computing PCMCI adjacency for an epoch.")
        return None


def compute_pcmci_adjacency(
    preprocessed_epochs: np.ndarray,
    tau_max: int = 1,
    alpha_level: float = 0.05,
) -> np.ndarray:
    """Aggregate PCMCI adjacency matrices over a subject's EEG epochs.

    Parameters
    ----------
    preprocessed_epochs:
        Array with shape ``(n_epochs, n_channels, n_timepoints)``.
    tau_max:
        Largest positive time lag tested by PCMCI, in samples.
    alpha_level:
        Significance threshold for parent selection and edge filtering.

    Returns
    -------
    numpy.ndarray
        Epoch-averaged ``float32`` adjacency matrix using
        ``[target, source]`` orientation. Returns an all-zero matrix if every
        epoch fails.
    """
    _, n_channels, _ = preprocessed_epochs.shape

    # Joblib uses every available CPU by default. The setting can be overridden
    # in CONFIG when memory or CPU availability is limited.
    matrices = Parallel(
        n_jobs=CONFIG.get("pcmci_n_jobs", -1),
        verbose=10,
    )(
        delayed(_compute_pcmci_epoch)(epoch, tau_max, alpha_level)
        for epoch in preprocessed_epochs
    )

    # Worker failures are represented by None and excluded from the average.
    matrices = [matrix for matrix in matrices if matrix is not None]

    if not matrices:
        print("No valid PCMCI adjacency matrices were computed.")
        return np.zeros(
            (n_channels, n_channels),
            dtype=np.float32,
        )

    # The cast keeps the aggregate consistent with each per-epoch matrix.
    return np.mean(matrices, axis=0).astype(np.float32)
