from tigramite import data_processing as dp
from tigramite import plotting as tp
from tigramite.independence_tests.parcorr import ParCorr
import tigramite.pcmci as pcmci
import numpy as np
from joblib import Parallel, delayed
from config import CONFIG


def _compute_pcmci_epoch(epoch, tau_max, alpha_level):
    n_channels = epoch.shape[0]

    try:
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
        adjacency = np.zeros(
            (n_channels, n_channels),
            dtype=np.float32,
        )

        for source in range(n_channels):
            for target in range(n_channels):
                if source == target:
                    continue

                strengths = [
                    abs(val_matrix[source, target, tau])
                    for tau in range(1, tau_max + 1)
                    if p_matrix[source, target, tau] < alpha_level
                ]

                if strengths:
                    adjacency[target, source] = np.mean(strengths)

        return adjacency

    except Exception:
        print("Error computing PCMCI adjacency for an epoch.")
        return None


def compute_pcmci_adjacency(
preprocessed_epochs,
tau_max=1,
alpha_level=0.05,
):
    n_epochs, n_channels, _ = preprocessed_epochs.shape

    matrices = Parallel(
        n_jobs=CONFIG.get("pcmci_n_jobs", -1),
        verbose=10,
    )(
        delayed(_compute_pcmci_epoch)(
            epoch,
            tau_max,
            alpha_level,
        )
        for epoch in preprocessed_epochs
    )

    matrices = [matrix for matrix in matrices if matrix is not None]

    if not matrices:
        print("No valid PCMCI adjacency matrices were computed.")
        return np.zeros(
            (n_channels, n_channels),
            dtype=np.float32,
        )

    return np.mean(matrices, axis=0).astype(np.float32)