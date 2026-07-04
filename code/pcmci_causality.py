from tigramite import data_processing as dp
from tigramite import plotting as tp
from tigramite.independence_tests.parcorr import ParCorr
import tigramite.pcmci as pcmci
import numpy as np

def compute_pcmci_adjacency(preprocessed_epochs, tau_max=1, alpha_level=0.05):
    """
    Computes the adjacency matrix using PCMCI algorithm.

    Parameters:
    - preprocessed_epochs: numpy array containing the preprocessed time series data.
    - tau_max: maximum time lag to consider for the analysis.
    - alpha_level: significance level for the statistical tests.

    Returns:
    - adjacency_matrix: numpy array representing the adjacency matrix.
    """

    n_epochs, n_channels, _ = preprocessed_epochs.shape
    matrices = []

    for epoch_idx, epoch in enumerate(preprocessed_epochs):
        if epoch_idx % 100 == 0:
            print(f"    PCMCI epoch {epoch_idx + 1}/{n_epochs}")

        # Tigramite vuole shape [tempo, variabili]
        data = epoch.T  # Transpose to shape [time, variables]

        try:
            dataframe = dp.DataFrame(data)
            parcorr = ParCorr(significance='analytic')
            pcmci_instance = pcmci.PCMCI(dataframe=dataframe, cond_indep_test=parcorr)

            results = pcmci_instance.run_pcmci(tau_max=tau_max, pc_alpha=alpha_level)

            p_matrix = results['p_matrix']
            val_matrix = results['val_matrix']

            A = np.zeros((n_channels, n_channels), dtype = np.float32)

            for source in range(n_channels):
                for target in range(n_channels):
                    if source == target:
                        continue

                    link_strengths = []

                    for tau in range(1, tau_max + 1):
                        p_value = p_matrix[source, target, tau]
                        if p_value < alpha_level:
                            strength = abs(val_matrix[source, target, tau])
                            link_strengths.append(strength)
                    
                    if link_strengths:
                        A[target, source] = float(np.mean(link_strengths))
            matrices.append(A)
        except Exception:
            continue
    if len(matrices) == 0: 
        return np.zeros((n_channels, n_channels), dtype=np.float32)
    
    return np.mean(matrices, axis=0).astype(np.float32)


