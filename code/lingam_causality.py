import lingam
import numpy as np
from config import CONFIG


def compute_lingam_adjacency(preprocessed_epochs):
    """
    Calcola una matrice di adiacenza causale basata su VARLiNGAM.

    Stima un modello multivariato per epoca. Aggrega con una media i valori
    assoluti dei coefficienti sui lag selezionati, poi sulle epoche.
    B0 (effetti istantanei) e' incluso solo se configurato esplicitamente.

    Args:
        preprocessed_epochs (np.ndarray): array shape (n_epochs, n_channels, n_samples)
    Returns:
        np.ndarray: matrice A shape (n_channels, n_channels), non negativa,
                    con A[i,j] associato a j -> i e diagonale nulla.
                    I coefficienti non sono p-value ne' probabilita'; la
                    normalizzazione in [0,1] avviene in prepare_adjacency_for_hgcn.
    """
    n_epochs, n_channels, _ = preprocessed_epochs.shape
    matrices = []
    for epoch_idx, epoch in enumerate(preprocessed_epochs):
        if epoch_idx % 100 == 0:
            print(f"    VARLiNGAM epoch {epoch_idx + 1}/{n_epochs}")

        # lingam vuole shape [osservazioni, variabili]
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
                f"VARLiNGAM did not produce adjacency matrices for epoch {epoch_idx}."
            )

        # B[tau, target, source]: source(t-tau) -> target(t).
        # Riduciamo i lag PRIMA della media sulle epoche: BIC puo' selezionare
        # un numero diverso di lag per ciascuna epoca.
        B = np.asarray(B, dtype=np.float32)
        lag_matrices = B if CONFIG["varlingam_include_instantaneous"] else B[1:]

        if len(lag_matrices) == 0:
            A_epoch = np.zeros((n_channels, n_channels), dtype=np.float32)
        else:
            A_epoch = np.mean(np.abs(lag_matrices), axis=0)
        np.fill_diagonal(A_epoch, 0.0)

        matrices.append(A_epoch)
    
    if len(matrices) == 0:
        return np.zeros((n_channels, n_channels), dtype = np.float32)
    
    return np.mean(matrices, axis = 0).astype(np.float32)
