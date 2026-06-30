import lingam
import numpy as np


def compute_lingam_adiacency(preprocessed_epochs):
    """
    Calcola una matrice di adiacenza causale basata sul metodo LiNGAM.

    Per ogni epoca e per ogni coppia ordinata (source -> target) esegue il test di
    LiNGAM e accumula i risultati in una matrice di adiacenza.

    Args:
        preprocessed_epochs (np.ndarray): array shape (n_epochs, n_channels, n_samples)
    Returns:
        np.ndarray: matrice A shape (n_channels, n_channels) con valori in [0,1], dove
                    A[i,j] indica la forza di evidenza che j causa i (j -> i).
    """
    n_epochs, n_channels, _ = preprocessed_epochs.shape
    matrices = []
    for epoch in preprocessed_epochs:
        # lingam vuole shape [osservazioni, variabili]
        data = epoch.T

        try:
            model = lingam.DirectLiNGAM()
            model.fit(data)

            B = model.adjacency_matrix_

            # LiNGAM , B[i, j] rappresenta j --> i

            A_epoch = np.abs(B).astype(np.float32)
            np.fill_diagonal(A_epoch, 0.0)

            matrices.append(A_epoch)
        except Exception: 
            continue
    
    if len(matrices) == 0:
        return np.zeros((n_channels, n_channels), dtype = np.float32)
    
    return np.mean(matrices, axis = 0).astype(np.float32)
