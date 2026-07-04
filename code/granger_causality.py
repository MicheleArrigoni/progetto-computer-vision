import warnings

import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

def compute_granger_adjacency(processed_epochs, max_diff, alpha):
    """
    Calcola una matrice di adiacenza causale basata sul test di Granger.

    Per ogni epoca e per ogni coppia ordinata (source -> target) esegue il test di
    Granger (più lags fino a `max_diff`) e conta i lag significativi (p < alpha).
    La componente A[target, source] accumula la frazione di lag significativi
    (significant_lags / max_diff) per epoca; alla fine si normalizza per il numero
    di epoche per ottenere una stima media.

    Args:
        processed_epochs (np.ndarray): array shape (n_epochs, n_channels, n_samples)
        max_diff (int): valore massimo di lag da testare nel Granger causality test
        alpha (float): soglia p-value per considerare un lag "significativo"

    Returns:
        np.ndarray: matrice A shape (n_channels, n_channels) con valori in [0,1], dove
                    A[i,j] indica la forza di evidenza che j causa i (j -> i).
    """

    n_epochs, n_channels, _ = processed_epochs.shape

    # inizializziamo la matrice adiacenza a zero
    A = np.zeros((n_channels, n_channels), dtype = np.float32)

    # Per ogni epoca calcoliamo i test di Granger per tutte le coppie di canali
    for epoch_idx, epoch in enumerate(processed_epochs):
        if epoch_idx % 100 == 0:
            print(f"    Granger epoch {epoch_idx + 1}/{n_epochs}")

        for target in range(n_channels):
            for source in range(n_channels):
                if source == target:
                    continue

                # grangercausalitytests richiede un array 2D con shape (n_obs, 2)
                # colonna 0 = target, colonna 1 = source (ordine importante)
                pair_data = np.column_stack([
                    epoch[target],
                    epoch[source]
                ])

                try:
                    # results è un dict: key = lag, value = tuple con i risultati dei test
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="verbose is deprecated.*", category=FutureWarning)
                        results = grangercausalitytests(pair_data, maxlag = max_diff, verbose = False)
                    significant_lags = 0

                    # controlliamo la statistica F (ssr_ftest) per ogni lag
                    for lag in range(1, max_diff + 1):

                        # estraiamo il p-value del test F per questo lag
                        # rappresenta la probabilità di osservare un miglioramento della predizione
                        # aggiungendo `source` al modello di predizione di `target`
                        p_value = results[lag][0]['ssr_ftest'][1]
                        if p_value < alpha:
                            significant_lags += 1

                    # accumuliamo la frazione di lag significativi per questa epoca
                    # A[target, source] rappresenta l'evidenza che `source` causa `target`
                    A[target, source] += significant_lags / max_diff
                except Exception:
                    # se il test fallisce (es. epoca troppo corta) ignoriamo la coppia
                    # e proseguiamo; questo evita che errori rompano l'intera pipeline
                    continue

    # media across-epochs per ottenere una stima robusta
    if n_epochs > 0:
        A /= n_epochs

    return A




