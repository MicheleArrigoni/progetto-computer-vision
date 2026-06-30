from config import CONFIG
from preprocessing import (
    preprocess_epochs_for_subject,
    load_subject_raw,
    order_channels,
    filter_and_resample,
    split_in_epochs,
)
from granger import compute_granger_adjacency

def main():
    # 1. Preprocess: per ogni soggetto, carichiamo i dati, applichiamo
    # filtri, segmentazione in epoche e test di stazionarietà.
    all_processed_epochs = []
    print("Starting preprocessing for subjects...")
    for subject_dir in CONFIG["derivatives_dir"].glob("sub-*"):
        print(f"Preprocessing {subject_dir.name}...")
        try:
            raw = load_subject_raw(subject_dir)
            raw = order_channels(raw, CONFIG["channel_order"])
            raw = filter_and_resample(raw)

            eeg = raw.get_data()
            fs = raw.info.get('sfreq')

            epochs = split_in_epochs(eeg, fs, CONFIG.get('epoch_length_sec', CONFIG.get('epoch_lenght_sec', 10)))

            processed_epochs, stationarity_report, quality_report = preprocess_epochs_for_subject(epochs)

            if processed_epochs is None:
                print(f"  {subject_dir.name}: no valid epochs, skipping")
                continue

            all_processed_epochs.append(processed_epochs)
            print(f"  {subject_dir.name}: kept {quality_report['kept_epochs']} / {quality_report['total_epochs']} epochs")
        except Exception as e:
            print(f"  {subject_dir.name}: preprocessing failed: {e}")

    # 2. Costruzione dei grafi: usiamo le epoche preprocessate per calcolare
    # matrici di adiacenza causale basate sul test di Granger.
    adjacency_matrices = []
    print("Computing Granger adjacency matrices...")
    for idx, processed_epochs in enumerate(all_processed_epochs):
        print(f" Computing Granger for subject #{idx+1} ({processed_epochs.shape[0]} epochs)...")
        try:
            A = compute_granger_adjacency(
                processed_epochs,
                max_diff = CONFIG["max_diff"],
                alpha = CONFIG["alpha"]
            )
            adjacency_matrices.append(A)
            print(f"  Done: adjacency shape {A.shape}")
        except Exception as e:
            print(f"  Granger computation failed for subject #{idx+1}: {e}")

    # 3. Preparazione dei dati per il modello: a questo punto abbiamo le matrici
    # di adiacenza e possiamo costruire i dataset pronti per l'addestramento di HGCN.
    # Questa parte dipende molto dall'architettura specifica del modello e dalle
    # scelte di design (es. come rappresentare le features dei nodi, se usare
    # grafi dinamici o statici, ecc.). Qui potremmo normalizzare le matrici,
    # estrarre features aggiuntive dai nodi (es. bandpower) e salvare tutto in
    # un formato compatibile con la pipeline di addestramento.



if __name__ == "__main__":
    main()