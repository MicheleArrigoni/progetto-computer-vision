from pathlib import Path


# Root del repository.
# `config.py` vive dentro la cartella `code`, quindi `parents[1]` risale alla
# directory principale del progetto. Tutti i percorsi nel dizionario `CONFIG`
# partono da qui, cosi' gli script funzionano indipendentemente dalla directory
# da cui vengono lanciati.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Configurazione generale della pipeline.
# I parametri sono tenuti in un unico dizionario per poterli riutilizzare in
# preprocessing, costruzione dei grafi e preparazione dei dati per il modello.
CONFIG = {
    # Percorsi principali: input BIDS/derivatives e output pronto per HGCN.
    "project_root": PROJECT_ROOT,
    "dataset_dir": PROJECT_ROOT / "dataset",
    "derivatives_dir": PROJECT_ROOT / "dataset" / "derivatives",
    "output_dir": PROJECT_ROOT / "data_hgcn_ready",

    # Frequenza di campionamento target dopo eventuale ricampionamento.
    # Ridurre la frequenza aiuta a contenere costo computazionale e memoria.
    "target_fs": 100,

    # Banda del filtro passa-banda EEG.
    # Manteniamo le componenti tra 0.5 e 24 Hz, tipiche per analisi su ritmi EEG
    # lenti/medi, rimuovendo drift molto lenti e componenti ad alta frequenza.
    "low_freq": 0.5,
    "high_freq": 40.0,

    # Frequenza del filtro notch per attenuare il rumore di rete elettrica.
    # In Europa il disturbo principale e' tipicamente a 50 Hz.
    "notch_freq": 50.0,

    # Segmentazione temporale: ogni epoca dura 5 secondi e, con overlap 0.0,
    # le finestre non si sovrappongono.
    "epoch_length_sec": 5,
    "overlap": 0.0,

    # Parametri per il test di causalita' di Granger.
    # `maxlag` e' il massimo ritardo valutato; `alpha` e' la soglia di
    # significativita'; `max_diffs` limita il numero di differenziazioni usate
    # per rendere stazionarie le serie temporali.
    "maxlag": 4,
    "alpha": 0.05,
    "max_diffs": 2,

    # Parametri VARLiNGAM: selezione dei lag tramite BIC e pruning.
    # Escludiamo gli effetti istantanei B0 dall'adiacenza aggregata.
    "varlingam_criterion": "bic",
    "varlingam_prune": True,
    "varlingam_include_instantaneous": False,

    # Metodi di causalita' da calcolare. Per smoke test rapidi si puo' ridurre
    # temporaneamente a ["granger"].
    "causality_methods": ["granger", "lingam", "pcmci"],
    "adjacency_method": "granger",
    # Ordine canonico dei canali EEG usato per costruire matrici e grafi.
    # Mantenere questo ordine fisso evita disallineamenti tra campioni diversi.
    "channel_order": [
        "Fp1", "Fp2", "F3", "F4", "C3", "C4",
        "P3", "P4", "O1", "O2", "F7", "F8",
        "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz",
    ],

    # Codifica numerica delle classi cliniche.
    # AD = Alzheimer Disease, CN = controllo sano, FTD = demenza frontotemporale.
    "labels": {
        "AD": 1,
        "CN": 0,
        "FTD": 2,
    },
    "add_self_loops": False,
}
