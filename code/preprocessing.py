import os
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.signal import welch
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss, InterpolationWarning
import warnings

import mne
from config import CONFIG
import openneuro as on

def get_subject_label(subject_id: str):
    '''
    Restituisce la label clinica per un dato subject_id.

    Args:
        subject_id (str): stringa identificativa del soggetto (es. 'sub-001').

    Returns:
        str: una label tra 'AD', 'CN' o 'FTD'.
    Raises:
        ValueError: se l'ID non rientra nell'intervallo previsto.
    '''

    # uses only the last three digits of the subject ID to determine the label
    n = int(subject_id[-3:])

    # in this particular dataset the first 36 patient are "AD"
    # the patient from id = 37 to id = 65 are "CN"
    # the others are "FDT"

    if 0 <= n <= 36:
        return "AD"
    elif 37 <= n <= 65:
        return "CN"
    elif 66 <= n <= 88:
        return "FTD"
    else:
        raise ValueError(f"Invalid subject ID: {subject_id}")

def get_subjects_list(data_dir: str):
    '''
    Ritorna la lista ordinata di directory soggetto presenti in `data_dir`.

    Args:
        data_dir (str): percorso alla cartella del dataset.

    Returns:
        List[Path]: lista di oggetti `Path` per le cartelle che iniziano con 'sub-'.
    '''
 
    # to have access to path methods
    dataset_dir = Path(data_dir)

    subjects = sorted([
        p for p in dataset_dir.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    ])
    return subjects



# EEG RAW LOADING AND PREPROCESSING

def load_subject_raw(subject_dir):
    '''
    Carica il file EEGLAB (.set) presente nella sottocartella 'eeg' di `subject_dir` e
    restituisce l'oggetto MNE Raw precaricato.

    Args:
        subject_dir (Path): percorso alla cartella del soggetto (Path o stringa).

    Returns:
        mne.io.Raw: oggetto Raw con `preload=True`.

    Raises:
        RuntimeError: se non è presente esattamente un file .set nella sottocartella 'eeg'.
    '''

    # Lista di file eeg .set nella sottocartella 'eeg' del  singolo soggetto
    eeg_files = list((subject_dir/"eeg").glob("*_eeg.set"))

    if len(eeg_files) != 1:
        raise RuntimeError(f"error: expected one .set file \n, found: {len (eeg_files)}")
    

    # Crea il file mne.io.Raw a partire dal file EEGLAB .set, con preload=True per caricare i dati in memoria
    raw = mne.io.read_raw_eeglab(
        eeg_files[0],
        preload = True,
        verbose = False
    )

    # Il valore restituito `raw` è un oggetto MNE Raw (es. RawArray/RawEEGLAB) con le seguenti
    # proprietà/metodi utili (elenco non esaustivo):
    # - raw.info: dizionario con metadata (sfreq, ch_names, nchan, meas_date, description, ecc.)
    # - raw.ch_names: lista dei nomi dei canali
    # - raw.nchan / raw.info['nchan']: numero di canali
    # - raw.times: array dei tempi per ciascun campione (in secondi)
    # - raw.get_data(): restituisce i dati come array NumPy shape (n_channels, n_samples)
    # - raw.copy(), raw.crop(tmin, tmax), raw.resample(new_sfreq): metodi per manipolare i dati
    # - raw.annotations: oggetto con eventi/annotazioni temporali
    # - raw.info['sfreq']: frequenza di campionamento
    # - raw.load_data() / preload True: i dati sono precaricati in memoria se `preload=True`
    # - raw.plot(), raw.plot_psd(): metodi di visualizzazione utili per ispezione
    # - raw.pick_channels(list_of_names): selezione canali
    # Tipico uso rapido: `data = raw.get_data(); ch_names = raw.ch_names; sfreq = raw.info['sfreq']`

    return raw

def order_channels(raw, channel_order):
    '''
    Verifica la presenza dei canali richiesti in `raw` e restituisce una copia con i canali
    ordinati secondo `channel_order`.

    Args:
        raw (mne.io.Raw): oggetto Raw da cui selezionare/ordinare i canali.
        channel_order (List[str]): lista di nomi di canale nell'ordine desiderato.

    Returns:
        mne.io.Raw: copia di `raw` con canali ordinati.

    Raises:
        RuntimeError: se mancano canali richiesti.
    '''

    missing = [ch for ch in channel_order if ch not in raw.ch_names]

    if missing:
        raise RuntimeError(f"{missing} channels are missing, subject{raw.getName()}")

    # `RawEEGLAB` objects do not expose `pick_channel` (singular).
    # Use `pick_channels` which accepts a list of names and preserves the order.
    raw = raw.copy().pick(picks=channel_order)
    return raw

def filter_and_resample(raw):
    '''
    Applica filtro passa-banda e notch (se configurato) su una copia di `raw`.

    Args:
        raw (mne.io.Raw): oggetto Raw in ingresso.

    Returns:
        mne.io.Raw: copia filtrata di `raw`.
    '''
    raw = raw.copy()

    raw.filter(
        l_freq = CONFIG["low_freq"],
        h_freq = CONFIG["high_freq"],
        fir_design="firwin",
        verbose = False
    )

    if CONFIG["notch_freq"] is not None:
        raw.notch_filter(
            CONFIG["notch_freq"],
            verbose = False
        )
    return raw

def split_in_epochs(eeg, fs , epoch_lenght_sec):
    '''
    Suddivide una matrice EEG continua in epoche di lunghezza fissa.

    Args:
        eeg (np.ndarray): array shape (n_channels, n_samples).
        fs (float): frequenza di campionamento in Hz.
        epoch_lenght_sec (float): durata di ciascuna epoca in secondi.

    Returns:
        np.ndarray: array shape (n_epochs, n_channels, samples_per_epoch).
    '''
    samples_per_epoch = int (fs*epoch_lenght_sec)
    n_channels, n_samples = eeg.shape
    n_epochs = n_samples // samples_per_epoch

    epochs = []

    for i in range (n_epochs):
        start = i * samples_per_epoch
        end = (i+1) * samples_per_epoch

        # selects all rows (the cahnnels) from column start to end
        epochs.append(eeg[:, start:end])

    # if there are remaining samples that do not fit into a full epoch, we can discard them or handle them as needed
    # shape of the output will be (n_epochs, n_channels, samples_per_epoch)
    return np.stack(epochs)

def make_epoch_stationary(epoch, channels, max_diff):
    '''
    Applica differenziazioni progressive per rendere stazionaria l'epoca (per canale), valutando
    ADF e KPSS ad ogni passo.

    Args:
        epoch (np.ndarray): epoca shape (n_channels, samples_per_epoch).
        channels (List[str]): lista dei nomi dei canali corrispondenti agli indici di `epoch`.
        max_diff (int): numero massimo di differenziazioni consentite.

    Returns:
        Tuple[np.ndarray, dict]: (epoca trasformata, report sulla stazionarietà contenente per-canale
        i risultati dei test e il numero di differenze applicate).
    '''
    # epoch shape: (n_channels, samples_per_epoch)
    current = epoch.copy()
    report = {
        "n_diffs": 0,
        "diffed": False,
        "channels": {}
    }

    for diff_id in range (max_diff + 1):
        non_stationary = []
        
        for idx, ch in enumerate(channels):
            x = current [idx]
            adf_ok = passes_adf_test(x)
            kpss_ok = passes_kpss_test(x)

            report["channels"].setdefault(ch, []).append({
                "diff_id": diff_id,
                "adf_ok": adf_ok,
                "kpss_ok": kpss_ok
            })

            if not (adf_ok and kpss_ok):
                non_stationary.append(ch)
        
        if len (non_stationary) == 0:
            return current, report
        
        if diff_id < max_diff:
            current = np.diff(current, axis = 1)
            report["n_diffs"] += 1
            report["diffed"] = True

    return current, report

def passes_adf_test(x: np.ndarray, alpha: float = 0.05, maxlag: int = 4) -> bool:
    """
    Ritorna True se il segnale supera il test ADF (quindi indica STAZIONARIETÀ).
    Ipotesi nulla (H0): Serie NON stazionaria (ha una radice unitaria).
    Se p-value <= alpha, rifiutiamo H0 -> La serie è stazionaria.
    """
    try:
        # Calcoliamo il test di Dickey-Fuller aumentato
        result = adfuller(x, maxlag=maxlag)
        p_value = result[1]  # Il p-value è il secondo elemento restituito
        
        # Supera il test se il p-value è inferiore o uguale alla soglia alpha
        return p_value <= alpha
    except Exception:
        # In caso di errori di convergenza numerica, restituisce False per sicurezza
        return False

def passes_kpss_test(x: np.ndarray, alpha: float = 0.05) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            result = kpss(x, regression='c', nlags="auto")

        p_value = result[1]
        return p_value > alpha
    except Exception:
        return False

def is_good_epoch(epoch, amp_treshold = 150.0):
    '''
    Controlla la qualità di una epoca: valori finiti, ampiezza sotto soglia e varianza non nulla.

    Args:
        epoch (np.ndarray): epoca shape (n_channels, samples_per_epoch).
        amp_treshold (float): soglia massima per l'ampiezza assoluta dei segnali.

    Returns:
        bool: True se l'epoca è considerata buona, False altrimenti.
    '''
    if not np.isfinite(epoch).all():
        return False

    if np.max(np.abs(epoch)) > amp_treshold:
        return False

    channel_std = epoch.std(axis = 1) 
    if np.any (channel_std < 1e-8):
        return False
    
    return True

def zscore_epoch(epoch, eps=1e-8):
    """
    Esegue la normalizzazione Z-score (standardizzazione) a livello di singolo canale per un'epoca.
    Rappresenta di quanto un valore si discosta dalla media del canale in termini di deviazioni standard.
    
    Args:
        epoch (np.ndarray): Dati della singola epoca. Shape: (n_channels, samples_per_epoch)
        eps (float): Piccolo valore (epsilon) per evitare la divisione per zero in canali piatti.
        
    Returns:
        np.ndarray: L'epoca normalizzata. Shape: (n_channels, samples_per_epoch)
    """
    mean = epoch.mean(axis=1, keepdims=True)  # Calcola media per canale. Shape: (n_channels, 1)
    std = epoch.std(axis=1, keepdims=True)    # Calcola deviazione standard per canale. Shape: (n_channels, 1)
    return (epoch - mean) / (std + eps)

def crop_to_same_lenght(epochs):
    '''
    Esegue il crop di tutte le epoche alla lunghezza minima trovata tra di esse.

    Args:
        epochs (List[np.ndarray]): lista di epoche shape (n_channels, samples_i).

    Returns:
        np.ndarray: array shape (n_epochs, n_channels, min_samples).
    '''
    min_len = min(epoch.shape[1] for epoch in epochs)
    return np.stack([epoch[:, :min_len] for epoch in epochs])

def preprocess_epochs_for_subject(epochs):
    '''
    Applica controlli di qualità, normalizzazione e prova a rendere stazionarie le epoche
    restituendo il tensore preprocessato e report.

    Args:
        epochs (np.ndarray or List[np.ndarray]): lista o array di epoche raw.

    Returns:
        Tuple[np.ndarray or None, list, dict]: (preprocessed_epochs o None se nessuna valida,
        stationarity_report, quality_report).
    '''
    good_epochs = []
    stationarity_report = []

    quality_report = {
        "total_epochs": int(len(epochs)),
        "kept_epochs": 0,
        "discarded_epochs": 0
    }

    for epoch_idx, epoch in enumerate(epochs):
        ok = is_good_epoch(epoch)

        if not ok:
            quality_report["discarded_epochs"] += 1
            continue

        epoch = zscore_epoch(epoch)
        epoch_stat, report = make_epoch_stationary(
            epoch,
            channels = CONFIG["channel_order"],
            max_diff = CONFIG["max_diff"]        
        )


        good_epochs.append(epoch)
        stationarity_report.append(report)

        quality_report["kept_epochs"] += 1
    # after evaluating all epochs, check how many good epochs we collected
    if len(good_epochs) == 0:
        return None, stationarity_report, quality_report

    processed_epochs = crop_to_same_lenght(good_epochs)
    return processed_epochs, stationarity_report, quality_report

def stationary_feature_for_channel(stationarity_report, channel_idx):
    '''
    Calcola una feature riassuntiva di stazionarietà per un singolo canale.

    La funzione scorre i report prodotti da `make_epoch_stationary` e conta:
    - quante epoche hanno coinvolto il canale indicato
    - quante di queste epoche hanno richiesto almeno una differenziazione

    Il valore restituito è la frazione:

        epoche_differenziate / epoche_valide_per_il_canale

    Args:
        stationarity_report (list): lista di report per epoca, generata da
            `make_epoch_stationary`.
        channel_idx (int): indice del canale dentro `CONFIG["channel_order"]`.

    Returns:
        float: valore compreso tra 0 e 1, oppure 0.0 se il canale non compare
        in nessun report valido.
    '''
    diffs_required = 0
    total = 0
    ch_name = CONFIG["channel_order"][channel_idx]

    for epoch_report in stationarity_report:
        # Recupera i controlli registrati per questo canale nella singola epoca.
        checks = epoch_report["channels"].get(ch_name, [])

        # Se il canale non è stato valutato in questa epoca, la ignoriamo.
        if len (checks) == 0: continue

        total += 1

        # Se l'epoca è stata differenziata almeno una volta, contribuisce al conteggio.
        if epoch_report["diffed"]:
            diffs_required += 1

    # Se non abbiamo trovato alcuna epoca valida per il canale, evitiamo divisione per zero.
    if total == 0:
        return 0.0

    # Valore alto = il canale ha spesso richiesto differenziazione per risultare stazionario.
    return diffs_required / total

def mean_bandpower_epochs(ch_epochs, fs, band, relative=False, nperseg=None):
    """
    Calcola la mean bandpower per canale in modo semplice.

    Args:
        ch_epochs (np.ndarray): shape (n_epochs, n_samples) per un singolo canale.
        fs (float): sampling frequency in Hz.
        band (tuple): (fmin, fmax) della banda.
        relative (bool): se True restituisce bandpower relativa (bandpower/total_power).
        nperseg (int|None): parametro per welch (default min(256, n_samples)).

    Returns:
        float: valore medio della bandpower across-epochs (o frazione media se relative=True).
    """
    fmin, fmax = band
    nperseg = nperseg or min(256, ch_epochs.shape[1])
    bp_list = []
    frac_list = []

    for epoch in ch_epochs:
        f, Pxx = welch(epoch, fs=fs, nperseg=nperseg)
        mask = (f >= fmin) & (f <= fmax)
        bp = np.trapz(Pxx[mask], f[mask]) if mask.any() else 0.0
        if relative:
            total = np.trapz(Pxx, f) if f.size and Pxx.size else 0.0
            frac_list.append(bp / total if total > 0 else 0.0)
        else:
            bp_list.append(bp)

    return float(np.mean(frac_list if relative else bp_list)) if (relative and frac_list) or (not relative and bp_list) else 0.0

def build_node_features(preprocessed_epochs, fs, stationarity_report):
    '''
    Costruisce il vettore di feature per ciascun canale a partire dalle epoche preprocessate.

    Features calcolate per canale (aggregazione across-epochs via media):
      - bandpower delta (0.5-4 Hz)
      - bandpower theta (4-8 Hz)
      - bandpower alpha (8-13 Hz)
      - bandpower beta  (13-24 Hz)
      - mean (valore medio del segnale)
      - std (deviazione standard)
      - var (varianza)
      - stationarity_fraction (frazione di epoche che hanno richiesto differenziazione)

    Args:
        preprocessed_epochs (np.ndarray): array shape (n_epochs, n_channels, n_samples).
        fs (float): frequenza di campionamento in Hz.
        stationarity_report (list): report generato da `make_epoch_stationary` per ogni epoca.

    Returns:
        np.ndarray: matrice delle feature nodali shape (n_channels, 8), dtype float32.
    '''
    n_epochs, n_channels, n_samples = preprocessed_epochs.shape
    features = []

    for ch_idx in range(n_channels):
        ch_epochs = preprocessed_epochs[:, ch_idx, :]

    # definizione bande
    bandpower = mean_bandpower_epochs(
        ch_epochs,
        fs = CONFIG["fs"],
        bands = {
            "delta": (0.5, 4),  
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 24)
            }
        )
    mean = ch_epochs.mean()
    std = ch_epochs.std()
    var = ch_epochs.var()

    # calcolo frazione di epoche che hanno richiesto differenziazione
    stat_value = stationary_feature_for_channel(
        stationarity_report,
        channel_idx = ch_idx
    )


    node_features = [
        bandpower["delta"],
        bandpower["theta"],
        bandpower["alpha"],
        bandpower["beta"],
        mean,
        std,
        var,
        stat_value  
    ]

    features.append(node_features)

    return np.array(features, dtype=np.float32)

