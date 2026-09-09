import os
import json
import pickle
from pathlib import Path
import numpy as np
from scipy.signal import welch
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
import warnings

import mne
from config import CONFIG
import openneuro as on

# =====================================================================
# 5. FASE 1: IDENTIFICARE SOGGETTI E LABEL
# =====================================================================
# Identificazione dei soggetti sub-xxx e associazione alla classe clinica.
# =====================================================================
def get_subject_label(subject_id: str) -> str:
    """
    Mappa l'ID del soggetto alla classe clinica usando le convenzioni numeriche
    stabilite per il dataset OpenNeuro ds004504.
    """
    try:
        sub_number = int(subject_id[-3:])
    except ValueError:
        raise ValueError(f"ID soggetto non conforme allo standard BIDS (atteso sub-xxx): {subject_id}")

    if 1 <= sub_number <= 36:
        return "AD"
    if 37 <= sub_number <= 65:
        return "CN"
    if 66 <= sub_number <= 88:
        return "FTD"
    raise ValueError(f"ID soggetto fuori dal range operativo (001-088): {subject_id}")

def list_subjects(dataset_dir: str) -> list:
    """
    Rileva ed elenca in modo ordinato i percorsi delle cartelle soggetto validi.
    """
    dataset_path = Path(dataset_dir) 
    if not dataset_path.exists():
        raise FileNotFoundError(f"Directory del dataset non trovata: {dataset_path.resolve()}")
        
    # Filtra solo le directory che iniziano con il prefisso standard "sub-" e ordina alfabeticamente (sub-001, sub-002, ..., sub-088) 
    subjects = sorted([
        p for p in dataset_path.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    ])
    return subjects 

# =====================================================================
# 6. FASE 2: CARICARE E ALLINEARE GLI EEG
# =====================================================================
# 6.1 Caricamento 
# =====================================================================
def load_subject_raw(subject_dir: Path) -> mne.io.BaseRaw:
    """
    Individua il file EEGLAB (*_eeg.set) del singolo soggetto e lo carica via MNE.
    """
    subject_dir = Path(subject_dir)
    derivatives_eeg_dir = Path(CONFIG["derivatives_dir"]) / subject_dir.name / "eeg"

    # Ricerca prima nei file reali in `dataset/derivatives/<sub>/eeg`.
    eeg_files = []
    if derivatives_eeg_dir.exists():
        eeg_files = list(derivatives_eeg_dir.glob("*_eeg.set"))
    
    # Controllo di integrità sul numero di file trovati 
    if len(eeg_files) != 1:
        raise RuntimeError(
            f"{subject_dir.name}: expected one set file, "
            f"found {len(eeg_files)}" 
        )
        
    # Caricamento del file EEG pre-caricando i dati in memoria (preload=True) 
    raw = mne.io.read_raw_eeglab(
        str(eeg_files[0]),
        preload=True,  
        verbose=False 
    )
    return raw

# =====================================================================
# Allineamento dei canali 
# =====================================================================
def select_and_order_channels(raw: mne.io.BaseRaw, channel_order: list) -> mne.io.BaseRaw:
    """
    Seleziona e ordina i canali in modo rigido e coerente per la HGCN.
    Il nodo 0 corrisponderà sempre al primo canale, il nodo 1 al secondo, ecc.
    """
    # Verifica la presenza di tutti i canali target richiesti 
    missing = [ch for ch in channel_order if ch not in raw.ch_names]
    if missing:
        raise RuntimeError(f"Missing channels: {missing}") 
    # Estrae una copia ordinata basata sulla lista specificata (pick è la funzione aggiornata in MNE)
    raw_aligned = raw.copy().pick(picks=channel_order)
    return raw_aligned 

# =====================================================================
# FASE 3: FILTRAGGIO E RICAMPIONAMENTO
# =====================================================================
# Rimozione dei drift lenti e del rumore ad alta frequenza.
# Con target_fs a 50 Hz, la banda passante applicata è 0.5 Hz <= f < 24 Hz.
# =====================================================================
def filter_and_resample(raw: mne.io.BaseRaw, config: dict) -> mne.io.BaseRaw:
    """
    Applica i filtri passa-banda, il filtro notch e riduce la frequenza di campionamento.
    """
    raw_filtered = raw.copy() 
    # 1. Filtro Passa-Banda (FIR) con design 'firwin' 
    raw_filtered.filter(
        l_freq=config["low_freq"],   # 0.5 Hz 
        h_freq=config["high_freq"],  # 24.0 Hz 
        fir_design="firwin",         # Design del filtro FIR con finestra di Hanning (default di MNE)
        verbose=False                
    )
    
    # 2. Filtro Notch per abbattere la frequenza di rete (50 Hz) 
    if config["notch_freq"] is not None: 
        raw_filtered.notch_filter(
            freqs=[config["notch_freq"]], 
            verbose=False                 
        )
        
    # 3. Ricampionamento alla frequenza target (50 Hz) 
    raw_filtered.resample(
        sfreq=config["target_fs"],   
        npad="auto",      # Padding per evitare artefatti ai bordi durante il filtraggio           
        verbose=False                
    )
    
    return raw_filtered 

def split_fixed_length_epochs(eeg, fs, epoch_length_sec):
    """
    Suddivide il segnale continuo in epoche di lunghezza fissa non sovrapposte.
    
    Args:
        eeg (np.ndarray): Matrice del segnale EEG continuo. Shape: (n_channels, n_samples)
        fs (float): Frequenza di campionamento in Hz.
        epoch_length_sec (float): Durata di ciascuna epoca in secondi.
        
    Returns:
        np.ndarray: Tensore 3D contenente le epoche. Shape: (n_epochs, n_channels, samples_per_epoch)
    """
    samples_per_epoch = int(fs * epoch_length_sec)
    n_channels, n_samples = eeg.shape
    n_epochs = n_samples // samples_per_epoch
    epochs = []
    for i in range(n_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch
        # Viene estratto un frammento con shape: (n_channels, samples_per_epoch)
        epochs.append(eeg[:, start:end])
    # Lo stack raggruppa la lista in un array NumPy con shape finale: (n_epochs, n_channels, samples_per_epoch)
    return np.stack(epochs)
    
def is_good_epoch(epoch, amp_threshold=150.0):
    """
    Verifica se un'epoca supera i controlli di qualità (assenza di NaN, threshold di ampiezza e fluttuazione minima).
    
    Args:
        epoch (np.ndarray): Dati della singola epoca. Shape: (n_channels, samples_per_epoch)
        amp_threshold (float): Soglia massima di picco (es. in microvolt). Valori superiori scartano l'epoca.
        
    Returns:
        bool: True se l'epoca è considerata priva di artefatti invalidanti, False altrimenti.
    """
    # epoch is already in microvolt or standardized units,
    # depending on where this function is called.
    if not np.isfinite(epoch).all():
        return False
    if np.max(np.abs(epoch)) > amp_threshold:
        return False
    channel_std = epoch.std(axis=1)
    if np.any(channel_std < 1e-8):
        return False
    return True

def zscore_epoch(epoch, eps=1e-8):
    """
    Esegue la normalizzazione Z-score (standardizzazione) a livello di singolo canale per un'epoca.
    
    Args:
        epoch (np.ndarray): Dati della singola epoca. Shape: (n_channels, samples_per_epoch)
        eps (float): Piccolo valore (epsilon) per evitare la divisione per zero in canali piatti.
        
    Returns:
        np.ndarray: L'epoca normalizzata. Shape: (n_channels, samples_per_epoch)
    """
    mean = epoch.mean(axis=1, keepdims=True)  # Calcola media per canale. Shape: (n_channels, 1)
    std = epoch.std(axis=1, keepdims=True)    # Calcola deviazione standard per canale. Shape: (n_channels, 1)
    return (epoch - mean) / (std + eps)

def compute_mean_bandpower(ch_epochs: np.ndarray, fs: float, bands: dict) -> dict:
    """
    Calcola la potenza spettrale media per diverse bande di frequenza
    partendo dalle epoche di un singolo canale EEG.
    
    Parameters:
    -----------
    ch_epochs : np.ndarray
        Matrice NumPy con shape [n_epoche, n_campioni] per un singolo canale.
    fs : float
        Frequenza di campionamento del segnale (es. 50 Hz).
    bands : dict
        Dizionario con i limiti delle bande, es: {"delta": (0.5, 4), "theta": (4, 8), ...}
        
    Returns:
    --------
    bandpower : dict
        Dizionario contenente la potenza media calcolata per ogni banda.
    """
    # 1. Calcolo della PSD (Power Spectral Density) usando il metodo di Welch.
    # Calcoliamo la PSD lungo l'asse del tempo (axis=-1).
    # nperseg definisce la lunghezza della finestra. Per epoche corte a 50Hz (500 campioni), 
    # una finestra di 256 punti (circa 5 secondi) offre un ottimo compromesso di risoluzione.
    n_samples = ch_epochs.shape[1]
    nperseg = min(n_samples, 256)
    
    # Applica il Metodo di Welch per passare dal dominio del tempo al dominio della frequenza.
    # Trasforma il segnale frammentandolo in finestre (lunghe nperseg) e applicando la FFT.
    # Restituisce:
    # - frequencies: l'asse X, la lista delle frequenze estratte (es. 0.0, 0.2 ... fino a Nyquist).
    # - psds: l'asse Y (Densità di Potenza Spettrale), una matrice [n_epoche, n_frequenze] che indica
    #         l'energia del segnale per ogni frequenza in ciascuna singola epoca.
    frequencies, psds = welch(ch_epochs, fs=fs, nperseg=nperseg, axis=-1)
    
    # 2. Facciamo la media delle PSD di tutte le epoche per ottenere lo spettro 
    # medio del canale (shape finale: [n_frequenze])
    mean_psd = np.mean(psds, axis=0)
    
    # 3. Estrazione della potenza media per ogni banda spettrale
    bandpower = {}
    for band_name, (low_freq, high_freq) in bands.items():
        # Creiamo una maschera booleana per selezionare solo le frequenze nella banda target
        freq_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
        
        # Estraiamo i valori di potenza corrispondenti
        band_values = mean_psd[freq_mask]
        
        if len(band_values) == 0:
            # Se la risoluzione in frequenza è troppo bassa e non ci sono punti nella banda
            bandpower[band_name] = 0.0
        else:
            # Calcoliamo il valore medio di potenza nella banda
            bandpower[band_name] = float(np.mean(band_values))
            
    return bandpower

def make_stationary(time_series: np.ndarray, max_diffs: int = 2) -> np.ndarray:
    """
    Rende stazionaria una singola serie temporale necessaria per Granger.
    Utilizza l'Augmented Dickey-Fuller test per verificare la pendenza.
    """
    ts = time_series.copy()
    for _ in range(max_diffs + 1):
        # adfuller restituisce una tupla, il p-value è all'indice [1]
        p_value = adfuller(ts)[1]
        
        # Se la temp-series non ha trend statistici anomali (p < 0.05), va bene
        if p_value < 0.05:
            return ts
            
        # Altrimenti "appiattiamo" eventuali derive con la derivata numerica (diff)
        ts = np.diff(ts)
        
    return ts

def build_node_features(preprocessed_epochs, fs, stationary):
    """
    Costruisce le feature temporali e frequenziali per i nodi (canali EEG) del nostro grafo.
    Queste feature saranno l'input della Graph Convolutional Network (HGCN).
    Ad ogni iterazione elabora un singolo nodo (canale EEG).
    
    Args:
        preprocessed_epochs (np.ndarray): Tensore delle epoche pulite. Shape: (n_epochs, n_channels, samples_per_epoch)
        fs (float): Frequenza di campionamento.
        stationary (bool): Flag per controllare se calcolare indizi aggiuntivi (es. test di stazionarietà).
        
    Returns:
        list: Lista contenente i dizionari di feature estratte per ogni canale/nodo.
    """
    # 1. Estraiamo le dimensioni spaziali e temporali dal tensore 3D di input
    n_epochs, n_channels, _ = preprocessed_epochs.shape
    features = []

    # 2. Iteriamo su ogni singolo canale (che diventerà un 'nodo' nel nostro grafo)
    for ch_idx in range(n_channels):
        # Estraiamo tutte le epoche solo per il canale corrente.
        # Lo slicing [:, ch_idx, :] mantiene inalterato l'asse zero (epoche) e l'asse due (tempo),
        # schiacciando l'asse uno (canali).
        # Shape finale della variabile locale: (n_epochs, samples_per_epoch)
        ch_epochs = preprocessed_epochs[:, ch_idx, :]

        # 3. Estrazione delle feature in frequenza per il canale analizzato
        # Richiama la funzione di Welch commentata in precedenza per misurare la potenza delle 
        # ritmiche cerebrali classiche (ritmi).
        bandpower = compute_mean_bandpower(
            ch_epochs,
            fs=fs,
            bands={
                "delta": (0.5, 4),  # Onde Delta: associate al sonno profondo
                "theta": (4, 8),    # Onde Theta: associate al rilassamento o dormiveglia
                "alpha": (8, 13),   # Onde Alpha: presenti in veglia rilassata ad occhi chiusi
                "beta": (13, 24)    # Onde Beta: marcatore di veglia attiva e concentrazione visiva
            }
        )

        # 4. Creiamo un vettore combinato di queste feature per il nodo corrente
        node_feat = [
            bandpower["delta"], 
            bandpower["theta"], 
            bandpower["alpha"], 
            bandpower["beta"]
        ]
        
        # 5. Se è richiesto di inserire indici stazionari/temporali aggiuntivi
        if stationary:
            # Aggiungiamo ad esempio la deviazione standard del segnale 
            # come misura statistica della sua fluttuazione media
            signal_std = float(np.std(ch_epochs))
            node_feat.append(signal_std)
            
        # Aggiungiamo le feature del canale corrente alla mappa spaziale generale
        features.append(node_feat)

    # 6. Convertiamo e restituiamo un array NumPy compatto per l'addestramento
    # Shape risultante: (n_channels, n_features), es: (19, 4) o (19, 5) a seconda del flag stationary
    return np.array(features)

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

def make_epoch_stationary(epoch, channels, max_diffs=2):
    # epoch shape: channels x samples
    current = epoch.copy()
    report = {
    "n_diffs": 0,
    "diffed": False,
    "channels": {}
    }
    for diff_id in range(max_diffs + 1):
        non_stationary = []
        for idx, ch in enumerate(channels):
            x = current[idx]
            adf_ok = passes_adf_test(x)
            kpss_ok = passes_kpss_test(x)
            report["channels"].setdefault(ch, []).append({
            "diff_id": diff_id,
            "adf_ok": adf_ok,
            "kpss_ok": kpss_ok
            })
            if not (adf_ok and kpss_ok):
                non_stationary.append(ch)
        if len(non_stationary) == 0:
            return current, report
        if diff_id < max_diffs:
            current = np.diff(current, axis=1)
        report["n_diffs"] += 1
        report["diffed"] = True
    return current, report

# =====================================================================
# BLOCCO DI VALIDAZIONE LOCALE (ORCHESTRAZIONE SPERIMENTALE)
# =====================================================================
if __name__ == "__main__":

    '''
    # ID del dataset di OpenNeuro
    dataset_id = "ds004504"

    # Scarica l'INTERO dataset con i file reali completi nella cartella specifica
    # (Se vuoi solo testare un soggetto alla volta, aggiungi include=["sub-001*"])
    on.download(dataset=dataset_id, target_dir="../dataset/derivatives")
    print("Download completato! Ora hai i file .set e .fdt reali sul disco.")
    '''

    print("=== PIPELINE EEG -> HGCN (FASE 1 A FASE 3 CORRETTAMENTE MAPPATA) ===")
    
    # Simulazione e verifica del flusso logico
    dataset_dir = Path(CONFIG["derivatives_dir"])
    if not dataset_dir.exists():
        print(f"\n[Nota] Cartella '{CONFIG['derivatives_dir']}' non presente.")
    else:
        # Esecuzione prototipo sul primo soggetto rilevato
        subjects = list_subjects(CONFIG["derivatives_dir"])
        if subjects:
            processed_subjects = []
            
            for subject in subjects:

                print(f"\nProcessando {subject.name}...")

                # Esecuzione Fase 1
                label = get_subject_label(subject.name)
                print(f" -> Fase 1 completata. Label medica associata: {label}")
                
                # Esecuzione Fase 2
                raw_data = load_subject_raw(subject)
                raw_ordered = select_and_order_channels(raw_data, CONFIG["channel_order"])
                print(f" -> Fase 2 completata. Canali ordinati correttamente")
                
                # Esecuzione Fase 3
                raw_preprocessed = filter_and_resample(raw_ordered, CONFIG)
                final_matrix: np.ndarray = raw_preprocessed.get_data()
                # final_matrix mantiene la shape bidimensionale originale, es. (19, n_samples_post_resampling)
                print(f" -> Fase 3 completata. Matrice EEG preelaborata estratta con successo [Dim: {final_matrix.shape}].")

            
                eeg_data = raw_preprocessed.get_data()
                fs = raw_preprocessed.info['sfreq']
                epochs = split_fixed_length_epochs(eeg_data, fs, CONFIG["epoch_length_sec"])
                # epochs assume shape tridimensionale: (n_epochs, n_channels, samples_per_epoch)
                print(f" -> Fase 4 completata. Suddivisione in epoche completata. Numero di epoche generate: {epochs.shape[0]}")

                epochs_zscored = []
                for i, epoch in enumerate(epochs):
                    if is_good_epoch(epoch, amp_threshold=150.0):
                         # 1. Normalizzazione z-score
                        epoch_norm = zscore_epoch(epoch)
                        
                        # 2. Rendiamo l'epoca stazionaria per i test di Granger successivi
                        # Utilizziamo la funzione make_epoch_stationary definita in precedenza
                        epoch_stat, report = make_epoch_stationary(
                            epoch=epoch_norm, 
                            channels=CONFIG["channel_order"], 
                            max_diffs=2
                        )
                        
                        # 3. Aggiungiamo l'epoca (ora stazionaria) alla nostra lista
                        epochs_zscored.append(epoch_stat)                      
                
                # Lo stack condensa la lista in un NumPy array strutturato
                # Shape garantita dopo lo stack: (n_valid_epochs, n_channels, samples_per_epoch)
                if epochs_zscored:
                    min_samples = min(epoch.shape[1] for epoch in epochs_zscored)
                    epochs_zscored = [epoch[:, :min_samples] for epoch in epochs_zscored]
                    epochs_array = np.stack(epochs_zscored)
                else:
                    epochs_array = np.array([])
                
                if len(epochs_array) > 0:
                    # Raccogliamo i dati puliti raggruppandoli in un dizionario nativo
                    processed_subjects.append({
                        "id": subject.name,
                        "label": label,
                        "epochs": epochs_array  # Salvataggio tensore EEG normalizzato con shape 3D
                    })
                else:
                    print(f" -> Attenzione: Nessuna epoca valida estratta per {subject.name} (scartato).")
                
            
            # Calcolo e crop al numero minimo di epoche
            if processed_subjects:
                min_epochs = min(len(subj["epochs"]) for subj in processed_subjects)
                print(f"\nNumero minimo di epoche trovato: {min_epochs}. Procedo al ritaglio...")
                
                for subj in processed_subjects:
                    # Lo slicing [ :min_epochs ] agisce sul primo asse del tensore Numpy (l'asse 0, quello delle epoche).
                    # Aggiunge una nuova chiave "cropped_epochs" con shape finale omogenea (min_epochs, n_channels, samples_per_epoch)
                    subj["cropped_epochs"] = subj["epochs"][:min_epochs]
                    
                print(f"Crop applicato con successo! {len(processed_subjects)} soggetti ora hanno {min_epochs} epoche ciascuno.")
            else:
                print("\nNessun dato valido estratto dai soggetti processati.")

        else:
            print(f"\nNessun soggetto trovato in '{CONFIG['derivatives_dir']}'. Assicurati di avere i dati corretti o di aver scaricato il dataset.")
