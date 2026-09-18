# Cheatsheet del progetto EEG

Guida rapida, in italiano, alle funzioni e alle operazioni usate nel progetto. Le sezioni sono divise per libreria di appartenenza.

## Mappa del progetto

- `code/main.py`: coordina la pipeline completa.
- `code/config.py`: contiene percorsi e parametri nella variabile `CONFIG`.
- `code/preprocessing.py`: carica, filtra, divide e normalizza i segnali EEG; calcola le feature.
- `code/granger_causality.py`: calcola grafi causali con Granger.
- `code/lingam_causality.py`: calcola grafi causali con VARLiNGAM.
- `code/pcmci_causality.py`: calcola grafi causali con PCMCI/Tigramite.
- `code/dataset_module.py`: prepara e salva i grafi per HGCN.
- `code/ds_downloader.ipynb`: scarica il dataset OpenNeuro.
- `code/ds_download.py`: scarica BrainLat da Synapse in `brainlat_dataset/`.
- `code/brainlat_adapter.py`: converte BrainLat nel layout della pipeline e mappa gli elettrodi sui 19 canali canonici.
- `models/mlp.py`: caricamento dei campioni, split, standardizzazione, training e valutazione MLP.
- `models/logistic.py`: regressione logistica multiclasse con le utility condivise di `mlp.py`.
- `pyproject.toml`, `uv.lock`, `.python-version`: dipendenze e ambiente Python 3.12 gestito con uv.
- `src/progetto_computer_vision/__init__.py`: entry point del pacchetto; attualmente stampa un saluto e non avvia la pipeline.

Il vecchio prototipo `code/test_load.py` è stato rimosso.

## Flusso principale

```text
OpenNeuro / BrainLat (dopo brainlat_adapter.py)
      -> MNE: caricamento EEG
      -> MNE: canali, filtro e ricampionamento
      -> NumPy: epoche e normalizzazione
      -> epoche Z-score prima delle differenze: feature SciPy/NumPy
      -> epoche con differenze ADF/KPSS: grafi Granger/LiNGAM/PCMCI
      -> NumPy: normalizzazione matrici
      -> pickle/JSON/CSV/NPY: salvataggio HGCN
      -> PyTorch: baseline MLP / regressione logistica
      -> JSON: metriche e soggetti degli split
```

# 1. Python standard

Queste funzioni appartengono alla libreria standard di Python e non richiedono installazione aggiuntiva.

## `pathlib.Path`

Usato in configurazione, preprocessing, esportazione, adattatore BrainLat, classificatori e notebook.

| Operazione | Significato |
|---|---|
| `Path(percorso)` | Trasforma una stringa in un oggetto percorso. |
| `Path(__file__)` | Percorso del file Python corrente. |
| `.resolve()` | Restituisce il percorso assoluto. |
| `.parents[1]` | Risale di due livelli nella gerarchia delle cartelle. |
| `/` tra `Path` | Unisce parti di percorso: `root / "dataset"`. |
| `.iterdir()` | Elenca il contenuto di una cartella. |
| `.is_dir()` | Verifica se un elemento è una directory. |
| `.exists()` | Verifica se un percorso esiste. |
| `.glob("sub-*")` | Cerca elementi che rispettano un modello. |
| `.name` | Restituisce solo il nome del file o della cartella. |
| `.mkdir(parents=True, exist_ok=True)` | Crea una cartella e le cartelle genitore mancanti. |

## `os`, `warnings`

- `os`: modulo per funzioni generiche del sistema operativo. È importato nel progetto, ma non è centrale nella pipeline attuale.
- `warnings.catch_warnings()`: limita la gestione degli avvisi al blocco `with`.
- `warnings.simplefilter("ignore", WarningType)`: nasconde uno specifico tipo di warning.

## `json`

Gestisce dati nel formato JSON.

- `json.dump(obj, file, indent=2)`: scrive un dizionario o una lista in un file JSON.
- `json.load(file)`: legge un JSON; i classificatori lo usano per caricare la label da `y.json`.
- `json.dumps(obj)`: convertirebbe un oggetto in stringa JSON.

## `csv`

Gestisce file tabellari CSV.

- `csv.DictWriter(file, fieldnames=[...])`: crea uno scrittore che usa le chiavi di un dizionario come colonne.
- `.writeheader()`: scrive la riga con i nomi delle colonne.
- `.writerow({...})`: scrive una riga.

## `pickle`

Serializza oggetti Python.

- `pickle.dump(graphs, file)`: salva l'oggetto `graphs` in formato binario.
- `pickle.load(file)`: ricarica un oggetto salvato con pickle.
- `with open(percorso, "rb")`: apre un file binario in lettura.

## Funzioni e sintassi Python ricorrenti

| Costrutto | Significato |
|---|---|
| `def nome(argomento):` | Definisce una funzione. |
| `return risultato` | Restituisce il risultato della funzione. |
| `if / elif / else` | Sceglie quale blocco eseguire. |
| `for x in elementi` | Ripete il blocco per ogni elemento. |
| `enumerate(lista)` | Itera ottenendo indice e valore. |
| `range(n)` | Crea gli indici da `0` a `n-1`. |
| `try / except` | Gestisce errori senza interrompere necessariamente tutto il programma. |
| `with open(...) as f` | Apre una risorsa e la chiude automaticamente. |
| `f"testo {x}"` | F-string: inserisce `x` dentro il testo. |
| `[x for x in lista if condizione]` | List comprehension: costruisce una lista filtrata o trasformata. |
| `{chiave: valore for ...}` | Dictionary comprehension: costruisce un dizionario. |
| `obj.metodo(...)` | Chiama un metodo dell'oggetto. |
| `if __name__ == "__main__"` | Esegue il blocco solo quando il file è avviato direttamente. |

# 2. NumPy

NumPy gestisce gli array numerici e le operazioni matematiche sui dati EEG.

## Creazione e conversione

- `np.array(lista, dtype=np.float32)`: converte una lista in array numerico.
- `np.asarray(A, dtype=np.float32)`: converte in array, evitando una copia se possibile.
- `np.zeros((n, m), dtype=np.float32)`: crea una matrice piena di zeri.
- `np.stack(lista)`: impila array della stessa forma aggiungendo un asse.
- `np.column_stack((a, b))`: mette vettori come colonne affiancate.
- `np.asarray(lista)[:min_epochs]`: converte e poi applica uno slice.

## Statistica e controllo dati

- `np.mean(x, axis=0)`: media lungo l'asse indicato.
- `np.std(x, axis=1)`: deviazione standard.
- `np.var(x)`: varianza.
- `np.max(x)`: massimo.
- `np.abs(x)`: valore assoluto elemento per elemento.
- `np.isfinite(x)`: maschera booleana per valori finiti.
- `.all()`: `True` se tutti gli elementi sono veri.
- `np.any(x)`: `True` se almeno un elemento è vero.
- `np.nan_to_num(x, nan=0, posinf=0, neginf=0)`: sostituisce valori non validi.
- `np.fill_diagonal(A, value)`: modifica la diagonale della matrice.
- `A.T`: trasposta della matrice.
- `np.trapezoid(y, x)`: integra usando il metodo dei trapezi.
- `float(x)`, `int(x)`, `bool(x)`: convertono un valore in tipo Python standard.

## Indici e forme

- `x.shape`: tupla con le dimensioni dell'array.
- `x.shape[0]`: dimensione del primo asse.
- `x[:, start:end]`: tutti i canali, intervallo temporale selezionato.
- `x[:, ch_idx, :]`: tutte le epoche e i campioni di un canale.
- `x[:n]`: primi `n` elementi.
- `x[indices]`: seleziona gli elementi agli indici indicati.
- `np.linspace(start, stop, num, dtype=int)`: crea `num` valori equidistanti.
- `//`: divisione intera, utile per calcolare quante finestre complete entrano nel segnale.

### Mini esempio sulle dimensioni

```python
import numpy as np

# 2 epoche, 3 canali, 4 campioni per canale
epochs = np.zeros((2, 3, 4))

print(epochs.shape)       # (2, 3, 4)
print(epochs.shape[0])    # 2: numero di epoche
print(epochs.shape[1])    # 3: numero di canali
print(epochs.shape[2])    # 4: campioni per epoca

one_epoch = epochs[0]
print(one_epoch.shape)    # (3, 4): una epoca, tutti i canali

one_channel = epochs[:, 1, :]
print(one_channel.shape)  # (2, 4): tutte le epoche del canale 1

one_sample = epochs[0, 1, 2]
print(one_sample.shape)   # (): un singolo numero, quindi nessuna dimensione
```

L'ordine degli assi è importante:

```text
epochs[epoca, canale, campione]
```

Per esempio, `epochs[:, 1, :]` significa: prendi tutte le epoche, il canale con indice `1` e tutti i campioni. Gli indici Python partono da `0`, quindi il canale con indice `1` è il secondo canale.

## Broadcasting

In `zscore_epoch`:

```python
mean = epoch.mean(axis=1, keepdims=True)  # (canali, 1)
result = (epoch - mean) / std             # (canali, campioni)
```

NumPy espande automaticamente `(canali, 1)` su tutti i campioni del canale.

# 3. SciPy

Il progetto usa `scipy.signal.welch` per l'analisi in frequenza.

## `scipy.signal.welch`

```python
frequencies, psd = welch(signal, fs=fs, nperseg=256)
```

- `signal`: segnale nel dominio del tempo.
- `fs`: frequenza di campionamento.
- `nperseg`: lunghezza delle finestre analizzate.
- `frequencies`: frequenze calcolate.
- `psd`: densità spettrale di potenza, cioè energia stimata a ogni frequenza.

In `mean_bandpower_epochs`, una maschera seleziona una banda, ad esempio 8-13 Hz per alpha, e `np.trapezoid` ne calcola la potenza integrata.

# 4. MNE-Python

MNE gestisce i file EEG e gli oggetti `Raw`.

## Lettura e dati

- `mne.io.read_raw_eeglab(file, preload=True)`: legge un file EEGLAB `.set`.
- `raw.get_data()`: restituisce i dati come array `(canali, campioni)`.
- `raw.info["sfreq"]`: legge la frequenza di campionamento.
- `raw.ch_names`: lista dei nomi dei canali.
- `raw.copy()`: crea una copia dell'oggetto Raw.

## Selezione e preprocessing

- `raw.pick(picks=channel_order)`: seleziona e ordina i canali.
- `raw.filter(l_freq, h_freq, fir_design="firwin")`: filtro passa-banda.
- `raw.notch_filter(freqs)`: attenua una frequenza di disturbo, tipicamente 50 Hz.
- `raw.resample(sfreq=target_fs)`: ricampiona il segnale.

Nel progetto il preprocessing principale usa 19 canali, filtro 0.5-40 Hz, notch a 50 Hz e frequenza target 100 Hz, secondo `CONFIG`.

# 5. Statsmodels

Statsmodels fornisce test statistici sulle serie temporali e il test di Granger.

## Test di stazionarietà

- `adfuller(x, maxlag=4)`: test Augmented Dickey-Fuller. Ipotesi nulla: la serie non è stazionaria. Nel progetto passa se `p_value <= alpha`.
- `kpss(x, regression="c", nlags="auto")`: test KPSS. Nel progetto passa se `p_value > alpha`.
- `result[1]`: posizione del p-value nei risultati restituiti come tupla.

`make_epoch_stationary` applica `np.diff(current, axis=1)` e ripete i test fino a `max_diffs` volte.

Al raggiungimento del limite restituisce comunque l'epoca: il codice attuale non scarta le epoche che continuano a fallire i test. Il report dei test restituito dalla funzione non viene incluso nel `quality_report` salvato.

## `grangercausalitytests`

Verifica se la storia di una serie `source` migliora la previsione di una serie `target`.

```python
pair_data = np.column_stack((target, source))
results = grangercausalitytests(pair_data, maxlag=max_lag)
p_value = results[lag][0]["ssr_ftest"][1]
```

Nel grafo viene salvato in `adjacency[target, source]` il rapporto tra lag significativi e lag totali.

# 6. Joblib

Joblib velocizza il calcolo indipendente tra epoche.

- `Parallel(n_jobs=-1)`: esegue lavori in parallelo usando i processori disponibili.
- `delayed(funzione)(argomenti)`: prepara una chiamata da eseguire in seguito.

Schema usato:

```python
matrices = Parallel(n_jobs=-1)(
      delayed(compute_one_epoch)(epoch)
      for epoch in processed_epochs
)
```

`cast(list[np.ndarray], ...)` di `typing` serve solo all'analizzatore dei tipi: non cambia i dati a runtime.

I tre moduli leggono rispettivamente `CONFIG.get("granger_n_jobs", -1)`, `CONFIG.get("lingam_n_jobs", -1)` e `CONFIG.get("pcmci_n_jobs", -1)`: queste chiavi opzionali permettono di limitare i processi.

# 7. LiNGAM

La libreria `lingam` stima relazioni causali usando VARLiNGAM.

- `lingam.VARLiNGAM(lags=..., criterion="bic", prune=True)`: crea il modello.
- `model.fit(data)`: adatta il modello ai dati con forma `(osservazioni, variabili)`.
- `model.adjacency_matrices_`: restituisce le matrici per i diversi lag.
- `np.abs(lag_matrices)`: conserva la forza della relazione ignorando il segno.
- `np.mean(..., axis=0)`: aggrega i lag.

In `lingam_causality.py`, `epoch.T` converte la forma da `(canali, campioni)` a `(campioni, canali)`. La diagonale viene azzerata perché non si vogliono auto-collegamenti.

Con `varlingam_include_instantaneous=False` si esclude la matrice istantanea `B(0)`. Si mediano i valori assoluti dei lag selezionati, poi le matrici delle epoche.

# 8. Tigramite / PCMCI

Tigramite analizza relazioni temporali condizionate tra serie.

- `tigramite.data_processing.DataFrame(data)`: prepara l'array per Tigramite.
- `ParCorr(significance="analytic")`: test di indipendenza basato sulla correlazione parziale.
- `pcmci.PCMCI(dataframe, cond_ind_test=parcorr)`: crea l'analizzatore PCMCI.
- `run_pcmci(tau_max=tau_max, pc_alpha=alpha_level)`: esegue l'analisi.
- `results["p_matrix"]`: p-value delle relazioni.
- `results["val_matrix"]`: valori/intensità delle relazioni.

Il codice considera solo i lag da `1` a `tau_max`, esclude gli auto-collegamenti e conserva le relazioni con p-value inferiore ad `alpha_level`. Media i valori assoluti significativi e converte l'ordine Tigramite `[source, target, tau]` in `adjacency[target, source]`. Le epoche fallite sono escluse dalla media; se falliscono tutte, restituisce una matrice nulla.

# 9. OpenNeuro

Usato nel notebook `code/ds_downloader.ipynb` e importato anche in `preprocessing.py`.

```python
openneuro.download(
      dataset="ds004504",
      target_dir=Path("dataset"),
)
```

- `dataset`: identificativo del dataset OpenNeuro.
- `target_dir`: cartella locale di destinazione.
- `Path(...).mkdir(...)`: crea la destinazione prima del download.
- `target_dir.resolve()`: mostra il percorso assoluto finale.

# 10. Codice locale del progetto

Queste funzioni non appartengono a una libreria esterna: sono state scritte nel progetto.

## `config.py`

- `PROJECT_ROOT = Path(__file__).resolve().parents[1]`: trova la radice del repository.
- `CONFIG`: dizionario condiviso con percorsi, frequenze, soglie, metodi causali e label.
- `CONFIG["labels"]`: mappa `AD -> 1`, `CN -> 0`, `FTD -> 2`.

| Parametro | Valore attuale |
|---|---|
| `derivatives_dir` | `dataset/openneuro_derivatives` |
| `output_dir` | `data_hgcn_ready/openneuro` |
| `target_fs`, `epoch_length_sec` | 100 Hz, 5 secondi (500 campioni prima delle differenze) |
| `low_freq`, `high_freq`, `notch_freq` | 0.5 Hz, 40 Hz, 50 Hz |
| `maxlag`, `alpha`, `max_diffs` | 4 campioni, 0.05, 2 |
| `causality_methods` | `granger`, `lingam`, `pcmci` |
| `varlingam_criterion`, `varlingam_prune` | `bic`, `True` |
| `varlingam_include_instantaneous`, `add_self_loops` | `False`, `False` |
| `excluded_subjects` | OpenNeuro: `sub-086`; BrainLat: `sub-078` |

`overlap=0.0` è presente nella configurazione, ma `split_in_epochs` crea sempre finestre non sovrapposte senza leggere questa chiave. `adjacency_method` non seleziona i metodi di `compute_graphs`: conta `causality_methods`.

## `preprocessing.py`

### Identificazione e caricamento

- `get_subject_label(subject_id)`: legge le ultime tre cifre dell'ID: `001–036 -> AD`, `037–065 -> CN`, `066–088 -> FTD`. Supporta anche gli ID estesi prodotti dall'adattatore BrainLat.
- `get_subjects_list(data_dir)`: elenca le cartelle `sub-*` ordinate.
- `load_subject_raw(subject_dir)`: trova un solo `*_eeg.set` e lo apre con MNE.
- `order_channels(raw, channel_order)`: verifica e ordina i canali.

### Pulizia del segnale

- `filter_and_resample(raw)`: copia, filtra, applica notch e ricampiona.
- `split_in_epochs(eeg, fs, epoch_lenght_sec)`: divide `(canali, campioni)` in `(epoche, canali, campioni)`, scartando il resto incompleto.
- `is_good_epoch(epoch)`: rifiuta NaN, infiniti, ampiezza eccessiva e canali piatti.
- `zscore_epoch(epoch)`: sottrae la media e divide per la deviazione standard di ogni canale.
- `crop_to_same_lenght(epochs)`: taglia tutte le epoche alla lunghezza minima.
- `preprocess_epochs_for_subject(epochs)`: restituisce `(processed_epochs, raw_good_epochs, quality_report)`. `raw_good_epochs` contiene le epoche già standardizzate prima delle differenze, usate per le feature; `processed_epochs` contiene quelle trasformate per i grafi. Se nessuna epoca supera il controllo qualità, restituisce `(None, [], quality_report)`.

### Feature

- `mean_bandpower_epochs(ch_epochs, fs, band)`: media la potenza di una banda tra le epoche.
- `build_node_features(preprocessed_epochs, fs)`: crea una matrice `(canali, 8)` con delta, theta, alpha, beta, gamma, media, deviazione standard e varianza.

Le bande attuali sono delta 0.5–4, theta 4–8, alpha 8–13, beta 13–24 e gamma 24–40 Hz. La pipeline calcola potenze non relative sulle epoche Z-score, prima delle differenze.

## `main.py`

- `get_dataset_key(derivatives_dir)`: riconosce BrainLat se il nome della directory contiene `brainlat`, altrimenti usa OpenNeuro; seleziona così le esclusioni specifiche.
- `compute_graphs(processed_epochs)`: chiama solo i metodi presenti in `CONFIG["causality_methods"]`.
- `main()`: coordina caricamento, preprocessing, feature, grafi e salvataggio.

La prima passata esclude i soggetti configurati e preprocessa tutti gli altri. La seconda usa il minimo numero di epoche valide tra i soggetti, prendendo le **prime `min_epochs`** con `[:min_epochs]` sia per le feature sia per i grafi. Il messaggio di log parla ancora di epoche uniformemente distribuite, ma non corrisponde allo slicing attuale. Gli errori per soggetto vengono riportati senza fermare tutti gli altri.

La chiamata finale:

```python
if __name__ == "__main__":
      main()
```

fa partire la pipeline solo se si esegue direttamente `main.py`.

## Moduli dei grafi causali

- `_compute_granger_epoch(epoch, max_lag, alpha)`: matrice Granger di una singola epoca.
- `compute_granger_adjacency(processed_epochs, max_lag, alpha)`: parallelizza le epoche e le media.
- `_compute_lingam_epoch(epoch, epoch_idx)`: matrice VARLiNGAM di una singola epoca.
- `compute_lingam_adjacency(preprocessed_epochs)`: parallelizza e aggrega VARLiNGAM.
- `_compute_pcmci_epoch(epoch, tau_max, alpha_level)`: matrice PCMCI di una singola epoca.
- `compute_pcmci_adjacency(preprocessed_epochs, tau_max, alpha_level)`: parallelizza, scarta errori e media le matrici valide.

Il prefisso `_` indica funzioni interne, normalmente chiamate da una funzione pubblica del modulo.

## `dataset_module.py`

- `prepare_adjacency_for_hgcn(A, config)`: restituisce `normal` (diretta) e `symmetric` (media delle due direzioni). Entrambe sono divise per il massimo della matrice simmetrica pulita, se positivo; la diretta può quindi superare 1. Pulizia dei valori non finiti, azzeramento diagonale e self-loop opzionali riguardano solo la copia simmetrica.
- `build_subject_graph(...)`: crea il dizionario completo di un soggetto.
- `_json_safe(value)`: converte `Path`, tipi NumPy e array in valori compatibili con JSON.
- `make_subject_splits(graphs, ...)`: divide i soggetti in train, validation e test; usa `stratify` per mantenere le classi bilanciate quando possibile.
- `save_hgcn_ready_dataset(graphs, config)`: salva pickle, JSON, CSV e file `.npy`.

`make_subject_splits` usa di default circa 70% train, 20% validation e 10% test, seed 42, con chiavi JSON `train`, `val`, `test`. Se ci sono meno di 10 grafi o una sola classe, assegna i primi due al train, il terzo alla validation e i restanti al test.

## `brainlat_adapter.py`

- `discover_subjects`: trova i `.set` e ricava le diagnosi dai CSV `records*_eeg_data.csv`, oppure dalla cartella.
- `normalize_label`: converte `HC`/`Control`/`CONTROL` in `CN` e `bvFTD`/`bVFTD` in `FTD`.
- `assign_pipeline_ids`: assegna ID con suffisso compatibile con `get_subject_label`; esaurito un intervallo usa ID estesi, ad esempio `sub-brainlat-cn-044-051`.
- `map_brainlat_channels_to_1020`: per ogni canale canonico sceglie l'elettrodo disponibile più vicino alle coordinate normalizzate del montage `standard_1020`, senza riutilizzarlo.
- `load_and_adapt_raw`: seleziona, rinomina e ordina i 19 canali.
- `convert_subjects`: esporta `sub-<id>/eeg/sub-<id>_task-eyesclosed_eeg.set`; salta i file esistenti salvo `--overwrite`.

La mappatura è basata sulla prossimità delle coordinate. Vedi anche [mappatura degli elettrodi](mappatura_elettrodi_brainlat_openneuro.md).

# 11. Forme degli array

| Variabile | Forma | Significato |
|---|---|---|
| `eeg = raw.get_data()` | `(canali, campioni)` | Segnale continuo. |
| `epochs` | `(epoche, canali, campioni)` | Finestre temporali. |
| `ch_epochs` | `(epoche, campioni)` | Tutte le epoche di un canale. |
| `node_features` / `X` | `(canali, 8)` | Feature dei nodi. |
| `A` | `(canali, canali)` | Matrice di adiacenza del grafo. |
| `p_matrix` | `(canali, canali, lag)` | p-value PCMCI per coppia e ritardo. |
| `samples` dei classificatori | `(soggetti, 494)` | 152 feature dei nodi + 342 archi diretti fuori diagonale. |

Convenzione del grafo: `A[target, source]` indica una relazione dal canale `source` al canale `target`.

# 12. File prodotti

La funzione `save_hgcn_ready_dataset` scrive in `CONFIG["output_dir"]`, attualmente `data_hgcn_ready/openneuro/`; gli export BrainLat sono in `data_hgcn_ready/brainlat/`. Ogni directory contiene:

- `graphs.pkl`: tutti i grafi in un unico file Python binario.
- `config.json`: configurazione serializzata.
- `splits.json`: liste di ID sotto le chiavi `train`, `val`, `test`.
- `summary.csv`: riassunto tabellare dei soggetti.
- `subjects/sub-xxx/X.npy`: feature dei nodi.
- `subjects/sub-xxx/preprocessed_epochs.npy`: epoche preprocessate.
- `subjects/sub-xxx/A_<metodo>_symmetric.npy`: adiacenza simmetrizzata.
- `subjects/sub-xxx/A_<metodo>_normal.npy`: matrice originale normalizzata.
- `subjects/sub-xxx/y.json`: label numerica e testuale.
- `subjects/sub-xxx/metadata.json`: frequenza, dimensioni e report qualità.

# 13. Differenza tra librerie e codice locale

- **Librerie**: forniscono funzioni generiche già implementate, ad esempio `np.mean`, `welch`, `raw.filter` e `train_test_split`.
- **Codice locale**: decide come combinare quelle funzioni per questo progetto, ad esempio `preprocess_epochs_for_subject`, `compute_graphs` e `save_hgcn_ready_dataset`.
- **Configurazione**: `CONFIG` evita di ripetere gli stessi parametri nei vari moduli.

# 14. PyTorch e classificatori baseline

Entrambi gli script allenano un modello distinto per Granger, LiNGAM e PCMCI. Il dataset predefinito è **solo BrainLat**, definito da `DEFAULT_DATASET_DIRS` in `models/mlp.py` e condiviso con la regressione logistica.

## Caricamento, split e normalizzazione

- `load_samples(dataset_dirs, method)`: carica `X.npy`, `A_<method>_normal.npy` e `y.json` dalle cartelle dei soggetti. Concatena `X.ravel()` (19 × 8 = 152 valori) e i 342 archi diretti fuori diagonale: 494 valori per soggetto.
- `validate_sample`: controlla forme `(19, 8)` e `(19, 19)`, valori finiti e label intere 0–2.
- Gli ID dei risultati includono il dataset, ad esempio `brainlat:sub-001`, per distinguere ID uguali tra dataset.
- `stratified_train_val_test`: genera split per classe con proporzioni 70% train, 10% validation, 20% test, arrotondate a conteggi interi.
- `external_test_data`: divide i dataset di training in 85% train e 15% validation e usa tutti i soggetti dei dataset esterni come test. Richiede entrambi gli argomenti train e test; usare dataset distinti, perché la funzione non controlla le sovrapposizioni.
- `make_split_tensors`: calcola media e deviazione standard **solo sul train** e le applica anche a validation e test. Richiede tre split non vuoti.
- `compute_class_weights`: calcola pesi inversamente proporzionali alle frequenze delle classi nel train.

I classificatori **non leggono `splits.json`**: ricreano gli split e caricano tutte le cartelle `subjects/sub-*`. Le esclusioni di `CONFIG` sono applicate dalla pipeline; il loader dei modelli non le riapplica e l'esportazione non elimina eventuali vecchie cartelle di soggetti.

## Modelli e training

| Elemento | Implementazione attuale |
|---|---|
| `SimpleMLP` | `Linear(494, 16) -> ReLU -> Dropout(0.3) -> Linear(16, 3)` |
| `LogisticRegression` | Un solo `Linear(494, 3)` |
| Output | Logit nelle classi `CN`, `AD`, `FTD` (0, 1, 2) |
| Loss di training | `nn.CrossEntropyLoss` con pesi di classe |
| Loss di validation | Cross-entropy senza pesi |
| Ottimizzatore | `torch.optim.AdamW`, training sull'intero split per ogni epoca |
| Early stopping | Loss di validation, ripristino dello stato migliore |
| MLP | Learning rate `1e-3`, weight decay `1e-3` |
| Logistica da CLI | Learning rate `1e-2`, weight decay `1e-3` |
| CLI comuni | Seed 42, massimo 500 epoche, patience 50 |

La chiamata Python diretta a `logistic.train_and_evaluate` ha default diversi dalla CLI: seed 32 e weight decay `1e-4`. I modelli e i tensori rimangono su CPU: gli script non effettuano trasferimenti su GPU.

- `torch.from_numpy`: converte array NumPy in tensori.
- `model.train()` / `model.eval()`: attivano training o valutazione, cambiando anche il comportamento del dropout.
- `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`: azzerano i gradienti, li calcolano e aggiornano i pesi.
- `torch.no_grad()`: disabilita il calcolo dei gradienti durante la valutazione.
- `model.state_dict()` / `model.load_state_dict(...)`: leggono e ripristinano i pesi; lo stato migliore viene copiato in memoria.
- `argmax(dim=1)`: sceglie la classe con logit massimo.

## Risultati

`evaluate_model` calcola con NumPy precision, recall, F1 e matrice di confusione (righe = vere classi; colonne = predizioni, ordine CN/AD/FTD). La balanced accuracy è la media dei recall delle classi presenti nel test.

- MLP: `results/mlp/classifier_results.json`.
- Logistica: `results/logistic/logistic_results.json`.

Per ogni metodo il JSON contiene `balanced_accuracy`, `accuracy`, `training` (`epochs_run`, `best_epoch`, `best_val_loss`), `confusion_matrix`, `classification_report` e gli ID di train/validation/test. I pesi dei modelli non vengono salvati su disco. I percorsi dei risultati sono relativi alla directory di esecuzione e ogni nuova esecuzione sovrascrive il rispettivo JSON.

# 15. Comandi rapidi

Eseguire dalla radice del repository. Il progetto richiede Python `>=3.12,<3.13`; `pyproject.toml` include sorgenti PyTorch/ROCm specifiche per Windows e Python 3.12.

```powershell
# Sincronizza l'ambiente con uv.lock
uv sync

# Controlla i soggetti BrainLat senza esportare file
uv run python code/brainlat_adapter.py --dry-run

# Converte i file da brainlat_dataset/EEG data a dataset/brainlat_derivatives
uv run python code/brainlat_adapter.py

# Pipeline con i percorsi definiti in code/config.py (default: OpenNeuro)
uv run python code/main.py

# MLP sul dataset predefinito (BrainLat)
uv run python models/mlp.py

# Logistica: la directory di output deve già esistere
New-Item -ItemType Directory -Force -Path results/logistic | Out-Null
uv run python models/logistic.py

# MLP su OpenNeuro
uv run python models/mlp.py --dataset-dir data_hgcn_ready/openneuro

# MLP sui due dataset uniti prima dello split
uv run python models/mlp.py --dataset-dir data_hgcn_ready/openneuro --dataset-dir data_hgcn_ready/brainlat

# Training/validation su OpenNeuro, test esterno su BrainLat
uv run python models/mlp.py --train-dataset-dir data_hgcn_ready/openneuro --test-dataset-dir data_hgcn_ready/brainlat

# Iperparametri esposti dalla CLI della logistica
uv run python models/logistic.py --seed 42 --epochs 500 --patience 50 --lr 0.01 --weight-decay 0.001
```

Le opzioni `--dataset-dir`, `--train-dataset-dir` e `--test-dataset-dir` sono ripetibili e disponibili in entrambi i classificatori. `--lr` e `--weight-decay` sono esposte solo dalla CLI logistica.

Per preprocessare BrainLat, impostare in `code/config.py` entrambi i percorsi prima di eseguire `code/main.py`:

```python
"derivatives_dir": PROJECT_ROOT / "dataset" / "brainlat_derivatives",
"output_dir": PROJECT_ROOT / "data_hgcn_ready" / "brainlat",
```

L'adattatore accetta anche `--brainlat-root <percorso>`, `--target <percorso>` e `--overwrite`. Il download OpenNeuro resta nel notebook; `code/ds_download.py` usa invece `synapseclient.Synapse`, `syn.login` e `synapseutils.syncFromSynapse` per BrainLat (entità `syn51549340`).

Il comando del pacchetto `uv run progetto-computer-vision` esegue soltanto il saluto dell'entry point: per elaborare i dati usare `uv run python code/main.py`.
