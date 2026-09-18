# Mappatura degli elettrodi BrainLat sugli elettrodi OpenNeuro

Questo documento descrive come il progetto rende confrontabili i segnali EEG del dataset BrainLat con quelli del dataset OpenNeuro. La mappatura non si basa sui nomi originali dei canali BrainLat, ma sulla posizione spaziale degli elettrodi: per ogni elettrodo del montaggio 10-20 usato da OpenNeuro viene scelto l'elettrodo BrainLat geometricamente piu vicino.

## Obiettivo

OpenNeuro usa direttamente 19 elettrodi del sistema internazionale 10-20:

`Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz`

Nel progetto questi 19 canali sono anche l'ordine canonico dei nodi del grafo, definito in `code/config.py` nella chiave `CONFIG["channel_order"]`. Questo e importante perche le matrici di adiacenza e le feature nodali devono avere sempre lo stesso significato: ad esempio, la riga e la colonna associate a `Fp1` devono indicare `Fp1` sia per OpenNeuro sia per BrainLat.

BrainLat, invece, contiene elettrodi con nomi propri del suo sistema di acquisizione, per esempio `C29`, `D4`, `A1`. Prima di inserirlo nella stessa pipeline, il progetto seleziona 19 canali BrainLat e li rinomina con i nomi 10-20 usati da OpenNeuro.

## Dove avviene la mappatura

La mappatura e implementata in `code/brainlat_adapter.py`, soprattutto in due funzioni:

- `load_and_adapt_raw(eeg_path)`: carica un file EEGLAB BrainLat, applica la mappatura, seleziona i canali e li rinomina.
- `map_brainlat_channels_to_1020(raw)`: calcola quale canale BrainLat corrisponde a ciascun canale target 10-20.

Dopo la conversione, i file BrainLat vengono esportati in una struttura compatibile con la pipeline:

`dataset/brainlat_derivatives/sub-XXX/eeg/sub-XXX_task-eyesclosed_eeg.set`

Da quel momento, la pipeline principale puo trattare BrainLat come OpenNeuro, perche i canali hanno gli stessi nomi e lo stesso ordine.

## Algoritmo usato

Per ogni soggetto BrainLat il codice segue questi passaggi:

1. Carica il file `.set` con MNE.
2. Carica il montaggio standard `standard_1020` di MNE.
3. Per ogni canale target nell'ordine canonico, recupera la posizione 3D dell'elettrodo 10-20.
4. Normalizza sia la posizione target sia le posizioni BrainLat come vettori unitari. In questo modo la scelta dipende dalla direzione sulla testa, non dalla scala assoluta delle coordinate.
5. Calcola la distanza euclidea tra il target 10-20 e ogni elettrodo BrainLat ancora non usato.
6. Sceglie l'elettrodo BrainLat con distanza minima.
7. Marca quell'elettrodo come gia usato, cosi lo stesso canale BrainLat non puo essere assegnato a due target diversi.
8. Rinomina il canale BrainLat scelto con il nome 10-20 corrispondente.
9. Riordina il segnale finale secondo `CONFIG["channel_order"]`.

In forma sintetica:

`target OpenNeuro/10-20 -> elettrodo BrainLat piu vicino -> rinomina al nome target`

Quindi BrainLat non viene "interpolato" su OpenNeuro: viene selezionato, per ogni posizione 10-20 richiesta, il sensore BrainLat piu vicino disponibile.

## Tabella della mappatura osservata

La seguente tabella e stata ricavata dai file BrainLat presenti nel progetto. Sul dataset locale sono stati trovati 162 file `.set`; 134 sono stati letti e mappati correttamente, mentre 28 non sono stati inclusi nel conteggio delle distanze perche non hanno superato la lettura/mappatura automatica usata nello script di verifica. La mappatura e stabile per quasi tutti i canali; l'unica eccezione osservata e `P3`, che viene assegnato a `A18` in 71 soggetti e a `A7` in 63 soggetti.

| Elettrodo OpenNeuro / 10-20 | Elettrodo BrainLat sorgente |
| --- | --- |
| Fp1 | C29 |
| Fp2 | C16 |
| F3 | D4 |
| F4 | C4 |
| C3 | D19 |
| C4 | B22 |
| P3 | A18 / A7 |
| P4 | B4 |
| O1 | A16 |
| O2 | A29 |
| F7 | D7 |
| F8 | C7 |
| T3 | D24 |
| T4 | B14 |
| T5 | A10 |
| T6 | B7 |
| Fz | C21 |
| Cz | A1 |
| Pz | A19 |

Esempio concreto: il canale BrainLat `C29` viene selezionato come il piu vicino alla posizione standard `Fp1`; dopo la conversione, nel file esportato quel canale si chiama `Fp1`.

## Quantificazione delle distanze

Le distanze sono state calcolate nello stesso modo usato da `code/brainlat_adapter.py`: le coordinate 3D del montaggio standard 10-20 e quelle degli elettrodi BrainLat sono state normalizzate come vettori unitari, poi e stata misurata la distanza euclidea tra i due vettori. Questa distanza e quindi una distanza normalizzata sulla sfera unitaria, non una distanza fisica in centimetri.

Per rendere il valore piu leggibile, nella tabella e riportata anche la distanza angolare in gradi:

| Elettrodo 10-20 | BrainLat | Distanza media normalizzata | Distanza angolare media |
| --- | --- | ---: | ---: |
| Fp1 | C29 | 0.049312 | 2.826 deg |
| Fp2 | C16 | 0.049831 | 2.855 deg |
| F3 | D4 | 0.049822 | 2.855 deg |
| F4 | C4 | 0.072094 | 4.132 deg |
| C3 | D19 | 0.126426 | 7.248 deg |
| C4 | B22 | 0.117684 | 6.747 deg |
| P3 | A18 / A7 | 0.167028 | 9.581 deg |
| P4 | B4 | 0.148346 | 8.507 deg |
| O1 | A16 | 0.106872 | 6.126 deg |
| O2 | A29 | 0.104859 | 6.011 deg |
| F7 | D7 | 0.133194 | 7.637 deg |
| F8 | C7 | 0.132510 | 7.598 deg |
| T3 | D24 | 0.145629 | 8.351 deg |
| T4 | B14 | 0.157560 | 9.037 deg |
| T5 | A10 | 0.150279 | 8.619 deg |
| T6 | B7 | 0.157101 | 9.010 deg |
| Fz | C21 | 0.081035 | 4.644 deg |
| Cz | A1 | 0.091248 | 5.230 deg |
| Pz | A19 | 0.026761 | 1.533 deg |

Statistiche complessive sui 134 soggetti mappati:

- distanza media normalizzata: `0.108821`
- distanza minima normalizzata: `0.026761`, osservata per `Pz -> A19`
- distanza massima normalizzata: `0.168070`, osservata nella variante di `P3`
- distanza angolare media: `6.239 deg`
- distanza angolare minima: `1.533 deg`, osservata per `Pz -> A19`
- distanza angolare massima: `9.641 deg`, osservata nella variante di `P3`

Interpretazione: gli elettrodi BrainLat selezionati sono generalmente entro circa 1.5-9.6 gradi dalla posizione 10-20 target. I canali piu vicini sono `Pz`, `Fp1`, `Fp2` e `F3`; quelli con discrepanza maggiore sono nell'area parietale/temporale, in particolare `P3`, `T4`, `T6`, `T5` e `P4`.

## Effetto sulla pipeline

OpenNeuro entra nella pipeline con i nomi 10-20 gia presenti. BrainLat viene prima adattato, poi passa alla stessa funzione di ordinamento usata per tutti i dati. Il risultato e che entrambi i dataset producono tensori e grafi con forma coerente:

- feature nodali: `19 x 8`
- matrici di adiacenza: `19 x 19`
- ordine dei nodi: sempre `Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz`

Questo permette di addestrare e valutare modelli sugli stessi indici di canale. Per esempio, l'indice associato a `Cz` rappresenta sempre `Cz`, indipendentemente dal fatto che il soggetto provenga da OpenNeuro o da BrainLat.

## Nota metodologica

La mappatura e approssimata geometricamente: assume che il canale BrainLat piu vicino alla posizione 10-20 sia il miglior equivalente disponibile. Questa scelta e ragionevole per armonizzare dataset con montaggi diversi, ma va documentata perche eventuali differenze residue di posizione degli elettrodi possono introdurre piccole differenze spaziali tra i dataset.
