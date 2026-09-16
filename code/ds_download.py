"""Download the BrainLat dataset from Synapse into a local project folder.

This script is a one-off data acquisition helper. It authenticates with
Synapse, recursively synchronizes the configured entity into
``./brainlat_dataset``, and prints actionable messages for common access
errors such as missing terms-of-use acceptance.
"""

import os
import getpass
import synapseclient
import synapseutils
from synapseclient.core.exceptions import SynapseHTTPError

def main():
    """Authenticate with Synapse and synchronize the configured dataset."""
    # 1. Configurazione: Sostituisci questo con l'ID REALE del dataset su Synapse
    # Esempio: se l'url è https://www.synapse.org/#!Synapse:syn12345678, l'ID è "syn12345678"
    DATASET_SYN_ID = "syn51549340"
    
    # 2. Cartella locale dove verranno salvati i file
    download_dir = "./brainlat_dataset"
    os.makedirs(download_dir, exist_ok=True)
    
    print("=== Inizializzazione Synapse Client ===")
    syn = synapseclient.Synapse()
    
    # 3. Autenticazione sicura (non scrivere mai il token in chiaro nel codice!)
    token = "eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc4OTIxMDI1NywiaWF0IjoxNzg5MjEwMjU3LCJqdGkiOiI0Njg4MyIsInN1YiI6IjM2MTM2OTcifQ.pkC-_yvNjTvDkGuaiL8hwQMZ9_ggDaYc90R6ylNhwpRO1rMM3z2ramWhdo_jB-lEh6rBDYErYi68BQM59KPKLSgyvO_wH1PMU6fy8s13q2lfaM7nrXyDy_ATe7pklM59XIky63O9NMnDztBPTeN7ISAd2HoUpsRjWJyb130Y2z9VK8RyX-12zRlvlk5ZKq4o2VFjW4gkIpiRmfRI7Y6_uW2OIif18hT9g8jhyS6uwuwCP7fOb_jbPDCXZ8Qdpkf2VObw4DTd9mxoNfwnOYZ9-dPTqAnJSik8jPW4dwJUXjHJe9deUYh4jhwU89WNSGouEG4DGsnCT8sJt2o2_cFwtg"
    if not token:
        token = getpass.getpass("Inserisci il tuo Personal Access Token di Synapse (il testo sarà invisibile mentre scrivi): ")
    
    try:
        syn.login(authToken=token, silent=True)
        print("Login effettuato con successo!")
    except Exception as e:
        print(f"\n[ERRORE] Login fallito. Verifica che il token sia corretto. Dettagli: {e}")
        return

    # 4. Avvio del download
    print(f"\nAvvio sincronizzazione del dataset '{DATASET_SYN_ID}'...")
    print(f"I file verranno salvati in: {os.path.abspath(download_dir)}")
    print("Questa operazione potrebbe richiedere del tempo a seconda delle dimensioni del dataset.\n")
    
    try:
        # syncFromSynapse scarica tutto ricorsivamente (file e sottocartelle)
        downloaded_files = synapseutils.syncFromSynapse(
            syn=syn,
            entity=DATASET_SYN_ID,
            path=download_dir
        )
        print(f"\n✅ Download completato con successo! Sono stati scaricati/sincronizzati {len(downloaded_files)} file.")
        
    except SynapseHTTPError as e:
        if "404" in str(e):
            print(f"\n[ERRORE 404] L'ID '{DATASET_SYN_ID}' non esiste. Verifica di aver copiato il synID corretto dal sito web.")
        elif "403" in str(e):
            print(f"\n[ERRORE 403] Accesso negato all'ID '{DATASET_SYN_ID}'.")
            print("Hai accettato i 'Terms of Use' sul sito web di Synapse per questo specifico dataset?")
        else:
            print(f"\n[ERRORE DI RETE] {e}")
    except Exception as e:
        print(f"\n[ERRORE INASPETTATO] {e}")

if __name__ == "__main__":
    main()
