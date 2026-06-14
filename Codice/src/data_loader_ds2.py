"""
Utility di caricamento per DS2 (SCG-RHC Wearable Seismocardiogram dataset).

Questo modulo gestisce il dataset usato come validazione esterna della pipeline
sviluppata sul dataset principale. La presenza di un secondo dataset è
metodologicamente importante perché permette di verificare che preprocessing,
estrazione di feature e analisi restino utili anche su dati raccolti in un
contesto clinico diverso e su pazienti con caratteristiche emodinamiche più
eterogenee.

Nel caso specifico, SCG-RHC include 83 pazienti con insufficienza cardiaca
(HFrEF, HFpEF, OHT, VAD) registrati durante procedura di right heart
catheterization: per questo motivo il loader deve preservare sia i segnali sia
la metainformazione clinica, così da consentire confronti coerenti con il
dataset principale senza appiattire le differenze tra le due coorti.

Riferimento
-----------
Lahiri MK et al., "SCG-RHC Wearable Seismocardiogram Signal and Right Heart
Catheter Database" (PhysioNet 2023).

Funzioni
--------
list_records_ds2()
    Restituisce la lista ordinata di tutti gli ID record in processed_data/.
load_record_ds2(record_id)
    Carica segnali WFDB e metadati JSON di un singolo record.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import wfdb

from .config import ROOT


# ── DS2 paths ──────────────────────────────────────────────────────────────────
DS2_BASE = (
    ROOT
    / "ds2"
    / "scg-rhc-wearable-seismocardiogram-signal-and-right-heart-catheter-database-1.0.0"
)
DS2_DATA = DS2_BASE / "processed_data"
DS2_FS   = 500  # Hz — all channels sampled at 500 Hz


def list_records_ds2() -> list[str]:
    """
    Restituisce la lista ordinata di tutti gli ID record disponibili in DS2.

    L'ordinamento esplicito rende riproducibili le analisi su validazione
    esterna: in una tesi è importante che la sequenza di elaborazione non
    dipenda dal filesystem, così da poter confrontare facilmente risultati e
    tabelle generate in esecuzioni diverse.
    """
    return sorted(p.stem for p in DS2_DATA.glob("*.hea"))


def load_record_ds2(record_id: str) -> dict:
    """
    Carica segnali WFDB e metadati JSON di un record del dataset DS2.

    Questo loader mantiene una struttura di output vicina a quella del dataset
    principale, ma include campi specifici di SCG-RHC perché la validazione
    esterna deve restare comparabile senza ignorare le differenze di origine
    dei dati. In altre parole, l'obiettivo non è forzare DS2 a sembrare uguale
    al primo dataset, bensì verificare se la pipeline generalizza quando cambia
    il contesto clinico e strumentale.

    Restituisce
    -----------
    dict con chiavi:
        record_id    : str
        fs           : int (500)
        ecg          : np.ndarray  — patch_ECG in mV
        acc_lat      : np.ndarray  — patch_ACC_lat in mg (laterale)
        acc_hf       : np.ndarray  — patch_ACC_hf in mg  (head-foot)
        acc_dv       : np.ndarray  — patch_ACC_dv in mg  (dorsal-ventrale)
        meta         : dict        — metadati JSON grezzi
        cdecomp      : int         — 0=compensato, 1=scompensato (-1 se ignoto)
        nyhac        : int | None  — classe NYHA dal JSON
        fine_align   : bool        — True se il record ha allineamento fine
        record       : wfdb.Record — oggetto WFDB completo
    """
    path = str(DS2_DATA / record_id)
    rec = wfdb.rdrecord(path)

    sig_names = rec.sig_name
    signals   = rec.p_signal.astype(np.float32)

    def _channel(name: str) -> Optional[np.ndarray]:
        if name in sig_names:
            return signals[:, sig_names.index(name)]
        return None

    # L'estrazione per nome evita assunzioni sull'ordine dei canali: in un
    # dataset esterno la robustezza del loader conta più della comodità locale.
    ecg     = _channel("patch_ECG")
    acc_lat = _channel("patch_ACC_lat")
    acc_hf  = _channel("patch_ACC_hf")
    acc_dv  = _channel("patch_ACC_dv")

    # I metadati vengono caricati separatamente perché in DS2 contengono
    # informazioni cliniche utili a interpretare le differenze rispetto al
    # dataset principale, non solo dettagli accessori di acquisizione.
    json_path = DS2_DATA / f"{record_id}.json"
    meta: dict = {}
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

    # Manteniamo default espliciti per distinguere il dato davvero assente da
    # una classe clinica valida: nella validazione esterna è preferibile una
    # mancanza dichiarata a un'imputazione silenziosa.
    cdecomp    = int(meta.get("CDecomp", -1))
    nyhac      = meta.get("NYHAC", None)
    fine_align = bool(meta.get("fine_alignment", False))

    return {
        "record_id": record_id,
        "fs":        DS2_FS,
        "ecg":       ecg,
        "acc_lat":   acc_lat,
        "acc_hf":    acc_hf,
        "acc_dv":    acc_dv,
        "meta":      meta,
        "cdecomp":   cdecomp,
        "nyhac":     nyhac,
        "fine_align": fine_align,
        "record":    rec,
    }
