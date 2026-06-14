"""
Utility per il caricamento dei dati dal dataset WFDB (formato PhysioNet).

Il dataset è organizzato in coppie di file ECG + ACC per ogni paziente e sessione.
Questo modulo astrae completamente l'accesso al filesystem: il resto del codice
non deve conoscere la struttura delle directory né il formato WFDB.

Il formato WFDB (Waveform Database) è lo standard open-source di PhysioNet per
la distribuzione di segnali biomedici: ogni record è composto da un file header
(.hea) con i metadati e un file dati (.dat) con il segnale campionato.
Le annotazioni (.atr) contengono gli onset dei test riabilitativi.
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import wfdb

from .config import (
    ACC_DIR, ECG_DIR, SUBJECT_INFO_CSV, TEST_AVAIL_CSV,
    ECG_FS, ACC_FS, TEST_LABELS,
)


# ── Funzioni interne ───────────────────────────────────────────────────────────

def _record_path(base_dir: Path, patient_id: str, session: int, suffix: str) -> str:
    """Costruisce il percorso base del record WFDB (senza estensione)."""
    name = f"{patient_id}_{session}_{suffix}"
    return str(base_dir / name)


def _parse_patient_ids() -> list[str]:
    """
    Ricava la lista dei pazienti dalla directory ECG tramite pattern matching.
    Non dipende da un file indice: robusto in caso di dataset incompleti.
    """
    ids = set()
    for p in ECG_DIR.glob("*_ecg.hea"):
        m = re.match(r"^(\d+)_\d+_ecg\.hea$", p.name)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


# ── API pubblica ───────────────────────────────────────────────────────────────

def list_patients() -> list[str]:
    """Restituisce la lista ordinata di tutti i pazienti nel dataset."""
    return _parse_patient_ids()


def get_patient_sessions(patient_id: str) -> list[int]:
    """Restituisce gli indici di sessione disponibili per un paziente."""
    sessions = set()
    for p in ECG_DIR.glob(f"{patient_id}_*_ecg.hea"):
        m = re.match(rf"^{patient_id}_(\d+)_ecg\.hea$", p.name)
        if m:
            sessions.add(int(m.group(1)))
    return sorted(sessions)


def load_ecg_record(patient_id: str, session: int = 1) -> dict:
    """
    Carica il record ECG (segnale + annotazioni) per una sessione.

    Il segnale è in millivolt (mV), campionato a 130 Hz. Le annotazioni
    contengono gli onset dei test riabilitativi in campioni (es. 'STAIR', '6MWT').
    La prima colonna del segnale multiderivazione è estratta: nel dataset
    è presente una sola derivazione ECG (monoderivazione).

    Restituisce un dizionario con:
        signal      : np.ndarray (n_campioni,) — ECG in mV
        fs          : int — frequenza di campionamento (130 Hz)
        ann_samples : np.ndarray — indici campione delle annotazioni
        ann_labels  : list[str] — etichette delle annotazioni
        record      : wfdb.Record — oggetto con tutti i metadati
    """
    path = _record_path(ECG_DIR, patient_id, session, "ecg")
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    return {
        "signal": rec.p_signal[:, 0].astype(np.float32),  # prima (e unica) derivazione
        "fs": rec.fs,
        "ann_samples": ann.sample,
        "ann_labels": ann.aux_note,
        "record": rec,
    }


def load_acc_record(patient_id: str, session: int = 1) -> dict:
    """
    Carica il record accelerometrico (segnale triassiale + annotazioni).

    Il sensore è un MEMS triassiale, campionato a 200 Hz, indossato sul torace.
    Le tre assi (X, Y, Z) sono restituite separatamente perché il preprocessing
    rimuove la gravità per asse prima di calcolare la magnitudine vettoriale.

    Restituisce un dizionario con:
        signal_x, signal_y, signal_z : np.ndarray — accelerazione in g per asse
        fs          : int — frequenza di campionamento (200 Hz)
        ann_samples : np.ndarray
        ann_labels  : list[str]
        record      : wfdb.Record
    """
    path = _record_path(ACC_DIR, patient_id, session, "acc")
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    sig = rec.p_signal.astype(np.float32)
    return {
        "signal_x": sig[:, 0],
        "signal_y": sig[:, 1],
        "signal_z": sig[:, 2],
        "fs": rec.fs,
        "ann_samples": ann.sample,
        "ann_labels": ann.aux_note,
        "record": rec,
    }


def load_subject_info() -> pd.DataFrame:
    """
    Carica il file subject-info.csv con le informazioni cliniche e demografiche.

    Il CSV ha un'intestazione a due livelli (categoria + sottocategoria):
    viene appiattita con il separatore '|'. Il BMI è ricalcolato da peso e
    altezza se le colonne corrispondenti sono presenti, per evitare dipendenza
    da un campo precalcolato che potrebbe mancare in soggetti con dati incompleti.
    I separatori decimali a virgola (formato italiano) sono normalizzati a punto.
    """
    df = pd.read_csv(SUBJECT_INFO_CSV, header=[0, 1], encoding="utf-8-sig")
    # Appiattimento intestazione a due livelli
    df.columns = [
        f"{a.strip()}|{b.strip()}" if not b.startswith("Unnamed") else a.strip()
        for a, b in df.columns
    ]
    df = df.rename(columns={df.columns[0]: "patient_id"})
    df["patient_id"] = df["patient_id"].astype(str).str.zfill(3)
    df = df.set_index("patient_id")

    # Normalizzazione separatore decimale (il CSV originale usa la virgola)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(",", ".", regex=False)

    # Calcolo BMI da altezza e peso se disponibili
    height_col = next((c for c in df.columns if "Height" in c), None)
    weight_col = next((c for c in df.columns if "Weight" in c), None)
    if height_col and weight_col:
        h = pd.to_numeric(df[height_col], errors="coerce")
        w = pd.to_numeric(df[weight_col], errors="coerce")
        df["BMI"] = w / (h / 100) ** 2   # BMI = peso(kg) / altezza(m)²

    return df


def load_test_availability() -> pd.DataFrame:
    """
    Carica il file test-availability.csv.

    Indica quali test sono stati eseguiti da ciascun paziente. I valori '-'
    (test non disponibile) sono sostituiti con NaN per compatibilità con pandas.
    """
    df = pd.read_csv(TEST_AVAIL_CSV, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["Patient ID"] = df["Patient ID"].astype(str).str.zfill(3)
    df = df.set_index("Patient ID")
    df = df.replace("-", np.nan)
    return df


def load_patient(patient_id: str, session: int = 1) -> dict:
    """
    Carica e restituisce ECG + ACC allineati per una sessione paziente.

    I due segnali hanno frequenze di campionamento diverse (ECG 130 Hz, ACC 200 Hz)
    ma partono dallo stesso istante di clock: l'allineamento temporale è gestito
    convertendo gli indici campione in secondi durante l'estrazione dei segmenti.
    """
    ecg = load_ecg_record(patient_id, session)
    acc = load_acc_record(patient_id, session)
    return {"patient_id": patient_id, "session": session, "ecg": ecg, "acc": acc}


def get_test_segments(
    ecg_ann_samples: np.ndarray,
    ecg_ann_labels: list,
    acc_ann_samples: np.ndarray,
    acc_ann_labels: list,
    ecg_sig_len: int,
    acc_sig_len: int,
    include_rest: bool = True,
    max_rest_sec: float = 300.0,
) -> dict:
    """
    Costruisce un dizionario che mappa ogni etichetta di test ai suoi intervalli
    campione (inizio/fine) sia nella timeline ECG che in quella ACC.

    La fine del test N coincide con l'inizio del test N+1 (o con la fine del
    segnale per l'ultimo test): le annotazioni delimitano solo gli onset.

    Se include_rest=True, aggiunge un segmento 'REST' corrispondente ai
    max_rest_sec secondi immediatamente precedenti il primo test. Questo
    segmento è usato come classe negativa per OBJ2 (REST vs ACTIVITY).

    Restituisce un dizionario: label → {ecg_start, ecg_end, acc_start, acc_end}
    """
    # Estrazione e ordinamento degli onset per ECG e ACC separatamente
    ecg_onsets = sorted(
        [(s, lbl) for s, lbl in zip(ecg_ann_samples, ecg_ann_labels)
         if lbl in TEST_LABELS],
        key=lambda x: x[0],
    )
    acc_onsets = sorted(
        [(s, lbl) for s, lbl in zip(acc_ann_samples, acc_ann_labels)
         if lbl in TEST_LABELS],
        key=lambda x: x[0],
    )

    # Dizionario label → campione di inizio
    ecg_start_by_label = {lbl: samp for samp, lbl in ecg_onsets}
    acc_start_by_label = {lbl: samp for samp, lbl in acc_onsets}

    def _end_samples(onsets, sig_len):
        """Assegna la fine di ogni test: inizio del successivo o fine segnale."""
        ends = {}
        for i, (samp, lbl) in enumerate(onsets):
            end = onsets[i + 1][0] if i + 1 < len(onsets) else sig_len
            ends[lbl] = end
        return ends

    ecg_end = _end_samples(ecg_onsets, ecg_sig_len)
    acc_end = _end_samples(acc_onsets, acc_sig_len)

    # Solo i test presenti in entrambe le modalità (ECG e ACC)
    segments = {}
    common_labels = set(ecg_start_by_label) & set(acc_start_by_label)
    for lbl in common_labels:
        segments[lbl] = {
            "ecg_start": int(ecg_start_by_label[lbl]),
            "ecg_end": int(ecg_end[lbl]),
            "acc_start": int(acc_start_by_label[lbl]),
            "acc_end": int(acc_end[lbl]),
        }

    # Segmento REST: finestra prima del primo test, limitata a max_rest_sec
    if include_rest and ecg_onsets and acc_onsets:
        first_ecg = ecg_onsets[0][0]
        first_acc = acc_onsets[0][0]
        rest_ecg_end = first_ecg
        rest_acc_end = first_acc
        # Tronca a max_rest_sec per evitare segmenti troppo lunghi e disomogenei
        rest_ecg_start = max(0, rest_ecg_end - int(max_rest_sec * ECG_FS))
        rest_acc_start = max(0, rest_acc_end - int(max_rest_sec * ACC_FS))
        if (rest_ecg_end - rest_ecg_start) > 0:
            segments["REST"] = {
                "ecg_start": rest_ecg_start,
                "ecg_end": rest_ecg_end,
                "acc_start": rest_acc_start,
                "acc_end": rest_acc_end,
            }

    return segments

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import wfdb

from .config import (
    ACC_DIR, ECG_DIR, SUBJECT_INFO_CSV, TEST_AVAIL_CSV,
    ECG_FS, ACC_FS, TEST_LABELS,
)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _record_path(base_dir: Path, patient_id: str, session: int, suffix: str) -> str:
    """Return the WFDB record base path (no extension)."""
    name = f"{patient_id}_{session}_{suffix}"
    return str(base_dir / name)


def _parse_patient_ids() -> list[str]:
    """Infer the list of patient IDs from the ECG directory."""
    ids = set()
    for p in ECG_DIR.glob("*_ecg.hea"):
        m = re.match(r"^(\d+)_\d+_ecg\.hea$", p.name)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


# ── Public API ─────────────────────────────────────────────────────────────────

def list_patients() -> list[str]:
    """Return sorted list of all patient IDs in the dataset."""
    return _parse_patient_ids()


def get_patient_sessions(patient_id: str) -> list[int]:
    """Return the sorted list of session indices available for *patient_id*."""
    sessions = set()
    for p in ECG_DIR.glob(f"{patient_id}_*_ecg.hea"):
        m = re.match(rf"^{patient_id}_(\d+)_ecg\.hea$", p.name)
        if m:
            sessions.add(int(m.group(1)))
    return sorted(sessions)


def load_ecg_record(patient_id: str, session: int = 1) -> dict:
    """
    Load the ECG record (signal + annotations) for one session.

    Returns
    -------
    dict with keys:
        signal      : np.ndarray, shape (n_samples,)  — ECG in mV
        fs          : int  — sampling frequency (130 Hz)
        ann_samples : np.ndarray  — annotation sample indices
        ann_labels  : list[str]   — annotation labels (e.g. 'STAIR', '6MWT')
        record      : wfdb.Record object (full metadata)
    """
    path = _record_path(ECG_DIR, patient_id, session, "ecg")
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    return {
        "signal": rec.p_signal[:, 0].astype(np.float32),
        "fs": rec.fs,
        "ann_samples": ann.sample,
        "ann_labels": ann.aux_note,
        "record": rec,
    }


def load_acc_record(patient_id: str, session: int = 1) -> dict:
    """
    Load the accelerometer record (signal + annotations) for one session.

    Returns
    -------
    dict with keys:
        signal_x, signal_y, signal_z : np.ndarray — per-axis acceleration in g
        fs          : int  — sampling frequency (200 Hz)
        ann_samples : np.ndarray
        ann_labels  : list[str]
        record      : wfdb.Record object
    """
    path = _record_path(ACC_DIR, patient_id, session, "acc")
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    sig = rec.p_signal.astype(np.float32)
    return {
        "signal_x": sig[:, 0],
        "signal_y": sig[:, 1],
        "signal_z": sig[:, 2],
        "fs": rec.fs,
        "ann_samples": ann.sample,
        "ann_labels": ann.aux_note,
        "record": rec,
    }


def load_subject_info() -> pd.DataFrame:
    """
    Load subject-info.csv.

    Cleans column names and adds a computed BMI column.
    Returns a DataFrame indexed by Patient ID.
    """
    df = pd.read_csv(SUBJECT_INFO_CSV, header=[0, 1], encoding="utf-8-sig")
    # Flatten multi-level header
    df.columns = [
        f"{a.strip()}|{b.strip()}" if not b.startswith("Unnamed") else a.strip()
        for a, b in df.columns
    ]
    # Rename the patient-id column (first column)
    df = df.rename(columns={df.columns[0]: "patient_id"})
    df["patient_id"] = df["patient_id"].astype(str).str.zfill(3)
    df = df.set_index("patient_id")

    # Normalise decimal separator (some cells use comma instead of dot)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(",", ".", regex=False)

    # Try to compute BMI if height/weight columns exist
    height_col = next((c for c in df.columns if "Height" in c), None)
    weight_col = next((c for c in df.columns if "Weight" in c), None)
    if height_col and weight_col:
        h = pd.to_numeric(df[height_col], errors="coerce")
        w = pd.to_numeric(df[weight_col], errors="coerce")
        df["BMI"] = w / (h / 100) ** 2

    return df


def load_test_availability() -> pd.DataFrame:
    """
    Load test-availability.csv.

    Returns a DataFrame indexed by Patient ID, with '-' replaced by NaN.
    """
    df = pd.read_csv(TEST_AVAIL_CSV, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["Patient ID"] = df["Patient ID"].astype(str).str.zfill(3)
    df = df.set_index("Patient ID")
    df = df.replace("-", np.nan)
    return df


def load_patient(patient_id: str, session: int = 1) -> dict:
    """
    Load and temporally align ECG + ACC signals for one session.

    The two signals start at the same clock time (same header timestamp) so
    we just return them with their respective sample indices. Alignment is
    handled in preprocessing by converting sample indices to seconds.

    Returns
    -------
    dict with keys 'ecg' and 'acc', each a sub-dict as returned by
    load_ecg_record / load_acc_record, plus 'patient_id' and 'session'.
    """
    ecg = load_ecg_record(patient_id, session)
    acc = load_acc_record(patient_id, session)
    return {"patient_id": patient_id, "session": session, "ecg": ecg, "acc": acc}


def get_test_segments(
    ecg_ann_samples: np.ndarray,
    ecg_ann_labels: list,
    acc_ann_samples: np.ndarray,
    acc_ann_labels: list,
    ecg_sig_len: int,
    acc_sig_len: int,
    include_rest: bool = True,
    max_rest_sec: float = 300.0,
) -> dict:
    """
    Build a dict mapping each test label to its (start_ecg, end_ecg,
    start_acc, end_acc) sample indices.

    The 'end' of test N is the 'start' of test N+1 (or end-of-signal for
    the last test). Segments are sorted by onset time.

    If *include_rest* is True, a "REST" entry is added covering up to
    *max_rest_sec* seconds immediately before the first test onset.

    Returns
    -------
    dict[str, dict]  keyed by test label, value has:
        ecg_start, ecg_end, acc_start, acc_end : int
    """
    # ── ECG segments ──────────────────────────────────────────────────────────
    ecg_onsets = sorted(
        [(s, lbl) for s, lbl in zip(ecg_ann_samples, ecg_ann_labels)
         if lbl in TEST_LABELS],
        key=lambda x: x[0],
    )
    # ── ACC segments ──────────────────────────────────────────────────────────
    acc_onsets = sorted(
        [(s, lbl) for s, lbl in zip(acc_ann_samples, acc_ann_labels)
         if lbl in TEST_LABELS],
        key=lambda x: x[0],
    )

    # Build lookup: label → start sample for both modalities
    ecg_start_by_label = {lbl: samp for samp, lbl in ecg_onsets}
    acc_start_by_label = {lbl: samp for samp, lbl in acc_onsets}

    # Assign end samples: next test onset or end-of-signal
    def _end_samples(onsets, sig_len):
        ends = {}
        for i, (samp, lbl) in enumerate(onsets):
            end = onsets[i + 1][0] if i + 1 < len(onsets) else sig_len
            ends[lbl] = end
        return ends

    ecg_end = _end_samples(ecg_onsets, ecg_sig_len)
    acc_end = _end_samples(acc_onsets, acc_sig_len)

    segments = {}
    common_labels = set(ecg_start_by_label) & set(acc_start_by_label)
    for lbl in common_labels:
        segments[lbl] = {
            "ecg_start": int(ecg_start_by_label[lbl]),
            "ecg_end": int(ecg_end[lbl]),
            "acc_start": int(acc_start_by_label[lbl]),
            "acc_end": int(acc_end[lbl]),
        }

    # ── REST segment (pre-test) ────────────────────────────────────────────
    if include_rest and ecg_onsets and acc_onsets:
        first_ecg = ecg_onsets[0][0]
        first_acc = acc_onsets[0][0]
        rest_ecg_end = first_ecg
        rest_acc_end = first_acc
        # Cap to max_rest_sec before first test
        rest_ecg_start = max(0, rest_ecg_end - int(max_rest_sec * ECG_FS))
        rest_acc_start = max(0, rest_acc_end - int(max_rest_sec * ACC_FS))
        if (rest_ecg_end - rest_ecg_start) > 0:
            segments["REST"] = {
                "ecg_start": rest_ecg_start,
                "ecg_end": rest_ecg_end,
                "acc_start": rest_acc_start,
                "acc_end": rest_acc_end,
            }

    return segments
