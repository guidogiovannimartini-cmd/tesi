"""
Signal Quality Index (SQI) per la valutazione di qualità di ECG e ACC.

In un dataset fisiologico reale non basta estrarre feature: prima bisogna
capire se la finestra osservata è abbastanza affidabile da rappresentare il
fenomeno biologico e non, invece, problemi di contatto, saturazione o
movimento eccessivo. Questo modulo raccoglie indici semplici ma interpretabili,
pensati per scartare porzioni di segnale in cui la qualità comprometterebbe
analisi successive come HRV, fusione multimodale o classificazione.

Funzioni
--------
ecg_sqi(filtered, rpeaks_raw, rpeaks_clean, rr_ms, fs)
    Calcola metriche di qualità ECG su una singola finestra.
acc_sqi(x, y, z, magnitude)
    Calcola metriche di qualità ACC su una singola finestra.
is_window_valid(eq, aq, thresholds)
    Restituisce True solo se ECG e ACC superano entrambi i controlli.
quality_report(df)
    Riassume le statistiche di scarto su un DataFrame di feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RR_MIN_MS, RR_MAX_MS, ECG_FS, ACC_FS

# ── Default quality thresholds ─────────────────────────────────────────────────

DEFAULT_ECG_THRESHOLDS = {
    # Un tratto troppo piatto suggerisce distacco elettrodo o segnale quasi
    # assente: in ECG un cuore vivo non produce lunghi segmenti realmente
    # costanti, quindi questa soglia è un buon indicatore di perdita di contatto.
    "flatline_ratio_max": 0.05,
    # Il clipping è indice di saturazione elettronica: quando l'ADC "taglia" il
    # segnale, la morfologia QRS non è più affidabile per tempi e ampiezze.
    "clip_ratio_max": 0.02,
    # Se pochi picchi sopravvivono alla pulizia, significa che il rilevamento R
    # è fragile in quella finestra; usarla per HRV porterebbe stime fuorvianti.
    "rr_quality_ratio_min": 0.50,
    # Una variabilità relativa eccessiva degli R-R, in questo contesto, è più
    # spesso sintomo di rumore o miss-detection che di fisiologia reale.
    "cv_rr_max": 0.40,
    # Impongo un minimo di picchi puliti perché metriche derivate dagli R-R con
    # troppo pochi battiti sarebbero statisticamente poco stabili.
    "min_rpeaks": 5,
}

DEFAULT_ACC_THRESHOLDS = {
    # Anche per l'ACC il clipping compromette la leggibilità del gesto, perché
    # le accelerazioni più intense vengono "schiacciate" al valore massimo.
    "clip_ratio_max": 0.02,
    # Una magnitudine quasi piatta è compatibile con sensore fermo, spento o
    # scollegato: più che quiete fisiologica, qui interessa rilevare inerzia non plausibile.
    "flatline_ratio_max": 0.10,
    # Se la dinamica è troppo piccola, il sensore potrebbe essere inattivo o la
    # componente utile essere stata persa; 5 mg è una soglia minima prudenziale.
    "min_magnitude_range": 0.005,
    # Una deviazione standard enorme della magnitudine è spesso compatibile con
    # artefatti grossolani o urti, non con movimento fisiologico analizzabile.
    "max_magnitude_std": 10.0,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _flatline_ratio(signal: np.ndarray, eps: float = 1e-6, run_len: int = 10) -> float:
    """
    Stima la frazione di campioni appartenenti a tratti quasi costanti.

    Il criterio sui run consecutivi è preferibile a un semplice controllo della
    varianza globale, perché individua bene i segmenti "morti" anche quando il
    resto della finestra contiene attività normale. In pratica è utile per
    intercettare disconnessioni brevi ma metodologicamente rilevanti.
    """
    if len(signal) < run_len:
        return 0.0
    diff = np.abs(np.diff(signal.astype(np.float64)))
    flat = np.zeros(len(signal), dtype=bool)
    i = 0
    while i < len(diff):
        if diff[i] < eps:
            j = i
            while j < len(diff) and diff[j] < eps:
                j += 1
            if (j - i + 1) >= run_len:
                flat[i: j + 1] = True
            i = j + 1
        else:
            i += 1
    return float(flat.sum()) / len(signal)


def _clip_ratio(signal: np.ndarray, percentile_margin: float = 0.1) -> float:
    """
    Stima la frazione di campioni prossimi al massimo assoluto.

    Non cerco il clipping con una soglia fissa, perché l'ampiezza dipende dal
    sensore e dal preprocessing. Lavorare in percentuale rispetto al massimo
    osservato rende il criterio più portabile e più adatto a dati eterogenei.
    """
    amax = np.max(np.abs(signal))
    if amax == 0:
        return 0.0
    threshold = amax * (1.0 - percentile_margin / 100.0)
    return float(np.sum(np.abs(signal) >= threshold)) / len(signal)


# ── ECG SQI ───────────────────────────────────────────────────────────────────

def ecg_sqi(
    filtered: np.ndarray,
    rpeaks_raw: np.ndarray,
    rpeaks_clean: np.ndarray,
    rr_ms: np.ndarray,
    fs: float = ECG_FS,
) -> dict:
    """
    Calcola metriche SQI ECG per una singola finestra.

    Le metriche scelte cercano di coprire problemi diversi e complementari:
    continuità del segnale, saturazione, affidabilità del rilevamento dei
    battiti e plausibilità della sequenza R-R. Insieme permettono di evitare
    che una finestra apparentemente "filtrata bene" venga considerata valida
    anche quando la struttura cardiaca utile è stata persa.

    Parametri
    ---------
    filtered
        Traccia ECG filtrata nella finestra.
    rpeaks_raw
        Tutti i picchi R rilevati prima della pulizia.
    rpeaks_clean
        Picchi R sopravvissuti alla rimozione degli outlier.
    rr_ms
        Intervalli R-R puliti in millisecondi.
    fs
        Frequenza di campionamento.

    Restituisce
    -----------
    Un dizionario con metriche scalari di qualità e flag booleano `ecg_valid`.
    """
    q: dict = {}

    # Un'elevata quota di flatline suggerisce perdita di informazione elettrica
    # più che vero riposo cardiaco: il cuore continua comunque a generare QRS.
    q["ecg_flatline_ratio"] = _flatline_ratio(filtered)

    # Il clipping altera le ampiezze e può deformare il picco R, rendendo meno
    # affidabile anche la fase di peak detection.
    q["ecg_clip_ratio"] = _clip_ratio(filtered)

    # Confrontare picchi grezzi e picchi puliti è un modo semplice per stimare
    # quanto il detector stia lavorando in un contesto rumoroso.
    n_raw = len(rpeaks_raw)
    n_clean = len(rpeaks_clean)
    q["ecg_rr_quality_ratio"] = float(n_clean / n_raw) if n_raw > 0 else 0.0
    q["ecg_n_rpeaks_raw"] = int(n_raw)
    q["ecg_n_rpeaks_clean"] = int(n_clean)

    # Il coefficiente di variazione normalizza la dispersione rispetto alla
    # media: così la soglia resta interpretabile anche al variare della FC.
    if len(rr_ms) >= 2:
        mean_rr = float(np.mean(rr_ms))
        std_rr = float(np.std(rr_ms, ddof=1))
        q["ecg_cv_rr"] = std_rr / mean_rr if mean_rr > 0 else np.nan
    else:
        q["ecg_cv_rr"] = np.nan

    # Il rapporto di potenza a 50/60 Hz è utile perché l'ECG è molto sensibile
    # al rumore di rete: se quella banda pesa troppo, la finestra rischia di
    # essere dominata da interferenza elettromagnetica più che da fisiologia.
    if len(filtered) >= int(fs * 2):
        from scipy.signal import welch
        freqs, psd = welch(filtered, fs=fs, nperseg=min(512, len(filtered)))
        total_power = float(np.sum(psd)) + 1e-12
        # Considero entrambe le frequenze di rete per mantenere il criterio
        # riusabile in setup diversi o dataset provenienti da ambienti differenti.
        pl_mask = (
            ((freqs >= 48) & (freqs <= 52)) |
            ((freqs >= 58) & (freqs <= 62))
        )
        q["ecg_powerline_ratio"] = float(np.sum(psd[pl_mask])) / total_power
    else:
        q["ecg_powerline_ratio"] = np.nan

    # Il flag finale combina controlli diversi perché nessuna singola metrica,
    # da sola, basta a rappresentare la qualità complessiva di una finestra ECG.
    t = DEFAULT_ECG_THRESHOLDS
    q["ecg_valid"] = (
        q["ecg_flatline_ratio"] <= t["flatline_ratio_max"] and
        q["ecg_clip_ratio"] <= t["clip_ratio_max"] and
        q["ecg_rr_quality_ratio"] >= t["rr_quality_ratio_min"] and
        n_clean >= t["min_rpeaks"] and
        (np.isnan(q["ecg_cv_rr"]) or q["ecg_cv_rr"] <= t["cv_rr_max"])
    )

    return q


# ── ACC SQI ───────────────────────────────────────────────────────────────────

def acc_sqi(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    magnitude: np.ndarray,
) -> dict:
    """
    Calcola metriche SQI per l'accelerometro su una singola finestra.

    L'obiettivo qui non è stimare "quanto si muove" il soggetto in assoluto,
    ma distinguere movimento interpretabile da comportamento strumentale
    anomalo. Per questo vengono considerate saturazione, tratti piatti,
    escursione dinamica e coerenza reciproca tra assi.

    Parametri
    ---------
    x, y, z
        Componenti dinamiche dell'accelerazione dopo rimozione della gravità.
    magnitude
        Magnitudine vettoriale sqrt(x² + y² + z²).

    Restituisce
    -----------
    Un dizionario con metriche scalari di qualità e flag booleano `acc_valid`.
    """
    q: dict = {}

    # La magnitudine è il punto giusto dove cercare clipping, perché condensa i
    # tre assi in una misura meno dipendente dall'orientamento del sensore.
    q["acc_clip_ratio"] = _clip_ratio(magnitude)

    # Anche qui il flatline è sospetto: una magnitudine troppo immobile tende a
    # indicare sensore inattivo più che reale assenza completa di micro-dinamica.
    q["acc_flatline_ratio"] = _flatline_ratio(magnitude)

    # Range e deviazione standard aiutano a discriminare tra sensore "morto" e
    # sensore invece dominato da artefatti di ampiezza eccessiva.
    q["acc_magnitude_range"] = float(np.max(magnitude) - np.min(magnitude))
    q["acc_magnitude_std"] = float(np.std(magnitude, ddof=1)) if len(magnitude) > 1 else 0.0

    # Correlazioni molto alte tra tutti gli assi possono segnalare cross-talk o
    # malfunzionamento, perché in un movimento reale i tre assi non sono quasi
    # mai copie perfette l'uno dell'altro per tutta la finestra.
    if len(x) > 2:
        q["acc_corr_xy"] = float(np.corrcoef(x, y)[0, 1])
        q["acc_corr_xz"] = float(np.corrcoef(x, z)[0, 1])
        q["acc_corr_yz"] = float(np.corrcoef(y, z)[0, 1])
        q["acc_max_intercorr"] = float(max(
            abs(q["acc_corr_xy"]), abs(q["acc_corr_xz"]), abs(q["acc_corr_yz"])
        ))
    else:
        q["acc_corr_xy"] = q["acc_corr_xz"] = q["acc_corr_yz"] = np.nan
        q["acc_max_intercorr"] = np.nan

    # Il criterio composito cerca un equilibrio: scartare sia finestre troppo
    # povere di dinamica sia finestre talmente estreme da essere poco credibili.
    t = DEFAULT_ACC_THRESHOLDS
    q["acc_valid"] = (
        q["acc_clip_ratio"] <= t["clip_ratio_max"] and
        q["acc_flatline_ratio"] <= t["flatline_ratio_max"] and
        q["acc_magnitude_range"] >= t["min_magnitude_range"] and
        q["acc_magnitude_std"] <= t["max_magnitude_std"]
    )

    return q


# ── Combined validity ─────────────────────────────────────────────────────────

def is_window_valid(ecg_q: dict, acc_q: dict) -> bool:
    """
    Restituisce True solo se ECG e ACC sono entrambi accettabili.

    La scelta è volutamente conservativa: nella fusione multimodale basta che
    una sola modalità sia chiaramente scadente per compromettere l'affidabilità
    della finestra complessiva.
    """
    return bool(ecg_q.get("ecg_valid", False)) and bool(acc_q.get("acc_valid", False))


# ── Quality report ────────────────────────────────────────────────────────────

def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Riassume le statistiche di scarto su un DataFrame di feature.

    Questo report serve a quantificare il costo del quality control: sapere
    quante finestre vengono eliminate è importante per capire il trade-off tra
    rigore metodologico e quantità finale di dati disponibili per il modello.
    """
    rows = []
    total = len(df)

    if "ecg_valid" in df.columns:
        n_ecg_bad = int((~df["ecg_valid"]).sum())
        rows.append({"check": "ECG quality", "rejected": n_ecg_bad,
                     "pct": 100.0 * n_ecg_bad / total if total else 0.0})

    if "acc_valid" in df.columns:
        n_acc_bad = int((~df["acc_valid"]).sum())
        rows.append({"check": "ACC quality", "rejected": n_acc_bad,
                     "pct": 100.0 * n_acc_bad / total if total else 0.0})

    if "ecg_valid" in df.columns and "acc_valid" in df.columns:
        # Questa riga evidenzia la perdita effettiva in uno scenario multimodale:
        # basta che una delle due modalità fallisca perché la finestra sia critica.
        both_bad = int((~df["ecg_valid"] | ~df["acc_valid"]).sum())
        rows.append({"check": "Either ECG or ACC bad", "rejected": both_bad,
                     "pct": 100.0 * both_bad / total if total else 0.0})
        rows.append({"check": "Both valid (kept)", "rejected": total - both_bad,
                     "pct": 100.0 * (total - both_bad) / total if total else 0.0})

    return pd.DataFrame(rows)
