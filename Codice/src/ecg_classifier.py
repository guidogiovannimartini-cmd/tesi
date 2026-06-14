"""
Classificatore rule-based dei ritmi ECG — Scenario B.

Ogni finestra viene assegnata a una di quattro categorie usando soltanto
feature HRV già presenti nel DataFrame, senza richiedere il segnale grezzo.
La scelta rule-based è intenzionale: in ambito tesi e in prospettiva clinica
consente massima interpretabilità, non richiede una fase di training separata e
si appoggia a soglie già consolidate nella pratica medica, riducendo il rischio
di introdurre complessità non necessaria per classificazioni di base come
bradicardia e tachicardia.

Classi
------
NORMAL       — HR 60-100 bpm, intervalli R-R regolari
BRADY        — HR < 60 bpm  (bradicardia)
TACHY        — HR > 100 bpm (tachicardia)
POSSIBLE_AF  — elevata irregolarità degli intervalli R-R, compatibile con AF

Algoritmo
---------
1.  BRADY  : hr_mean < BRADY_THRESHOLD
2.  TACHY  : hr_mean > TACHY_THRESHOLD
3.  POSSIBLE_AF : irregularity score >= AF_SCORE_THRESHOLD
    Punteggio = somma delle condizioni soddisfatte:
      - cv_rr  > 0.20   (coefficiente di variazione > 20 %)
      - rmssd  > 50 ms  (alta variabilità battito-battito)
      - pnn50  > 20 %   (molte differenze successive > 50 ms)
      - sd1    > 35 ms  (dispersione a breve termine nel piano di Poincaré)
4.  NORMAL : nessuna delle condizioni precedenti

Riferimenti
-----------
Task Force of ESC/NASPE (1996), Eur. Heart J. 17:354-381.
Lim & Leung (2017), "Automatic AF detection from single-lead ECG",
  Physiol. Meas. 38:1560-1576.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Soglie ───────────────────────────────────────────────────────────────────

BRADY_HR_MAX   = 60.0    # bpm — soglia clinica standard per preservare leggibilità e confrontabilità
TACHY_HR_MIN   = 100.0   # bpm — soglia clinica standard che evita di "apprendere" limiti da coorti piccole

# Il punteggio AF combina più indici perché una singola feature HRV isolata
# sarebbe più fragile al rumore e meno convincente in ottica clinica.
AF_SCORE_MIN        = 3     # need at least 3 out of 4 criteria
AF_CV_RR_MIN        = 0.20  # coefficient of variation of R-R (dimensionless)
AF_RMSSD_MIN        = 50.0  # ms
AF_PNN50_MIN        = 20.0  # %
AF_SD1_MIN          = 35.0  # ms

# La codifica intera facilita l'integrazione con pipeline successive senza
# rinunciare alle etichette testuali, che restano più leggibili nella discussione.
CLASS_MAP = {"NORMAL": 0, "BRADY": 1, "TACHY": 2, "POSSIBLE_AF": 3}
CLASS_LABELS = {v: k for k, v in CLASS_MAP.items()}


# ── Core classification logic ─────────────────────────────────────────────────

def classify_window(row: pd.Series | dict) -> str:
    """
    Classifica una singola finestra in una delle quattro classi di ritmo.

    L'obiettivo non è sostituire un classificatore ML generale, ma fornire una
    baseline trasparente e immediatamente verificabile. In questo scenario,
    soglie esplicite e regole semplici aiutano a motivare ogni decisione del
    sistema, aspetto particolarmente rilevante quando il risultato deve essere
    discusso in chiave clinica.

    Parametri
    ---------
    row : dict o pd.Series con almeno 'hr_mean' e opzionalmente
          'cv_rr', 'rmssd', 'pnn50', 'sd1'.

    Restituisce
    -----------
    str : 'NORMAL' | 'BRADY' | 'TACHY' | 'POSSIBLE_AF'
    """
    hr = _get(row, "hr_mean")

    # Restituire UNKNOWN evita inferenze arbitrarie quando manca la metrica più
    # informativa: è una scelta conservativa più difendibile di un fallback.
    if np.isnan(hr):
        return "UNKNOWN"

    # La bradicardia viene valutata per prima perché la soglia <60 bpm è una
    # regola clinica diretta, più forte e più interpretabile di indici derivati.
    if hr < BRADY_HR_MAX:
        return "BRADY"

    # Anche la tachicardia usa una soglia clinica standard >100 bpm, scelta
    # proprio per evitare dipendenze da training o da distribuzioni locali.
    if hr > TACHY_HR_MIN:
        return "TACHY"

    # L'AF viene lasciata come terzo step perché richiede un ragionamento più
    # indiziario: prima si intercettano pattern semplici, poi l'irregolarità.
    af_score = 0
    cv_rr = _get(row, "cv_rr")
    rmssd = _get(row, "rmssd")
    pnn50 = _get(row, "pnn50")
    sd1   = _get(row, "sd1")

    if not np.isnan(cv_rr) and cv_rr > AF_CV_RR_MIN:
        af_score += 1
    if not np.isnan(rmssd) and rmssd > AF_RMSSD_MIN:
        af_score += 1
    if not np.isnan(pnn50) and pnn50 > AF_PNN50_MIN:
        af_score += 1
    if not np.isnan(sd1) and sd1 > AF_SD1_MIN:
        af_score += 1

    if af_score >= AF_SCORE_MIN:
        return "POSSIBLE_AF"

    return "NORMAL"


def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge a *df* le colonne 'ecg_class' (stringa) ed 'ecg_class_id' (int).

    Opera sulle feature HRV già disponibili, senza bisogno dell'ECG grezzo.
    Restituisce una copia con due nuove colonne aggiunte: mantenere la copia
    separata evita effetti collaterali sul dataset originale, scelta utile
    quando si confrontano più strategie sperimentali nella stessa analisi.

    Parametri
    ---------
    df : DataFrame contenente feature HRV per finestra.

    Restituisce
    -----------
    pd.DataFrame con colonne aggiunte:
        ecg_class     : str  ('NORMAL', 'BRADY', 'TACHY', 'POSSIBLE_AF', 'UNKNOWN')
        ecg_class_id  : int  (0=NORMAL, 1=BRADY, 2=TACHY, 3=POSSIBLE_AF, -1=UNKNOWN)
    """
    df = df.copy()
    df["ecg_class"] = df.apply(classify_window, axis=1)
    df["ecg_class_id"] = df["ecg_class"].map(CLASS_MAP).fillna(-1).astype(int)
    return df


def classification_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restituisce una tabella riassuntiva della distribuzione delle classi ECG.

    Parametri
    ---------
    df : DataFrame con colonna 'ecg_class' (output di classify_dataframe).

    Restituisce
    -----------
    pd.DataFrame con colonne: class, count, pct
    """
    if "ecg_class" not in df.columns:
        raise ValueError("DataFrame must contain 'ecg_class' column. "
                         "Run classify_dataframe() first.")

    counts = df["ecg_class"].value_counts().reset_index()
    counts.columns = ["class", "count"]
    counts["pct"] = (counts["count"] / len(df) * 100).round(2)
    counts = counts.sort_values("class").reset_index(drop=True)
    return counts


def af_detail_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per le finestre etichettate come POSSIBLE_AF, mostra quali criteri si attivano.

    Questo report è utile nella tesi perché rende verificabile il motivo per cui
    una finestra è stata considerata compatibile con AF, rafforzando il valore
    esplicativo dell'approccio rule-based.
    """
    if "ecg_class" not in df.columns:
        df = classify_dataframe(df)

    af_rows = df[df["ecg_class"] == "POSSIBLE_AF"].copy()
    if af_rows.empty:
        return pd.DataFrame()

    af_rows["af_crit_cv"]    = af_rows.get("cv_rr",  np.nan) > AF_CV_RR_MIN
    af_rows["af_crit_rmssd"] = af_rows.get("rmssd",  np.nan) > AF_RMSSD_MIN
    af_rows["af_crit_pnn50"] = af_rows.get("pnn50",  np.nan) > AF_PNN50_MIN
    af_rows["af_crit_sd1"]   = af_rows.get("sd1",    np.nan) > AF_SD1_MIN
    af_rows["af_score"]      = (
        af_rows["af_crit_cv"].astype(int) +
        af_rows["af_crit_rmssd"].astype(int) +
        af_rows["af_crit_pnn50"].astype(int) +
        af_rows["af_crit_sd1"].astype(int)
    )
    cols = ["af_crit_cv", "af_crit_rmssd", "af_crit_pnn50", "af_crit_sd1", "af_score"]
    return af_rows[cols]


# ── Helper ────────────────────────────────────────────────────────────────────

def _get(row, key: str, default: float = float("nan")) -> float:
    """
    Recupera in modo sicuro un valore numerico da un dict o da una pd.Series.

    La funzione centralizza la gestione dei casi mancanti o malformati per
    mantenere uniforme il comportamento del classificatore ed evitare che errori
    di formattazione si traducano in decisioni clinicamente fuorvianti.
    """
    try:
        val = row[key]
        return float(val) if val is not None else default
    except (KeyError, TypeError, ValueError):
        return default
