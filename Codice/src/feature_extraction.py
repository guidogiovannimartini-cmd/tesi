"""
Estrazione delle feature da finestre ECG e accelerometro pre-elaborate.

Questo modulo implementa il calcolo di 72 feature totali (19 HRV + 40 ACC + cliniche)
che costituiscono il vettore di ingresso per i classificatori ML.

Le feature sono organizzate in tre gruppi:
  - HRV (ECG): indici nel dominio del tempo (SDNN, RMSSD, pNN50, HTI, TINN),
    della frequenza (LF, HF, LF/HF) e non lineari (ApEn, SampEn, DFA, SD1/SD2)
  - ACC (accelerometro): indici statistici (media, std, percentili), spettrali
    (frequenza dominante via Lomb-Scargle, potenza per banda) e cinematici
    (SMA, jerk, magnitudine)
  - Cliniche: dati anagrafici e post-operatori dal subject-info.csv

Funzioni principali
-------------------
ecg_features(rr_ms)
    Feature HRV da un array di intervalli R-R in ms.
acc_features(x, y, z, magnitude, fs)
    Feature statistiche e spettrali da una finestra ACC triassiale.
clinical_features(subject_row)
    Feature numeriche dai metadati clinici del paziente.
extract_all_features(patient_data, ecg_preprocessed, acc_preprocessed, segments, subject_row)
    Pipeline completa: estrae le feature per ogni finestra di ogni segmento di test.
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from scipy.signal import welch

from .config import ECG_FS, ACC_FS, WINDOW_SEC, OVERLAP, MIN_RPEAKS_PER_WINDOW
from .preprocessing import segment_signal, segment_rr
from .signal_quality import ecg_sqi, acc_sqi, is_window_valid


# ── HRV non-linear helpers ────────────────────────────────────────────────────

def _tinn(rr: np.ndarray) -> float:
    """Calcola il TINN (Triangular Interpolation of NN Histogram).

    Il TINN stima la base del triangolo che approssima meglio l'istogramma
    degli intervalli R-R: in pratica riassume quanto è "larga" la
    distribuzione dei battiti e quindi quanta variabilità globale è presente.
    I bin da 1/128 s seguono la convenzione classica HRV, così il confronto con
    la letteratura resta coerente.
    """
    bin_w = 1000.0 / 128.0
    bins = np.arange(rr.min(), rr.max() + bin_w, bin_w)
    if len(bins) < 3:
        return float("nan")
    hist, edges = np.histogram(rr, bins=bins)
    n_bins = len(hist)
    N_idx = int(np.argmax(hist))
    peak_h = float(hist[N_idx])
    centers = (edges[:-1] + edges[1:]) / 2

    best_err = np.inf
    best_tinn = float("nan")
    for m_idx in range(0, N_idx):
        for M_idx in range(N_idx + 1, n_bins):
            tri = np.zeros(n_bins)
            # Si cerca il triangolo con errore minimo perché il TINN non dipende
            # solo dal picco dell'istogramma, ma dalla sua forma complessiva.
            left = np.arange(m_idx, N_idx + 1)
            tri[left] = peak_h * (left - m_idx) / max(N_idx - m_idx, 1)
            right = np.arange(N_idx, M_idx + 1)
            tri[right] = peak_h * (M_idx - right) / max(M_idx - N_idx, 1)
            err = float(np.sum((hist - tri) ** 2))
            if err < best_err:
                best_err = err
                best_tinn = centers[M_idx] - centers[m_idx]
    return float(best_tinn)


def _approx_entropy(rr: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Calcola l'Approximate Entropy (ApEn).

    ApEn misura quanto i pattern degli intervalli R-R restano prevedibili
    quando si aumenta la lunghezza del pattern da m a m+1. Valori più alti
    indicano una dinamica meno regolare, utile quando si vuole descrivere la
    complessità del controllo autonomico oltre la sola varianza.
    """
    N = len(rr)
    r = r_factor * np.std(rr, ddof=1)
    if r == 0.0:
        return float("nan")

    def _phi(m_val: int) -> float:
        Nm = N - m_val + 1
        X = np.array([rr[i: i + m_val] for i in range(Nm)])
        total = 0.0
        for i in range(Nm):
            cheby = np.max(np.abs(X - X[i]), axis=1)
            # ApEn include anche l'auto-match: è una scelta storica che rende
            # la stima più stabile su serie corte, ma introduce dipendenza da N.
            total += np.log(np.sum(cheby <= r) / Nm)
        return total / Nm

    return float(_phi(m) - _phi(m + 1))


def _sample_entropy(rr: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Calcola la Sample Entropy (SampEn).

    SampEn è spesso preferita ad ApEn nelle finestre corte tipiche del
    monitoraggio fisiologico perché esclude le auto-correlazioni banali
    (self-matches) e quindi riduce il bias legato alla lunghezza N della
    sequenza. In questo modo la stima della complessità è più confrontabile tra
    finestre con numerosità simile ma non identica.
    """
    N = len(rr)
    r = r_factor * np.std(rr, ddof=1)
    if r == 0.0:
        return float("nan")
    Nm = N - m
    if Nm <= 1:
        return float("nan")

    Xm  = np.array([rr[i: i + m]     for i in range(Nm)])  # [N-m, m]
    Xm1 = np.array([rr[i: i + m + 1] for i in range(Nm)])  # [N-m, m+1]

    B = A = 0
    for i in range(Nm):
        # Il "- 1" rimuove esplicitamente l'auto-match: è il motivo principale
        # per cui SampEn risulta meno ottimistica e meno biasata di ApEn.
        B += np.sum(np.max(np.abs(Xm  - Xm[i]),  axis=1) <= r) - 1
        A += np.sum(np.max(np.abs(Xm1 - Xm1[i]), axis=1) <= r) - 1

    if B == 0:
        return float("nan")
    return float(-np.log(A / B)) if A > 0 else float("nan")


def _dfa(rr: np.ndarray) -> tuple[float, float]:
    """Detrended Fluctuation Analysis. Returns (alpha1, alpha2).

    alpha1: short-range correlation (scales 4–16)
    alpha2: long-range correlation  (scales 16–64)
    """
    N = len(rr)
    y = np.cumsum(rr - np.mean(rr))

    scales = np.unique(
        np.round(np.logspace(np.log10(4), np.log10(min(64, N // 4)), 20)).astype(int)
    )
    scales = scales[(scales >= 4) & (scales <= N // 2)]

    F_vals, S_used = [], []
    for n in scales:
        nw = N // n
        if nw == 0:
            continue
        segs = y[: nw * n].reshape(nw, n)
        t = np.arange(n, dtype=float)
        rms_sq = []
        for seg in segs:
            c = np.polyfit(t, seg, 1)
            rms_sq.append(np.mean((seg - np.polyval(c, t)) ** 2))
        F_vals.append(np.sqrt(np.mean(rms_sq)))
        S_used.append(n)

    if len(F_vals) < 4:
        return float("nan"), float("nan")

    log_s = np.log10(np.array(S_used, dtype=float))
    log_F = np.log10(np.array(F_vals) + 1e-12)

    s1 = np.array(S_used)
    m1 = (s1 >= 4)  & (s1 <= 16)
    m2 = (s1 >= 16) & (s1 <= 64)

    alpha1 = float(np.polyfit(log_s[m1], log_F[m1], 1)[0]) if m1.sum() >= 2 else float("nan")
    alpha2 = float(np.polyfit(log_s[m2], log_F[m2], 1)[0]) if m2.sum() >= 2 else float("nan")
    return alpha1, alpha2


# ── ECG / HRV features ────────────────────────────────────────────────────────

def ecg_features(rr_ms: np.ndarray) -> dict:
    """
    Estrae feature HRV temporali, geometriche, spettrali e non lineari.

    L'idea è descrivere la regolazione cardiaca da più punti di vista: la
    dispersione globale degli intervalli, la dinamica battito-per-battito, la
    distribuzione geometrica e la complessità del segnale. Se i dati in finestra
    non sono sufficienti, restituisce NaN per evitare di introdurre stime poco
    affidabili nel dataset finale.
    """
    feat = {}
    _nan_keys = (
        "mean_rr", "sdnn", "rmssd", "pnn50", "pnn20",
        "nn50", "nn20",                               # absolute NN counts
        "hr_mean", "hr_min", "hr_max", "rr_range", "cv_rr",
        "sd1", "sd2", "sd1_sd2_ratio",                # Poincaré plot indices
        "poincare_area",                              # ellipse area π·SD1·SD2
        "triangular_index", "tinn",                   # geometric indices
        "lf_power", "hf_power", "lf_hf_ratio",       # Frequency-domain
        "vlf_power", "total_power",
        "lf_nu", "hf_nu",                             # normalised units
        "spectral_exponent",                          # 1/f slope
        "apen", "sampen",                             # entropy indices
        "dfa_alpha1", "dfa_alpha2",                   # DFA scaling exponents
    )
    if len(rr_ms) < MIN_RPEAKS_PER_WINDOW:
        for k in _nan_keys:
            feat[k] = float("nan")
        return feat

    rr = rr_ms.astype(np.float64)
    diff_rr = np.diff(rr)

    # ── Time-domain ───────────────────────────────────────────────────────────
    feat["mean_rr"]  = float(np.mean(rr))
    # SDNN riassume la variabilità totale della finestra, quindi è utile come
    # indicatore sintetico del bilancio autonomico complessivo.
    feat["sdnn"]     = float(np.std(rr, ddof=1))
    # RMSSD enfatizza le differenze tra battiti consecutivi e per questo riflette
    # soprattutto la modulazione vagale a breve termine.
    feat["rmssd"]    = float(np.sqrt(np.mean(diff_rr ** 2)))
    # pNN50 conta quante transizioni superano una soglia clinicamente interpretabile:
    # è un modo semplice per intercettare oscillazioni marcate o battiti atipici.
    feat["pnn50"]    = float(np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100)
    feat["pnn20"]    = float(np.sum(np.abs(diff_rr) > 20) / len(diff_rr) * 100)
    feat["nn50"]     = int(np.sum(np.abs(diff_rr) > 50))
    feat["nn20"]     = int(np.sum(np.abs(diff_rr) > 20))

    hr = 60_000.0 / rr
    feat["hr_mean"]  = float(np.mean(hr))
    feat["hr_min"]   = float(np.min(hr))
    feat["hr_max"]   = float(np.max(hr))
    feat["rr_range"] = float(np.max(rr) - np.min(rr))
    feat["cv_rr"]    = float(feat["sdnn"] / feat["mean_rr"] * 100)

    # ── Poincaré plot indices (SD1, SD2) ─────────────────────────────────────
    # SD1: short-term variability (parasympathetic) ~ RMSSD / sqrt(2)
    # SD2: long-term variability (sympatho-vagal balance)
    feat["sd1"] = float(feat["rmssd"] / np.sqrt(2))
    feat["sd2"] = float(np.sqrt(max(0.0, 2 * feat["sdnn"] ** 2 - feat["sd1"] ** 2)))
    feat["sd1_sd2_ratio"] = (float(feat["sd1"] / feat["sd2"])
                             if feat["sd2"] > 0 else float("nan"))
    feat["poincare_area"] = float(np.pi * feat["sd1"] * feat["sd2"])

    # ── HRV Triangular Index (HTI) ────────────────────────────────────────────
    # HTI = total number of R-R intervals / height of histogram peak
    # Uses 1/128 s (≈ 7.8 ms) bin width as per Task Force standard
    if len(rr) >= 10:
        bin_width = 1000.0 / 128.0          # ms
        bins = np.arange(rr.min(), rr.max() + bin_width, bin_width)
        hist, _ = np.histogram(rr, bins=bins)
        peak = int(hist.max())
        feat["triangular_index"] = float(len(rr) / peak) if peak > 0 else float("nan")
        feat["tinn"] = _tinn(rr)
    else:
        feat["triangular_index"] = float("nan")
        feat["tinn"] = float("nan")

    # ── Non-linear indices (entropy & DFA) ───────────────────────────────────
    if len(rr) >= 20:
        feat["apen"]   = _approx_entropy(rr)
        feat["sampen"] = _sample_entropy(rr)
    else:
        feat["apen"]   = float("nan")
        feat["sampen"] = float("nan")

    if len(rr) >= 16:
        feat["dfa_alpha1"], feat["dfa_alpha2"] = _dfa(rr)
    else:
        feat["dfa_alpha1"] = float("nan")
        feat["dfa_alpha2"] = float("nan")

    # ── Frequency-domain (Lomb-Scargle on unevenly-sampled R-R) ─────────────
    # Requires at least 30 R-R intervals for reliable spectral estimates
    if len(rr) >= 30:
        try:
            from scipy.signal import lombscargle
            # Build R-peak timestamps (seconds)
            t = np.cumsum(rr) / 1000.0
            t -= t[0]
            # Angular frequencies for VLF, LF, HF bands
            # VLF: 0.003–0.04 Hz | LF: 0.04–0.15 Hz | HF: 0.15–0.40 Hz
            f_lo, f_hi = 0.003, 0.40
            freqs = np.linspace(f_lo, f_hi, 512)
            ang_freqs = 2 * np.pi * freqs
            # Lomb-Scargle è preferibile qui perché la serie R-R è definita sugli
            # istanti dei battiti e quindi non è uniformemente campionata.
            # Normalizzare prima della stima evita che il valor medio domini la PSD.
            rr_norm = rr - rr.mean()
            pgram = lombscargle(t, rr_norm, ang_freqs, normalize=True)

            df = freqs[1] - freqs[0]
            vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
            lf_mask  = (freqs >= 0.04)  & (freqs < 0.15)
            hf_mask  = (freqs >= 0.15)  & (freqs < 0.40)

            feat["vlf_power"]    = float(np.sum(pgram[vlf_mask]) * df)
            feat["lf_power"]     = float(np.sum(pgram[lf_mask])  * df)
            feat["hf_power"]     = float(np.sum(pgram[hf_mask])  * df)
            feat["total_power"]  = float(feat["vlf_power"] + feat["lf_power"] + feat["hf_power"])
            feat["lf_hf_ratio"]  = (float(feat["lf_power"] / feat["hf_power"])
                                    if feat["hf_power"] > 0 else float("nan"))

            # Normalised units: LF_nu = LF/(LF+HF)*100, HF_nu = HF/(LF+HF)*100
            lf_hf_sum = feat["lf_power"] + feat["hf_power"]
            if lf_hf_sum > 0:
                feat["lf_nu"] = float(feat["lf_power"] / lf_hf_sum * 100)
                feat["hf_nu"] = float(feat["hf_power"] / lf_hf_sum * 100)
            else:
                feat["lf_nu"] = float("nan")
                feat["hf_nu"] = float("nan")

            # Spectral exponent α: slope of log-log PSD in VLF band (1/f noise)
            vlf_freqs = freqs[vlf_mask]
            vlf_pgram = pgram[vlf_mask]
            if len(vlf_freqs) >= 4 and vlf_pgram.max() > 0:
                alpha_slope = np.polyfit(
                    np.log10(vlf_freqs),
                    np.log10(vlf_pgram + 1e-12),
                    1,
                )[0]
                feat["spectral_exponent"] = float(-alpha_slope)
            else:
                feat["spectral_exponent"] = float("nan")
        except Exception:
            for k in ("vlf_power", "lf_power", "hf_power", "total_power", "lf_hf_ratio",
                      "lf_nu", "hf_nu", "spectral_exponent"):
                feat[k] = float("nan")
    else:
        for k in ("vlf_power", "lf_power", "hf_power", "total_power", "lf_hf_ratio",
                  "lf_nu", "hf_nu", "spectral_exponent"):
            feat[k] = float("nan")

    return feat


# ── Accelerometer features ────────────────────────────────────────────────────

def acc_features(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    magnitude: np.ndarray,
    fs: float = ACC_FS,
) -> dict:
    """
    Estrae feature statistiche e spettrali da una finestra accelerometrica.

    Le feature ACC servono a riassumere intensità, direzione e rapidità del
    movimento: così il modello può distinguere non solo quanto il soggetto si
    muove, ma anche se il pattern motorio è regolare, brusco o multi-assiale.
    """
    feat = {}
    axes = {"x": x, "y": y, "z": z, "mag": magnitude}

    for name, sig in axes.items():
        feat[f"{name}_mean"] = float(np.mean(sig))
        feat[f"{name}_std"] = float(np.std(sig, ddof=1))
        feat[f"{name}_energy"] = float(np.sum(sig**2) / len(sig))
        feat[f"{name}_peak2peak"] = float(np.max(sig) - np.min(sig))
        feat[f"{name}_rms"] = float(np.sqrt(np.mean(sig**2)))
        # La frequenza dominante aiuta a separare attività lente, ritmiche o
        # impulsive. Qui basta Welch perché il segnale ACC resta uniformemente
        # campionato nella finestra; Lomb-Scargle è invece riservato alle serie
        # irregolari come gli intervalli R-R.
        freqs, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)))
        feat[f"{name}_dom_freq"] = float(freqs[np.argmax(psd)])
        feat[f"{name}_spectral_energy"] = float(np.sum(psd))
        # Shannon entropy on normalised PSD
        psd_norm = psd / (psd.sum() + 1e-12)
        feat[f"{name}_entropy"] = float(scipy_entropy(psd_norm + 1e-12))
        # Il jerk valorizza i cambiamenti rapidi di accelerazione, quindi è utile
        # per evidenziare transizioni brusche che l'energia media può smussare.
        jerk = np.diff(sig) * fs
        feat[f"{name}_jerk_mean"] = float(np.mean(np.abs(jerk)))

    # La SMA condensa il "carico" motorio complessivo della finestra sommando il
    # contributo assoluto dei tre assi, quindi è robusta rispetto all'orientamento.
    feat["sma"] = float((np.sum(np.abs(x)) + np.sum(np.abs(y)) + np.sum(np.abs(z))) / len(x))

    # Inter-axis correlations (proxy for cross-talk / crosstalk)
    feat["corr_xy"] = float(np.corrcoef(x, y)[0, 1])
    feat["corr_xz"] = float(np.corrcoef(x, z)[0, 1])
    feat["corr_yz"] = float(np.corrcoef(y, z)[0, 1])

    return feat


# ── Clinical features ─────────────────────────────────────────────────────────

def clinical_features(subject_row: pd.Series) -> dict:
    """
    Extract numerical clinical features from a subject-info row.

    Columns are searched by substring matching to be robust to header
    variations.  Missing values are left as NaN.
    """
    feat = {}
    _get = lambda kw: next(
        (float(subject_row[c]) for c in subject_row.index if kw.lower() in c.lower()
         and not pd.isna(subject_row[c])),
        float("nan"),
    )

    feat["age"] = _get("Age")
    feat["gender"] = _get("Gender")
    feat["bmi"] = _get("BMI") if "BMI" in subject_row.index else float("nan")
    feat["efs"] = _get("EFS")
    feat["days_post_surgery"] = _get("Days after surgery")
    feat["surgery_type"] = _get("Surgery type")

    # NYHA: map roman numerals to integers if stored as string
    nyha_val = next(
        (subject_row[c] for c in subject_row.index if "NYHA" in c),
        float("nan"),
    )
    _nyha_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
    if isinstance(nyha_val, str):
        feat["nyha"] = float(_nyha_map.get(nyha_val.strip("?"), float("nan")))
    else:
        try:
            feat["nyha"] = float(nyha_val)
        except (TypeError, ValueError):
            feat["nyha"] = float("nan")

    feat["has_af"] = _get("Atrial fibrillation")
    feat["has_copd"] = _get("Chronic obstructive")
    feat["has_depression"] = _get("Depression")
    feat["ace_inhibitors"] = _get("ACE inhibitors")
    feat["beta_blockers"] = _get("Beta blockers")

    return feat


# ── Full pipeline ─────────────────────────────────────────────────────────────

def extract_all_features(
    patient_id: str,
    session: int,
    ecg_prep: dict,
    acc_prep: dict,
    segments: dict,
    subject_row: pd.Series | None = None,
    window_sec: float = WINDOW_SEC,
    overlap: float = OVERLAP,
    keep_invalid: bool = False,
) -> pd.DataFrame:
    """
    Estrae tutte le feature finestra per finestra su ogni segmento di test.

    Parameters
    ----------
    patient_id   : str
    session      : int
    ecg_prep     : output of preprocessing.preprocess_ecg()
    acc_prep     : output of preprocessing.preprocess_acc()
    segments     : output of data_loader.get_test_segments()
    subject_row  : row from subject-info DataFrame (optional)
    window_sec   : window length in seconds
    overlap      : window overlap fraction
    keep_invalid : if False (default), windows failing SQI checks are dropped.
                   If True, they are kept with ecg_valid/acc_valid flags = False.

    Returns
    -------
    pd.DataFrame — una riga per finestra, con colonne:
        patient_id, session, test_label, window_start_ecg, window_end_ecg,
        ecg_valid, acc_valid,   ← SQI flags
        + tutte le feature ECG / ACC / cliniche
    """
    rows = []
    clin = clinical_features(subject_row) if subject_row is not None else {}

    for test_label, seg in segments.items():
        # ── ECG windows (R-R based) ──────────────────────────────────────────
        rr_windows = segment_rr(
            ecg_prep["rr_ms"],
            ecg_prep["rr_samples"],
            ecg_prep["fs"],
            seg["ecg_start"],
            seg["ecg_end"],
            window_sec,
            overlap,
        )

        # ── ACC windows (uniform) ────────────────────────────────────────────
        # Convert ECG window grid to ACC sample domain
        ecg_fs = ecg_prep["fs"]
        acc_fs = acc_prep["fs"]

        for ecg_start, ecg_end, rr_win in rr_windows:
            # Lo schema finestra-per-finestra mantiene allineati i descrittori
            # cardiaci e motori nello stesso intervallo temporale, così ogni riga
            # del dataset rappresenta uno stato fisiologico locale e confrontabile.
            # Map ECG window boundaries to ACC domain
            acc_win_start = int(ecg_start / ecg_fs * acc_fs)
            acc_win_end = int(ecg_end / ecg_fs * acc_fs)

            acc_win_start = max(acc_win_start, seg["acc_start"])
            acc_win_end = min(acc_win_end, seg["acc_end"])

            # Guard: enough ACC samples and ECG R-R intervals
            acc_len = acc_win_end - acc_win_start
            min_acc_samples = int(window_sec * acc_fs * 0.5)
            if acc_len < min_acc_samples or len(rr_win) < MIN_RPEAKS_PER_WINDOW:
                continue

            x_win = acc_prep["x_dynamic"][acc_win_start:acc_win_end]
            y_win = acc_prep["y_dynamic"][acc_win_start:acc_win_end]
            z_win = acc_prep["z_dynamic"][acc_win_start:acc_win_end]
            mag_win = acc_prep["magnitude"][acc_win_start:acc_win_end]

            # ── Signal quality check ─────────────────────────────────────────
            ecg_win_signal = ecg_prep["filtered"][ecg_start:ecg_end]
            # Raw R-peaks in this window (before outlier rejection)
            rpeaks_raw_win = ecg_prep["rpeaks"][
                (ecg_prep["rpeaks"] >= ecg_start) & (ecg_prep["rpeaks"] < ecg_end)
            ] - ecg_start
            # Clean R-peaks in this window
            rpeaks_clean_win = ecg_prep["rr_samples"][
                (ecg_prep["rr_samples"] >= ecg_start) & (ecg_prep["rr_samples"] < ecg_end)
            ] - ecg_start

            ecg_q = ecg_sqi(ecg_win_signal, rpeaks_raw_win, rpeaks_clean_win,
                            rr_win, ecg_prep["fs"])
            acc_q = acc_sqi(x_win, y_win, z_win, mag_win)

            if not keep_invalid and not is_window_valid(ecg_q, acc_q):
                continue

            # Si aggregano feature, indici di qualità e metadati nella stessa riga
            # per facilitare il training supervisionato senza perdere il contesto.
            row = {
                "patient_id": patient_id,
                "session": session,
                "test_label": test_label,
                "window_start_ecg": ecg_start,
                "window_end_ecg": ecg_end,
            }
            row.update({f"ecg_{k}": v for k, v in ecg_features(rr_win).items()})
            row.update({f"acc_{k}": v for k, v in acc_features(x_win, y_win, z_win, mag_win, acc_fs).items()})
            row.update(ecg_q)
            row.update(acc_q)
            row.update(clin)
            rows.append(row)

    return pd.DataFrame(rows)
