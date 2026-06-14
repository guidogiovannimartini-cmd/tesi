"""
Preprocessing dei segnali ECG e accelerometrici.

Questo modulo raccoglie le trasformazioni preliminari necessarie per portare
segnali fisiologici grezzi in una forma più adatta all'estrazione di feature.
L'idea metodologica è ridurre le componenti che non riflettono direttamente la
dinamica cardio-meccanica di interesse, preservando però la temporizzazione
degli eventi fisiologici. In un contesto ECG questo significa, soprattutto,
non alterare la posizione dei picchi R, perché da lì derivano gli intervalli
R-R e quindi molte misure legate alla variabilità cardiaca. Per l'ACC, invece,
è importante separare la componente gravitazionale lenta da quella dinamica,
che è più informativa rispetto al movimento reale del soggetto.

Funzioni principali
-------------------
preprocess_ecg(signal, fs)
    Filtraggio passa-banda, rilevazione dei picchi R e pulizia adattiva degli
    intervalli R-R.
preprocess_acc(signal_x, signal_y, signal_z, fs)
    Stima della gravità, rimozione della componente statica e costruzione della
    magnitudine vettoriale.
segment_signal(signal, fs, start_sample, end_sample, window_sec, overlap)
    Suddivisione di un segnale in finestre sovrapposte a lunghezza fissa.
"""

import numpy as np
import neurokit2 as nk
from scipy.signal import butter, filtfilt, sosfiltfilt, butter as _butter

from .config import (
    ECG_FS, ACC_FS,
    ECG_LOWCUT, ECG_HIGHCUT, ECG_FILTER_ORDER,
    ACC_GRAVITY_CUTOFF, ACC_LOWCUT, ACC_HIGHCUT, ACC_FILTER_ORDER,
    RR_MIN_MS, RR_MAX_MS,
    WINDOW_SEC, OVERLAP,
)


# ── Generic filter helpers ─────────────────────────────────────────────────────

def _bandpass(signal: np.ndarray, lowcut: float, highcut: float,
              fs: float, order: int = 4) -> np.ndarray:
    """
    Applica un filtro Butterworth passa-banda a fase nulla.

    La scelta del Butterworth nasce dal fatto che ha una risposta in ampiezza
    regolare in banda passante: in un segnale fisiologico questo è utile perché
    si preferisce non introdurre ondulazioni artificiali sulle componenti che
    vogliamo preservare. L'uso della forma SOS (Second-Order Sections) rende il
    filtraggio più stabile numericamente rispetto alla forma diretta, mentre il
    filtraggio forward-backward evita sfasamenti temporali dei picchi.
    """
    nyq = fs / 2.0
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def _lowpass(signal: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Applica un filtro Butterworth passa-basso a fase nulla.

    Qui il passa-basso serve soprattutto quando interessa isolare componenti
    lente, come la gravità sull'accelerometro. Mantenere fase nulla è utile
    anche in questo caso, perché evita di deformare l'allineamento temporale
    tra i tre assi e tra ACC ed eventuale ECG analizzato in parallelo.
    """
    nyq = fs / 2.0
    sos = butter(order, cutoff / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


def _highpass(signal: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Applica un filtro Butterworth passa-alto a fase nulla.

    Anche se in questo file non è il filtro usato più spesso, mantenerlo come
    helper dedicato rende esplicita la possibilità di rimuovere derive lente
    senza compromettere la localizzazione temporale degli eventi fisiologici.
    """
    nyq = fs / 2.0
    sos = butter(order, cutoff / nyq, btype="high", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


# ── ECG preprocessing ──────────────────────────────────────────────────────────

def preprocess_ecg(signal: np.ndarray, fs: float = ECG_FS) -> dict:
    """
    Esegue la pipeline completa di preprocessing dell'ECG.

    La logica seguita è quella tipica dell'analisi cardiaca: prima si ripulisce
    il segnale nelle bande più informative per la morfologia ECG, poi si
    individuano i picchi R e infine si controlla che gli intervalli R-R siano
    fisiologicamente plausibili. L'obiettivo non è solo "trovare battiti", ma
    trovare battiti sufficientemente affidabili da poter sostenere analisi di
    HR e HRV senza essere dominate da artefatti o errori di detezione.

    Passi
    -----
    1. Filtro passa-banda ECG (tipicamente 0.5–40 Hz): attenua baseline wander
       e rumore ad alta frequenza, lasciando intatte le componenti utili per la
       rilevazione del complesso QRS.
    2. Rilevazione dei picchi R con Pan-Tompkins via neurokit2: è una scelta
       classica e robusta per segnali ECG campionati in modo regolare.
    3. Calcolo degli intervalli R-R in millisecondi: unità più intuitive dal
       punto di vista fisiologico rispetto ai campioni.
    4. Rimozione adattiva degli outlier: prima si impongono limiti fisiologici
       minimi e massimi, poi si eliminano intervalli troppo lontani dal trend
       centrale locale.

    Restituisce
    -----------
    dict con:
        filtered    : segnale ECG filtrato
        rpeaks      : indici campionari dei picchi R rilevati
        rr_ms       : intervalli R-R puliti in millisecondi
        rr_samples  : campioni dei picchi R usati per gli intervalli puliti
        fs          : frequenza di campionamento
    """
    # Si mantiene una banda limitata perché il QRS cade in gran parte sotto
    # i 40 Hz, mentre drift respiratorio/elettrodo e rumore muscolare tendono
    # a stare rispettivamente più in basso e più in alto.
    filtered = _bandpass(signal, ECG_LOWCUT, ECG_HIGHCUT, fs, ECG_FILTER_ORDER)

    # Pan-Tompkins resta una scelta sensata in tesi applicative perché ha una
    # base metodologica solida sul rilievo della pendenza/energia del QRS.
    # L'opzione correct_artifacts prova inoltre a limitare picchi spurii che
    # altrimenti si trasformerebbero in errori macroscopici sugli R-R.
    _, info = nk.ecg_peaks(filtered, sampling_rate=int(fs), method="pantompkins1985",
                            correct_artifacts=True)
    rpeaks = info["ECG_R_Peaks"].astype(np.int64)

    if len(rpeaks) < 2:
        # Con meno di due picchi non esiste alcun intervallo R-R: restituisco
        # comunque il segnale filtrato, perché può servire per SQI o debug.
        return {
            "filtered": filtered, "rpeaks": rpeaks,
            "rr_ms": np.array([]), "rr_samples": rpeaks, "fs": fs,
        }

    # Gli R-R vengono portati in millisecondi perché i range fisiologici sono
    # più facili da ragionare in questa scala (es. bradicardia/tachicardia).
    rr_ms = np.diff(rpeaks) / fs * 1000.0

    # Il primo filtro sugli outlier usa soglie fisiologiche esplicite: valori
    # troppo brevi o troppo lunghi sono spesso dovuti a doppie detezioni,
    # picchi persi oppure artefatti di movimento.
    valid_mask = (rr_ms >= RR_MIN_MS) & (rr_ms <= RR_MAX_MS)
    if valid_mask.sum() > 4:
        # Quando ho abbastanza intervalli validi, uso mediana e deviazione
        # standard per un controllo adattivo: non voglio imporre un ritmo
        # "medio" assoluto, ma eliminare ciò che si discosta troppo dal contesto
        # locale della finestra analizzata.
        median_rr = np.median(rr_ms[valid_mask])
        std_rr = np.std(rr_ms[valid_mask])
        valid_mask &= np.abs(rr_ms - median_rr) <= 3 * std_rr

    rr_ms_clean = rr_ms[valid_mask]
    # rr_ms[i] rappresenta l'intervallo tra rpeaks[i] e rpeaks[i+1]; perciò
    # uso l'indice del picco iniziale per mantenere l'allineamento temporale
    # con la finestra ECG originaria.
    clean_rpeak_idx = np.where(valid_mask)[0]
    rr_samples = rpeaks[clean_rpeak_idx]

    return {
        "filtered": filtered,
        "rpeaks": rpeaks,
        "rr_ms": rr_ms_clean.astype(np.float32),
        "rr_samples": rr_samples,
        "fs": fs,
    }


# ── Accelerometer preprocessing ───────────────────────────────────────────────

def preprocess_acc(
    signal_x: np.ndarray,
    signal_y: np.ndarray,
    signal_z: np.ndarray,
    fs: float = ACC_FS,
) -> dict:
    """
    Esegue la pipeline completa di preprocessing dell'accelerometro.

    Dal punto di vista metodologico, l'accelerometro contiene sempre una
    componente dovuta all'orientamento rispetto alla gravità e una componente
    dovuta al movimento reale. Separarle è importante perché, se lasciassi la
    gravità dentro al segnale, rischierei di interpretare come attività dinamica
    un semplice cambio lento di postura.

    Passi
    -----
    1. Stima della gravità con un passa-basso molto lento.
    2. Sottrazione della gravità per ottenere l'accelerazione dinamica.
    3. Passa-banda sulla componente dinamica per concentrarsi sulle frequenze
       più plausibilmente associate al movimento corporeo utile all'analisi.
    4. Calcolo della magnitudine vettoriale per ottenere una misura meno
       sensibile all'orientamento del sensore sui tre assi.

    Restituisce
    -----------
    dict con i tre assi dinamici, le componenti gravitazionali stimate, la
    magnitudine e la frequenza di campionamento.
    """
    # Una cutoff molto bassa permette di modellare la gravità come componente
    # quasi-statica: così i movimenti più rapidi non "inquinano" la sua stima.
    gravity_x = _lowpass(signal_x, ACC_GRAVITY_CUTOFF, fs, ACC_FILTER_ORDER)
    gravity_y = _lowpass(signal_y, ACC_GRAVITY_CUTOFF, fs, ACC_FILTER_ORDER)
    gravity_z = _lowpass(signal_z, ACC_GRAVITY_CUTOFF, fs, ACC_FILTER_ORDER)

    # Sottraggo la gravità prima del passa-banda perché voglio analizzare la
    # dinamica vera del gesto/movimento, non il contributo statico del sensore.
    dyn_x = _bandpass(signal_x - gravity_x, ACC_LOWCUT, ACC_HIGHCUT, fs, ACC_FILTER_ORDER)
    dyn_y = _bandpass(signal_y - gravity_y, ACC_LOWCUT, ACC_HIGHCUT, fs, ACC_FILTER_ORDER)
    dyn_z = _bandpass(signal_z - gravity_z, ACC_LOWCUT, ACC_HIGHCUT, fs, ACC_FILTER_ORDER)

    # La magnitudine vettoriale rende il descrittore più robusto ai piccoli
    # cambi di orientamento, aspetto importante nei wearable reali.
    magnitude = np.sqrt(dyn_x**2 + dyn_y**2 + dyn_z**2).astype(np.float32)

    return {
        "x_dynamic": dyn_x,
        "y_dynamic": dyn_y,
        "z_dynamic": dyn_z,
        "gravity_x": gravity_x,
        "gravity_y": gravity_y,
        "gravity_z": gravity_z,
        "magnitude": magnitude,
        "fs": fs,
    }


# ── Windowing ─────────────────────────────────────────────────────────────────

def segment_signal(
    signal: np.ndarray,
    fs: float,
    start_sample: int = 0,
    end_sample: int | None = None,
    window_sec: float = WINDOW_SEC,
    overlap: float = OVERLAP,
) -> list[tuple[int, int, np.ndarray]]:
    """
    Suddivide un segnale in finestre fisse e sovrapposte.

    La finestratura serve a trasformare un segnale continuo in unità locali di
    analisi, più gestibili per estrazione di feature e classificazione. La
    sovrapposizione riduce il rischio di perdere eventi che cadono a cavallo
    tra due finestre e rende la stima temporale meno brusca.
    """
    if end_sample is None:
        end_sample = len(signal)

    win_len = int(window_sec * fs)
    # Uso uno step più corto della finestra quando c'è overlap per ottenere una
    # rappresentazione temporale più densa senza cambiare il contenuto interno
    # di ciascuna finestra.
    step = int(win_len * (1.0 - overlap))
    windows = []
    idx = start_sample
    while idx + win_len <= end_sample:
        windows.append((idx, idx + win_len, signal[idx: idx + win_len]))
        idx += step
    return windows


def segment_rr(
    rr_ms: np.ndarray,
    rr_samples: np.ndarray,
    fs: float,
    start_sample: int,
    end_sample: int,
    window_sec: float = WINDOW_SEC,
    overlap: float = OVERLAP,
) -> list[tuple[int, int, np.ndarray]]:
    """
    Suddivide la serie degli intervalli R-R su finestre allineate all'ECG.

    L'idea è usare la stessa griglia temporale impiegata per il segnale ECG,
    così le feature derivate dagli R-R possono essere confrontate o fuse con
    quelle morfologiche senza ambiguità di allineamento.
    """
    win_len = int(window_sec * fs)
    step = int(win_len * (1.0 - overlap))
    windows = []
    idx = start_sample
    while idx + win_len <= end_sample:
        # Il mascheramento sui campioni dei picchi R è preferibile a una
        # ricostruzione uniforme della serie RR, perché evita di inventare
        # campioni temporali non realmente osservati.
        mask = (rr_samples >= idx) & (rr_samples < idx + win_len)
        if mask.sum() > 0:
            windows.append((idx, idx + win_len, rr_ms[mask]))
        idx += step
    return windows
