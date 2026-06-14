"""
Configurazione globale del progetto: percorsi, frequenze di campionamento
e parametri di elaborazione del segnale.

Centralizzare tutti i parametri qui permette di modificarli in un unico punto
senza dover cercare valori sparsi nel codice. Ogni costante è documentata
con la motivazione della scelta numerica.
"""
from pathlib import Path

# ── Percorsi del progetto ──────────────────────────────────────────────────────
# Risolto in modo relativo al file corrente per garantire portabilità
ROOT = Path(__file__).resolve().parents[1]
DS_DIR = ROOT / "ds"
ACC_DIR = DS_DIR / "acc"          # Segnali accelerometro (WFDB format)
ECG_DIR = DS_DIR / "ecg"          # Segnali ECG (WFDB format)
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"

SUBJECT_INFO_CSV = DS_DIR / "subject-info.csv"   # Anagrafica pazienti
TEST_AVAIL_CSV = DS_DIR / "test-availability.csv" # Quali test sono disponibili per paziente

# ── Frequenze di campionamento ─────────────────────────────────────────────────
# Frequenze native del dataset SCG-RHC (non modificabili senza ricampionamento)
ECG_FS = 130    # Hz — frequenza di campionamento ECG
ACC_FS = 200    # Hz — frequenza di campionamento accelerometro

# ── Etichette dei test riabilitativi ──────────────────────────────────────────
# Cinque test del protocollo: salita scale, cammino 6 minuti, TUG, cicloergometro,
# analisi del cammino. Usati come classi per OBJ1 (classificazione tipo attività).
TEST_LABELS = ["STAIR", "6MWT", "TUG", "VELO", "GAIT_ANALYSIS"]

# ── Parametri filtro ECG ───────────────────────────────────────────────────────
# Banda 0.5–40 Hz: soglia inferiore rimuove baseline wander respiratorio (~0.15-0.3 Hz),
# soglia superiore elimina rumore ad alta frequenza e componente di rete (50 Hz).
# Ordine 4: buon compromesso tra attenuazione fuori banda e distorsione di fase.
ECG_LOWCUT = 0.5    # Hz  — taglio basso bandpass
ECG_HIGHCUT = 40.0  # Hz  — taglio alto bandpass
ECG_FILTER_ORDER = 4

# Soglie di rigetto outlier R-R (in ms) — limiti fisiologici assoluti
# FC > 200 bpm → RR < 300 ms → quasi certamente artefatto da movimento
# FC < 30 bpm → RR > 2000 ms → asistolia o artefatto di connessione
RR_MIN_MS = 300     # ms — limite inferiore fisiologico
RR_MAX_MS = 2000    # ms — limite superiore fisiologico

# ── Parametri filtro accelerometro ────────────────────────────────────────────
# La gravità è una componente DC a ~0 Hz: un low-pass a 0.3 Hz la isola.
# Sottraendo la gravità si ottiene la componente dinamica del movimento.
# Banda 0.3–20 Hz: cattura camminata (~1-3 Hz), corsa (~3-5 Hz), jerk (~5-15 Hz).
ACC_GRAVITY_CUTOFF = 0.3   # Hz  — low-pass per estrarre componente gravitazionale
ACC_LOWCUT = 0.3            # Hz  — taglio basso bandpass (componente dinamica)
ACC_HIGHCUT = 20.0          # Hz  — taglio alto bandpass (limite artefatti da movimento)
ACC_FILTER_ORDER = 4

# ── Windowing ─────────────────────────────────────────────────────────────────
# Finestre da 30 secondi: sufficienti per calcolare SDNN, LF/HF e potenza spettrale
# con risoluzione accettabile (0.033 Hz in frequenza). Overlap 50% per aumentare
# il numero di campioni senza perdere contesto temporale.
WINDOW_SEC = 30     # Durata finestra in secondi
OVERLAP = 0.5       # Sovrapposizione tra finestre consecutive (50%)

# Minimo numero di picchi R per finestra: con meno di 5 R-peak non è possibile
# calcolare RMSSD, pNN50 e altri indici HRV in modo affidabile.
MIN_RPEAKS_PER_WINDOW = 5

# ── Bande di frequenza HRV (Task Force ESC/NASPE 1996) ────────────────────────
# VLF: meccanismi termoregolatori e umorali (non analizzata su finestre corte)
# LF: attività simpatica + vagale (0.04–0.15 Hz)
# HF: aritmia sinusale respiratoria — indice di tono vagale (0.15–0.40 Hz)
HRV_VLF_BAND = (0.003, 0.04)   # Hz — Very Low Frequency
HRV_LF_BAND  = (0.04,  0.15)   # Hz — Low Frequency
HRV_HF_BAND  = (0.15,  0.40)   # Hz — High Frequency
# Minimo numero di intervalli R-R per l'analisi spettrale (Lomb-Scargle):
# con meno di 30 campioni la stima PSD non è attendibile
HRV_MIN_RR_FOR_SPECTRAL = 30
