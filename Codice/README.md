# Tesi — Correlazione Attività Motoria e ECG

Analisi della correlazione tra attività motoria (accelerometro MEMS) e segnali ECG (intervalli R-R) in pazienti post-cardiochirurgici, con classificazione multi-obiettivo tramite machine learning.

## Struttura

```
tesi/
├── ds/                  # Dataset WFDB (PhysioNet) — non modificare
├── src/                 # Codice sorgente pipeline
│   ├── config.py        # Percorsi e parametri globali
│   ├── data_loader.py   # Caricamento dati WFDB
│   ├── preprocessing.py # Filtri, R-peak detection, gravity removal
│   ├── feature_extraction.py  # HRV, feature ACC, feature cliniche
│   ├── correlation_analysis.py # Pearson/Spearman, regressione lineare
│   ├── classification.py      # RF, SVM, LR — 3 obiettivi
│   ├── evaluation.py          # Metriche, confusion matrix, feature importance
│   └── visualization.py       # Figure per la tesi
├── notebooks/           # Jupyter Notebooks esplorativi
├── results/             # Output: figure, tabelle, modelli
└── requirements.txt
```

## Installazione

```bash
pip install -r requirements.txt
```

## Utilizzo rapido

```python
from src.data_loader import load_patient, get_test_segments
from src.preprocessing import preprocess_ecg, preprocess_acc
from src.feature_extraction import extract_all_features
from src.correlation_analysis import compute_correlations
from src.classification import classify_activity_type

# Carica un paziente
data = load_patient("001", session=1)
ecg = data["ecg"]
acc = data["acc"]

# Preprocessing
ecg_prep = preprocess_ecg(ecg["signal"], ecg["fs"])
acc_prep = preprocess_acc(acc["signal_x"], acc["signal_y"], acc["signal_z"], acc["fs"])

# Segmenti per test motorio
segments = get_test_segments(
    ecg["ann_samples"], ecg["ann_labels"],
    acc["ann_samples"], acc["ann_labels"],
    len(ecg["signal"]), len(acc["signal_x"]),
)

# Feature extraction
features_df = extract_all_features("001", 1, ecg_prep, acc_prep, segments)

# Correlazione ACC ↔ ECG
corr = compute_correlations(features_df)
print(corr.head(10))
```

## Notebooks

| Notebook | Contenuto |
|---|---|
| `01_data_exploration` | Statistiche descrittive, visualizzazione segnali grezzi |
| `02_preprocessing` | Effetti dei filtri, R-peak detection, gravity removal |
| `03_feature_extraction` | Distribuzione delle feature estratte |
| `04_correlation_analysis` | Heatmap correlazione, scatter plot, regressione lineare |
| `05_classification` | Training modelli, cross-validation, confronto |
| `06_results_visualization` | Figure finali per la tesi |

## Dataset

Dataset WFDB PhysioNet — pazienti post-cardiochirurgici con 5 test motori:
- **STAIR**: salita/discesa scale
- **6MWT**: 6-Minute Walk Test
- **TUG**: Timed Up and Go
- **VELO**: veloergometria
- **GAIT_ANALYSIS**: analisi del cammino (sistema Zebris)
