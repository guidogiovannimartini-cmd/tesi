"""
Classificazione multi-obiettivo — script standalone.

Addestra e valuta tre classificatori (Random Forest, SVM, Logistic Regression)
su tre obiettivi distinti, usando la pipeline modulare definita in src/classification.py.

Esegui da PowerShell:
    cd C:\\Users\\Guido\\Desktop\\tesi\\tesi
    python run_classification.py

I risultati vengono serializzati in results/models/ come file pickle e caricati
direttamente dal notebook 05 senza rieseguire la classificazione (che richiede ~15 min).

Struttura degli obiettivi:
  OBJ1 — Tipo di attività (5 classi): STAIR, 6MWT, TUG, VELO, GAIT_ANALYSIS
  OBJ2 — REST vs ACTIVITY (binario): distingue fasi di riposo da fasi di movimento
  OBJ3a — Stato clinico NYHA (4 classi, window-level con SMOTE)
  OBJ3b — NYHA binario: lieve (I+II) vs grave (III+IV), soglia=2
  OBJ3c — NYHA binario, aggregazione paziente con LOOCV (patient-level)

Nota su OBJ3: la strategia di validazione è StratifiedGroupKFold (group=patient_id)
per garantire che tutte le finestre dello stesso paziente siano nello stesso fold.
Questo è fondamentale per evitare data leakage: poiché la classe NYHA è un'etichetta
a livello paziente (non finestra), avere finestre dello stesso paziente sia nel
training che nel test set gonfierebbe artificialmente le performance.
"""

import warnings
warnings.filterwarnings('ignore')

import pickle
import time
from pathlib import Path

import pandas as pd
import sys
sys.path.insert(0, '.')

from src.classification import (
    classify_activity_type,
    classify_effort_level,
    classify_clinical_state,
    classify_clinical_state_patient_level,
)
from src.config import TEST_LABELS, MODELS_DIR

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Carica il dataset delle feature estratte dal notebook 03 / build_dataset.py
df = pd.read_csv('results/tables/features_all.csv')

# Seleziona solo le feature ECG (HRV) e ACC — esclude metadati e colonne SQI
feat = [c for c in df.columns if c.startswith('ecg_') or c.startswith('acc_')]

# Crea la colonna binaria REST(0) / ACTIVITY(1) per OBJ2
# Le finestre con etichetta nei TEST_LABELS sono attività, le altre sono riposo
df['is_activity'] = df['test_label'].isin(TEST_LABELS).astype(int)

print(f"Dataset: {len(df)} finestre, {len(feat)} feature\n")

# ── Obiettivo 1: tipo di attività ─────────────────────────────────────────────
# Problema multi-classe su 5 categorie: STAIR, 6MWT, TUG, VELO, GAIT_ANALYSIS
# Non richiede group-aware CV perché le classi non sono etichette paziente
print("=" * 50)
print("OBJ1 — Tipo di attività (5 classi)")
print("=" * 50)
t0 = time.time()
r1 = classify_activity_type(df, feature_cols=feat)
elapsed1 = time.time() - t0
with open(MODELS_DIR / 'obj1_results.pkl', 'wb') as f:
    pickle.dump(r1, f)
print(f"OBJ1 completato in {elapsed1/60:.1f} min → salvato in results/models/obj1_results.pkl\n")

# ── Obiettivo 2: REST vs ACTIVITY ────────────────────────────────────────────
# Classificatore binario: dimostra che ACC da solo discrimina riposo/movimento
# con alta accuratezza (~95%) — fondamentale per filtrare finestre di riposo
# prima di calcolare i parametri HRV clinicamente rilevanti
print("=" * 50)
print("OBJ2 — REST vs ACTIVITY (binario)")
print("=" * 50)
t0 = time.time()
r2 = classify_effort_level(df, feature_cols=feat)
elapsed2 = time.time() - t0
with open(MODELS_DIR / 'obj2_results.pkl', 'wb') as f:
    pickle.dump(r2, f)
print(f"OBJ2 completato in {elapsed2/60:.1f} min → salvato in results/models/obj2_results.pkl\n")

# ── Obiettivo 3a: stato clinico NYHA (4 classi, window-level + SMOTE) ────────
# SMOTE (Synthetic Minority Over-sampling Technique) è necessario perché le
# classi NYHA I e IV sono rare nel dataset: senza bilanciamento il modello
# ignorerebbe le classi minoritarie. StratifiedGroupKFold garantisce che
# lo stesso paziente non appaia in training e test contemporaneamente.
print("=" * 50)
print("OBJ3a — Stato clinico NYHA (4 classi, window-level + SMOTE)")
print("=" * 50)
clin_feat = [c for c in df.columns if c in
             ('age', 'gender', 'bmi', 'days_post_surgery', 'surgery_type')]
# NOTA: esclusi efs, has_af, beta_blockers — correlati direttamente con NYHA
# (includere features derivate dall'outcome sarebbe data leakage clinico)
all_feat = feat + [c for c in clin_feat if c in df.columns]
t0 = time.time()
r3a = classify_clinical_state(df, target_col='nyha', feature_cols=all_feat,
                               use_smote=True)
elapsed3a = time.time() - t0
with open(MODELS_DIR / 'obj3a_results.pkl', 'wb') as f:
    pickle.dump(r3a, f)
print(f"OBJ3a completato in {elapsed3a/60:.1f} min → salvato in results/models/obj3a_results.pkl\n")

# ── Obiettivo 3b: NYHA binario (I+II vs III+IV) ───────────────────────────────
# La binarizzazione con soglia=2 (≤2 lieve, >2 grave) riduce il problema a
# classificazione binaria: più gestibile con campioni limitati e più utile
# clinicamente (distinzione lieve/grave è la decisione terapeutica chiave)
print("=" * 50)
print("OBJ3b — NYHA binario: lieve (I+II) vs grave (III+IV)")
print("=" * 50)
t0 = time.time()
r3b = classify_clinical_state(df, target_col='nyha', feature_cols=all_feat,
                               threshold=2.0, use_smote=True)
elapsed3b = time.time() - t0
with open(MODELS_DIR / 'obj3b_results.pkl', 'wb') as f:
    pickle.dump(r3b, f)
print(f"OBJ3b completato in {elapsed3b/60:.1f} min → salvato in results/models/obj3b_results.pkl\n")

# ── Obiettivo 3c: aggregazione a livello paziente (LOOCV) ────────────────────
# Con pochi pazienti unici, la granularità effettiva è il paziente (non la finestra).
# Aggregare le feature per paziente (mean/std/median) e usare LOOCV (Leave-One-Out
# Cross-Validation) elimina il data leakage e fornisce la stima più onesta
# della generalizzazione con dataset di dimensioni limitate.
print("=" * 50)
print("OBJ3c — NYHA binario, aggregazione paziente (LOOCV)")
print("=" * 50)
t0 = time.time()
r3c = classify_clinical_state_patient_level(
    df, target_col='nyha', feature_cols=all_feat, threshold=2.0)
elapsed3c = time.time() - t0
with open(MODELS_DIR / 'obj3c_results.pkl', 'wb') as f:
    pickle.dump(r3c, f)
print(f"OBJ3c completato in {elapsed3c/60:.1f} min → salvato in results/models/obj3c_results.pkl\n")

elapsed3 = elapsed3a + elapsed3b + elapsed3c
# Alias di retrocompatibilità: obj3_results.pkl punta alla variante binaria window-level
r3 = r3b
with open(MODELS_DIR / 'obj3_results.pkl', 'wb') as f:
    pickle.dump(r3, f)

print("=" * 50)
print(f"TUTTO COMPLETATO — tempo totale: {(elapsed1+elapsed2+elapsed3)/60:.1f} min")
print("Ora apri il notebook 05 in JupyterLab e carica i risultati con pickle.load()")
print("=" * 50)

import warnings
warnings.filterwarnings('ignore')

import pickle
import time
from pathlib import Path

import pandas as pd
import sys
sys.path.insert(0, '.')

from src.classification import (
    classify_activity_type,
    classify_effort_level,
    classify_clinical_state,
    classify_clinical_state_patient_level,
)
from src.config import TEST_LABELS, MODELS_DIR

MODELS_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv('results/tables/features_all.csv')
feat = [c for c in df.columns if c.startswith('ecg_') or c.startswith('acc_')]
df['is_activity'] = df['test_label'].isin(TEST_LABELS).astype(int)

print(f"Dataset: {len(df)} finestre, {len(feat)} feature\n")

# ── Obiettivo 1: tipo di attività ─────────────────────────────────────────────
print("=" * 50)
print("OBJ1 — Tipo di attività (5 classi)")
print("=" * 50)
t0 = time.time()
r1 = classify_activity_type(df, feature_cols=feat)
elapsed1 = time.time() - t0
with open(MODELS_DIR / 'obj1_results.pkl', 'wb') as f:
    pickle.dump(r1, f)
print(f"OBJ1 completato in {elapsed1/60:.1f} min → salvato in results/models/obj1_results.pkl\n")

# ── Obiettivo 2: REST vs ACTIVITY ────────────────────────────────────────────
print("=" * 50)
print("OBJ2 — REST vs ACTIVITY (binario)")
print("=" * 50)
t0 = time.time()
r2 = classify_effort_level(df, feature_cols=feat)
elapsed2 = time.time() - t0
with open(MODELS_DIR / 'obj2_results.pkl', 'wb') as f:
    pickle.dump(r2, f)
print(f"OBJ2 completato in {elapsed2/60:.1f} min → salvato in results/models/obj2_results.pkl\n")

# ── Obiettivo 3a: stato clinico NYHA (4 classi, window-level + SMOTE) ────────
print("=" * 50)
print("OBJ3a — Stato clinico NYHA (4 classi, window-level + SMOTE)")
print("=" * 50)
clin_feat = [c for c in df.columns if c in
             ('age', 'gender', 'bmi', 'days_post_surgery', 'surgery_type')]
# NOTA: esclusi efs, has_af, beta_blockers — correlati direttamente con NYHA
# OBJ3 usa solo ECG+ACC+dati anagrafici per evitare data leakage
all_feat = feat + [c for c in clin_feat if c in df.columns]
t0 = time.time()
r3a = classify_clinical_state(df, target_col='nyha', feature_cols=all_feat,
                               use_smote=True)
elapsed3a = time.time() - t0
with open(MODELS_DIR / 'obj3a_results.pkl', 'wb') as f:
    pickle.dump(r3a, f)
print(f"OBJ3a completato in {elapsed3a/60:.1f} min → salvato in results/models/obj3a_results.pkl\n")

# ── Obiettivo 3b: NYHA binario (I+II vs III+IV) ───────────────────────────────
print("=" * 50)
print("OBJ3b — NYHA binario: lieve (I+II) vs grave (III+IV)")
print("=" * 50)
t0 = time.time()
r3b = classify_clinical_state(df, target_col='nyha', feature_cols=all_feat,
                               threshold=2.0, use_smote=True)
elapsed3b = time.time() - t0
with open(MODELS_DIR / 'obj3b_results.pkl', 'wb') as f:
    pickle.dump(r3b, f)
print(f"OBJ3b completato in {elapsed3b/60:.1f} min → salvato in results/models/obj3b_results.pkl\n")

# ── Obiettivo 3c: aggregazione a livello paziente (LOOCV) ────────────────────
print("=" * 50)
print("OBJ3c — NYHA binario, aggregazione paziente (LOOCV)")
print("=" * 50)
t0 = time.time()
r3c = classify_clinical_state_patient_level(
    df, target_col='nyha', feature_cols=all_feat, threshold=2.0)
elapsed3c = time.time() - t0
with open(MODELS_DIR / 'obj3c_results.pkl', 'wb') as f:
    pickle.dump(r3c, f)
print(f"OBJ3c completato in {elapsed3c/60:.1f} min → salvato in results/models/obj3c_results.pkl\n")

elapsed3 = elapsed3a + elapsed3b + elapsed3c
# Keep backward-compat alias pointing to best window-level run
r3 = r3b
with open(MODELS_DIR / 'obj3_results.pkl', 'wb') as f:
    pickle.dump(r3, f)

print("=" * 50)
print(f"TUTTO COMPLETATO — tempo totale: {(elapsed1+elapsed2+elapsed3)/60:.1f} min")
print("Ora apri il notebook 05 in JupyterLab e carica i risultati con pickle.load()")
print("=" * 50)
