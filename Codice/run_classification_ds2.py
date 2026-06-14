"""
Classificazione DS2 — Compensato vs Scompensato (CDecomp binario).

Usa le stesse feature HRV + SCG estratte dal wearable patch (patch_ECG +
patch_ACC_lat/hf/dv), senza nessuna misura invasiva RHC.

Prima esegui:
    python build_dataset_ds2.py

Poi lancia:
    python run_classification_ds2.py

I risultati vengono salvati in results/models/ come file pickle.
Tempo atteso: ~5-10 minuti (dataset più piccolo di DS).
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import time
from pathlib import Path

import pandas as pd
import sys
sys.path.insert(0, ".")

from src.classification import (
    classify_clinical_state,
    classify_clinical_state_patient_level,
)
from src.config import MODELS_DIR

# ── Carica dataset DS2 ─────────────────────────────────────────────────────────
CSV_PATH = "results/tables/features_ds2.csv"
if not Path(CSV_PATH).exists():
    print(f"[ERRORE] {CSV_PATH} non trovato.")
    print("Esegui prima:  python build_dataset_ds2.py")
    sys.exit(1)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Mantieni solo finestre con CDecomp noto (0 o 1)
df = df[df["cdecomp"].isin([0, 1])].copy()
df["cdecomp"] = df["cdecomp"].astype(int)

feat = [c for c in df.columns if c.startswith("ecg_") or c.startswith("acc_")]
clin_feat = [c for c in ("age", "gender", "bmi", "sbp", "dbp") if c in df.columns]
all_feat = feat + clin_feat

print("=" * 60)
print("DS2 — Classificazione Compensato vs Scompensato (CDecomp)")
print("=" * 60)
print(f"Dataset: {len(df)} finestre | {len(all_feat)} feature")
print(f"Record unici: {df['patient_id'].nunique()}")
print(f"CDecomp — 0=Compensato: {(df['cdecomp']==0).sum()}  "
      f"1=Scompensato: {(df['cdecomp']==1).sum()}")
print()

# ── Obiettivo DS2a: window-level, StratifiedGroupKFold per paziente ─────────────
print("=" * 60)
print("OBJ_DS2a — Finestre, StratifiedGroupKFold (anti-leakage)")
print("=" * 60)
t0 = time.time()
r_ds2a = classify_clinical_state(
    df,
    target_col="cdecomp",
    feature_cols=all_feat,
    use_smote=True,
)
elapsed_a = time.time() - t0
with open(MODELS_DIR / "ds2_cdecomp_windows.pkl", "wb") as f:
    pickle.dump(r_ds2a, f)
print(f"Completato in {elapsed_a/60:.1f} min → ds2_cdecomp_windows.pkl\n")

# ── Obiettivo DS2b: patient-level aggregation, LOOCV ──────────────────────────
print("=" * 60)
print("OBJ_DS2b — Aggregazione paziente, LOOCV (gold standard)")
print("=" * 60)
t0 = time.time()
r_ds2b = classify_clinical_state_patient_level(
    df,
    target_col="cdecomp",
    feature_cols=all_feat,
)
elapsed_b = time.time() - t0
with open(MODELS_DIR / "ds2_cdecomp_patient.pkl", "wb") as f:
    pickle.dump(r_ds2b, f)
print(f"Completato in {elapsed_b/60:.1f} min → ds2_cdecomp_patient.pkl\n")

# ── Riepilogo ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("RIEPILOGO RISULTATI DS2")
print("=" * 60)
print("\nOBJ_DS2a — Window-level:")
for model_name, res in r_ds2a.items():
    f1 = res["report"]["weighted avg"]["f1-score"]
    print(f"  {model_name:<20} F1-weighted = {f1:.3f}")

print("\nOBJ_DS2b — Patient-level (LOOCV):")
for model_name, res in r_ds2b.items():
    f1 = res["report"]["weighted avg"]["f1-score"]
    print(f"  {model_name:<20} F1-weighted = {f1:.3f}")

total = elapsed_a + elapsed_b
print(f"\nTempo totale: {total/60:.1f} min")
print("Apri notebook 05 o 06 per visualizzare confusion matrix e ROC curve.")
