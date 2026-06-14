"""
Analisi di correlazione ACC ↔ HRV (ECG) su tutto il dataset.

Questo script calcola le correlazioni tra le feature estratte dall'accelerometro
e i parametri HRV derivati dall'ECG, per verificare l'ipotesi che il movimento
(ACC) sia statisticamente associato alla risposta cardiaca autonomica (HRV).

L'analisi è strutturata in 6 fasi:
  1. Correlazione ACC vs ecg_mean_rr (Pearson + Spearman)
  2. Correlazione ACC vs tutte le feature HRV (SDNN, RMSSD, LF/HF, SD1/SD2...)
  3. Regressione lineare OLS (R², slope) per quantificare la relazione
  4. Confronto REST vs ACTIVITY (t-test + Mann-Whitney) per ogni tipo di test
  5. Heatmap di correlazione ECG×ACC per le feature più informative
  6. Scatter plot con retta di regressione per le top-3 correlazioni

Perché sia Pearson che Spearman?
  Pearson misura la relazione lineare; Spearman (rank-based) è più robusta a
  outlier e non assume normalità delle distribuzioni. Usarle entrambe permette
  di distinguere correlazioni lineari da monotone non-lineari.

Perché Mann-Whitney oltre al t-test?
  La distribuzione degli intervalli R-R non è normalmente distribuita (è
  asimmetrica a destra per effetto degli outlier residui). Mann-Whitney è il
  test non parametrico equivalente al t-test di Welch.

Esegui:
    python run_correlation.py

Output:
    results/tables/correlation_acc_rr.csv       — Pearson/Spearman ACC vs mean_RR
    results/tables/regression_acc_rr.csv         — R², slope per ogni feature ACC
    results/tables/rest_vs_activity.csv          — t-test / Mann-Whitney RR per test
    results/tables/correlation_acc_hrv_full.csv  — correlazioni per ogni feature HRV
    results/figures/correlation_heatmap.png      — heatmap ECG+ACC features
    results/figures/scatter_top*.png             — scatter top correlazioni
"""

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import FIGURES_DIR, TABLES_DIR
from src.correlation_analysis import (
    compute_correlations,
    correlation_matrix,
    linear_regression_summary,
    activity_vs_rr_comparison,
    plot_correlation_heatmap,
    plot_scatter_regression,
)

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Carica dataset ─────────────────────────────────────────────────────────────
# Il dataset features_all.csv è prodotto da build_dataset.py e contiene
# una riga per finestra con tutte le feature ECG, ACC, SQI e metadati
df = pd.read_csv(TABLES_DIR / "features_all.csv")
ecg_cols = [c for c in df.columns if c.startswith("ecg_")]
acc_cols  = [c for c in df.columns if c.startswith("acc_")]
print(f"Dataset: {len(df)} finestre | {len(ecg_cols)} ECG features | {len(acc_cols)} ACC features\n")

# ── 1. Correlazione ACC vs mean_RR ────────────────────────────────────────────
# ecg_mean_rr è la variabile target principale: l'intervallo R-R medio (ms)
# è inversamente proporzionale alla frequenza cardiaca (FC = 60000/RR).
# Ci aspettiamo correlazione negativa con le feature di movimento:
# più movimento → maggiore FC → minore RR (r < 0 è fisiologicamente corretto)
print("1. Correlazione ACC → ecg_mean_rr...")
corr_rr = compute_correlations(df, ecg_col="ecg_mean_rr", acc_cols=acc_cols)
corr_rr.to_csv(TABLES_DIR / "correlation_acc_rr.csv", index=False)
print(f"   Top 5 correlazioni (Pearson):\n{corr_rr.head().to_string(index=False)}\n")

# ── 2. Correlazione ACC vs tutte le feature HRV ───────────────────────────────
# Estende l'analisi a tutti gli indici HRV per capire quali dimensioni della
# variabilità cardiaca sono maggiormente influenzate dall'attività motoria.
# SDNN e RMSSD catturano la variabilità totale; LF/HF il bilancio simpato-vagale;
# SD1/SD2 la geometria del poincaré plot.
print("2. Correlazione ACC → tutte le feature ECG/HRV...")
hrv_targets = ["ecg_mean_rr", "ecg_sdnn", "ecg_rmssd", "ecg_pnn50",
               "ecg_lf_hf_ratio", "ecg_sd1", "ecg_sd2", "ecg_hr_mean"]
hrv_targets = [c for c in hrv_targets if c in df.columns]

all_corr = []
for target in hrv_targets:
    corr = compute_correlations(df, ecg_col=target, acc_cols=acc_cols)
    corr.insert(0, "hrv_target", target)
    all_corr.append(corr)

full_corr_df = pd.concat(all_corr, ignore_index=True)
full_corr_df.to_csv(TABLES_DIR / "correlation_acc_hrv_full.csv", index=False)
print(f"   Salvato: {len(full_corr_df)} coppie ACC×HRV\n")

# ── 3. Regressione lineare ACC → mean_RR ─────────────────────────────────────
# OLS (Ordinary Least Squares) fornisce R² per ogni feature: misura la
# frazione di varianza di mean_RR spiegata da quella singola feature ACC.
# Risultato atteso: R²_max ≈ 0.09 (relazione reale ma moderata, come in letteratura)
print("3. Regressione lineare ACC → ecg_mean_rr...")
reg_df = linear_regression_summary(df, target_col="ecg_mean_rr", feature_cols=acc_cols)
reg_df.to_csv(TABLES_DIR / "regression_acc_rr.csv", index=False)
print(f"   Top 5 predittori (R²):\n{reg_df.head().to_string(index=False)}\n")

# ── 4. REST vs ACTIVITY per tipo di test ─────────────────────────────────────
# Confronto tra RR medio a riposo e durante ogni test: verifica che la
# risposta cardiaca all'esercizio sia statisticamente significativa.
# Un p < 0.05 su entrambi t-test e Mann-Whitney conferma la differenza.
print("4. Confronto REST vs ACTIVITY (RR intervals)...")
rest_act = activity_vs_rr_comparison(df)
if not rest_act.empty:
    rest_act.to_csv(TABLES_DIR / "rest_vs_activity.csv", index=False)
    print(rest_act[["test_label", "mean_rest", "mean_activity", "delta_mean",
                     "t_p", "mw_p"]].to_string(index=False))
else:
    print("   [SKIP] nessuna finestra REST trovata nel dataset")
print()

# ── 5. Heatmap correlazione ECG + ACC ────────────────────────────────────────
# Visualizza la struttura di correlazione tra le feature più informative.
# Utile per identificare ridondanze (feature altamente correlate tra loro)
# e per giustificare l'uso di feature importance del Random Forest invece
# di selezione manuale delle feature.
print("5. Heatmap correlazione ECG + ACC...")
key_cols = (
    ["ecg_mean_rr", "ecg_sdnn", "ecg_rmssd", "ecg_pnn50",
     "ecg_lf_hf_ratio", "ecg_sd1", "ecg_sd2", "ecg_hr_mean"]
    + corr_rr.head(15)["feature"].tolist()   # top-15 feature ACC per correlazione
)
key_cols = [c for c in key_cols if c in df.columns]
cm = correlation_matrix(df, cols=key_cols)
plot_correlation_heatmap(
    cm,
    output_path=FIGURES_DIR / "correlation_heatmap.png",
    title="Correlazione ECG (HRV) × ACC — top feature",
    figsize=(14, 12),
)
print(f"   ✓ Heatmap salvata\n")

# ── 6. Scatter plot top-3 correlazioni ───────────────────────────────────────
# Visualizza la relazione tra le 3 feature ACC più correlate e ecg_mean_rr,
# stratificata per tipo di test. Permette di verificare visivamente che la
# correlazione non sia artefatta da un singolo tipo di attività.
print("6. Scatter plot top correlazioni ACC → RR...")
top_acc = corr_rr.head(3)["feature"].tolist()
for acc_feat in top_acc:
    plot_scatter_regression(
        df,
        x_col=acc_feat,
        y_col="ecg_mean_rr",
        hue_col="test_label",
        output_path=FIGURES_DIR / f"scatter_{acc_feat}_vs_rr.png",
    )
    r = corr_rr.loc[corr_rr["feature"] == acc_feat, "pearson_r"].values[0]
    print(f"   ✓ {acc_feat} (r={r:.3f})")

print(f"\nTutto completato! Output in:\n  {TABLES_DIR.resolve()}\n  {FIGURES_DIR.resolve()}")

import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import FIGURES_DIR, TABLES_DIR
from src.correlation_analysis import (
    compute_correlations,
    correlation_matrix,
    linear_regression_summary,
    activity_vs_rr_comparison,
    plot_correlation_heatmap,
    plot_scatter_regression,
)

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Carica dataset ─────────────────────────────────────────────────────────────
df = pd.read_csv(TABLES_DIR / "features_all.csv")
ecg_cols = [c for c in df.columns if c.startswith("ecg_")]
acc_cols  = [c for c in df.columns if c.startswith("acc_")]
print(f"Dataset: {len(df)} finestre | {len(ecg_cols)} ECG features | {len(acc_cols)} ACC features\n")

# ── 1. Correlazione ACC vs mean_RR ────────────────────────────────────────────
print("1. Correlazione ACC → ecg_mean_rr...")
corr_rr = compute_correlations(df, ecg_col="ecg_mean_rr", acc_cols=acc_cols)
corr_rr.to_csv(TABLES_DIR / "correlation_acc_rr.csv", index=False)
print(f"   Top 5 correlazioni (Pearson):\n{corr_rr.head().to_string(index=False)}\n")

# ── 2. Correlazione ACC vs tutte le feature HRV ───────────────────────────────
print("2. Correlazione ACC → tutte le feature ECG/HRV...")
hrv_targets = ["ecg_mean_rr", "ecg_sdnn", "ecg_rmssd", "ecg_pnn50",
               "ecg_lf_hf_ratio", "ecg_sd1", "ecg_sd2", "ecg_hr_mean"]
hrv_targets = [c for c in hrv_targets if c in df.columns]

all_corr = []
for target in hrv_targets:
    corr = compute_correlations(df, ecg_col=target, acc_cols=acc_cols)
    corr.insert(0, "hrv_target", target)
    all_corr.append(corr)

full_corr_df = pd.concat(all_corr, ignore_index=True)
full_corr_df.to_csv(TABLES_DIR / "correlation_acc_hrv_full.csv", index=False)
print(f"   Salvato: {len(full_corr_df)} coppie ACC×HRV\n")

# ── 3. Regressione lineare ACC → mean_RR ─────────────────────────────────────
print("3. Regressione lineare ACC → ecg_mean_rr...")
reg_df = linear_regression_summary(df, target_col="ecg_mean_rr", feature_cols=acc_cols)
reg_df.to_csv(TABLES_DIR / "regression_acc_rr.csv", index=False)
print(f"   Top 5 predittori (R²):\n{reg_df.head().to_string(index=False)}\n")

# ── 4. REST vs ACTIVITY per tipo di test ─────────────────────────────────────
print("4. Confronto REST vs ACTIVITY (RR intervals)...")
rest_act = activity_vs_rr_comparison(df)
if not rest_act.empty:
    rest_act.to_csv(TABLES_DIR / "rest_vs_activity.csv", index=False)
    print(rest_act[["test_label", "mean_rest", "mean_activity", "delta_mean",
                     "t_p", "mw_p"]].to_string(index=False))
else:
    print("   [SKIP] nessuna finestra REST trovata nel dataset")
print()

# ── 5. Heatmap correlazione ECG + ACC ────────────────────────────────────────
print("5. Heatmap correlazione ECG + ACC...")
key_cols = (
    ["ecg_mean_rr", "ecg_sdnn", "ecg_rmssd", "ecg_pnn50",
     "ecg_lf_hf_ratio", "ecg_sd1", "ecg_sd2", "ecg_hr_mean"]
    + corr_rr.head(15)["feature"].tolist()   # top ACC features
)
key_cols = [c for c in key_cols if c in df.columns]
cm = correlation_matrix(df, cols=key_cols)
plot_correlation_heatmap(
    cm,
    output_path=FIGURES_DIR / "correlation_heatmap.png",
    title="Correlazione ECG (HRV) × ACC — top feature",
    figsize=(14, 12),
)
print(f"   ✓ Heatmap salvata\n")

# ── 6. Scatter plot top-3 correlazioni ───────────────────────────────────────
print("6. Scatter plot top correlazioni ACC → RR...")
top_acc = corr_rr.head(3)["feature"].tolist()
for acc_feat in top_acc:
    plot_scatter_regression(
        df,
        x_col=acc_feat,
        y_col="ecg_mean_rr",
        hue_col="test_label",
        output_path=FIGURES_DIR / f"scatter_{acc_feat}_vs_rr.png",
    )
    r = corr_rr.loc[corr_rr["feature"] == acc_feat, "pearson_r"].values[0]
    print(f"   ✓ {acc_feat} (r={r:.3f})")

print(f"\nTutto completato! Output in:\n  {TABLES_DIR.resolve()}\n  {FIGURES_DIR.resolve()}")
