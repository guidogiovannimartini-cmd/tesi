"""
Funzioni di visualizzazione per la tesi — grafici di qualità pubblicazione.

Questo modulo raccoglie le utility grafiche usate nei notebook e negli script
per produrre le figure inserite nella tesi. Separare la visualizzazione dalla
logica analitica facilita la riproduzione dei grafici senza rieseguire i calcoli.

Funzioni
--------
plot_ecg_acc_overview(ecg_signal, acc_magnitude, ecg_fs, acc_fs, ...)
    Visualizza ECG e magnitudine ACC allineati su un'unica figura.
plot_rr_timeseries(rr_ms, rr_samples, ecg_fs, test_segments, ...)
    Serie temporale degli intervalli R-R con marcatori di onset dei test.
plot_feature_boxplots(df, feature_cols, group_col, ...)
    Box plot delle feature raggruppate per tipo di test.
plot_psd(signal, fs, label, ...)
    Densità spettrale di potenza (PSD) tramite metodo di Welch.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.signal import welch

from .config import FIGURES_DIR, TEST_LABELS

# ── Style ──────────────────────────────────────────────────────────────────────
_PALETTE = sns.color_palette("tab10", n_colors=len(TEST_LABELS) + 1)
_TEST_COLOR = {lbl: _PALETTE[i] for i, lbl in enumerate(TEST_LABELS)}
_TEST_COLOR["REST"] = _PALETTE[len(TEST_LABELS)]


# ── ECG + ACC overview ─────────────────────────────────────────────────────────

def plot_ecg_acc_overview(
    ecg_signal: np.ndarray,
    acc_magnitude: np.ndarray,
    ecg_fs: float,
    acc_fs: float,
    test_segments: dict | None = None,
    output_path: Path | None = None,
    patient_id: str = "",
    max_duration_s: float = 600.0,
) -> None:
    """
    Salva una panoramica con ECG e modulo dell'ACC nello stesso intervallo.

    La figura affianca i due segnali per rendere immediato il confronto tra
    dinamica cardiaca e movimento: è utile soprattutto in fase esplorativa, dove
    si vuole capire se artefatti o cambi di task coincidono visivamente.
    """
    if output_path is None:
        output_path = FIGURES_DIR / f"overview_{patient_id}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_ecg = int(max_duration_s * ecg_fs)
    max_acc = int(max_duration_s * acc_fs)
    # Limitare la durata evita figure troppo dense: nelle overview conta la
    # leggibilità del trend globale più della fedeltà a ogni singolo campione.
    ecg_plot = ecg_signal[:max_ecg]
    acc_plot = acc_magnitude[:max_acc]

    t_ecg = np.arange(len(ecg_plot)) / ecg_fs
    t_acc = np.arange(len(acc_plot)) / acc_fs

    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=False)

    axes[0].plot(t_ecg, ecg_plot, lw=0.4, color="steelblue")
    axes[0].set_ylabel("ECG (mV)")
    axes[0].set_title(f"Patient {patient_id} — signal overview (first {max_duration_s:.0f} s)")

    axes[1].plot(t_acc, acc_plot, lw=0.4, color="darkorange")
    axes[1].set_ylabel("ACC |magnitude| (g)")
    axes[1].set_xlabel("Time (s)")

    # Evidenziare i segmenti con bande trasparenti permette di conservare il
    # segnale visibile sotto, che è più informativo di linee verticali isolate.
    if test_segments:
        patches = []
        for lbl, seg in test_segments.items():
            color = _TEST_COLOR.get(lbl, "grey")
            t_start_ecg = seg["ecg_start"] / ecg_fs
            t_end_ecg = seg["ecg_end"] / ecg_fs
            t_start_acc = seg["acc_start"] / acc_fs
            t_end_acc = seg["acc_end"] / acc_fs
            for ax, t_start, t_end in [
                (axes[0], t_start_ecg, t_end_ecg),
                (axes[1], t_start_acc, t_end_acc),
            ]:
                if t_start < max_duration_s:
                    ax.axvspan(t_start, min(t_end, max_duration_s),
                               alpha=0.15, color=color)
            patches.append(mpatches.Patch(color=color, label=lbl, alpha=0.4))
        axes[0].legend(handles=patches, loc="upper right", fontsize=7, ncol=3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── R-R interval time series ───────────────────────────────────────────────────

def plot_rr_timeseries(
    rr_ms: np.ndarray,
    rr_samples: np.ndarray,
    ecg_fs: float,
    test_segments: dict | None = None,
    output_path: Path | None = None,
    patient_id: str = "",
) -> None:
    """Salva la serie temporale degli intervalli R-R con le fasi del test.

    Questa vista è utile perché mostra direttamente dove la variabilità cambia
    nel tempo, senza comprimere tutto in un singolo indice riassuntivo.
    """
    if output_path is None:
        output_path = FIGURES_DIR / f"rr_timeseries_{patient_id}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_rr = rr_samples / ecg_fs

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t_rr, rr_ms, lw=0.8, color="crimson", zorder=3)
    ax.set_ylabel("R-R interval (ms)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Patient {patient_id} — R-R interval time series")

    if test_segments:
        patches = []
        for lbl, seg in test_segments.items():
            color = _TEST_COLOR.get(lbl, "grey")
            t_start = seg["ecg_start"] / ecg_fs
            t_end = seg["ecg_end"] / ecg_fs
            # Le aree colorate aiutano a collegare le variazioni HRV al contesto
            # sperimentale, che in tesi è spesso più importante del valore assoluto.
            ax.axvspan(t_start, t_end, alpha=0.2, color=color)
            patches.append(mpatches.Patch(color=color, label=lbl, alpha=0.4))
        ax.legend(handles=patches, loc="upper right", fontsize=7, ncol=3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── Feature box plots ─────────────────────────────────────────────────────────

def plot_feature_boxplots(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str = "test_label",
    output_dir: Path | None = None,
    max_per_fig: int = 6,
) -> None:
    """
    Salva box plot delle feature raggruppate per etichetta di test.

    I box plot sono adatti al confronto tra task perché mostrano insieme
    mediana, dispersione e outlier, quindi fanno emergere rapidamente feature
    potenzialmente discriminanti prima della modellazione.
    """
    if output_dir is None:
        output_dir = FIGURES_DIR / "boxplots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = [feature_cols[i: i + max_per_fig]
               for i in range(0, len(feature_cols), max_per_fig)]
    for batch_idx, batch in enumerate(batches):
        # Spezzare in batch evita pannelli troppo affollati, che renderebbero
        # difficile confrontare distribuzioni diverse in modo visivo affidabile.
        n = len(batch)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, batch):
            sub = df[[col, group_col]].dropna()
            order = sorted(sub[group_col].unique())
            # Si rimuovono i NaN solo per la feature corrente per non perdere
            # inutilmente informazione sulle altre variabili del dataframe.
            sns.boxplot(data=sub, x=group_col, y=col, order=order, ax=ax,
                        palette="tab10", fliersize=2)
            ax.set_title(col, fontsize=9)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=30, labelsize=7)
        plt.tight_layout()
        fig.savefig(output_dir / f"boxplots_batch{batch_idx:02d}.png", dpi=150)
        plt.close(fig)


# ── PSD plot ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: list[str],
    title: str = "",
    output_path: Path | None = None,
    normalize: bool = True,
) -> None:
    """
    Salva una confusion matrix come heatmap leggibile.

    La normalizzazione per riga è particolarmente utile in ambito clinico
    perché permette di confrontare il recall delle classi anche quando il
    dataset è sbilanciato.
    """
    if output_path is None:
        output_path = FIGURES_DIR / f"cm_{title.replace(' ', '_')}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # La divisione per riga mette in evidenza dove il modello perde sensibilità
        # su una classe specifica, non solo dove accumula errori in valore assoluto.
        cm_plot = np.where(row_sums == 0, 0, cm / row_sums)
        fmt = ".2f"
        vmax = 1.0
        cbar_label = "Recall (row-normalised)"
    else:
        cm_plot = cm
        fmt = "d"
        vmax = cm.max()
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=(max(5, len(label_names) * 1.4),
                                    max(4, len(label_names) * 1.2)))
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                vmin=0, vmax=vmax, linewidths=0.5, ax=ax,
                cbar_kws={"label": cbar_label})
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    title: str = "Feature Importance",
    top_n: int = 20,
    output_path: Path | None = None,
) -> None:
    """
    Salva un grafico a barre con le feature più importanti.

    Limitarsi alle top-N migliora l'interpretabilità: in tesi conta capire quali
    segnali guidano il modello, non mostrare una coda lunga di importanze minime.
    """
    if output_path is None:
        output_path = FIGURES_DIR / f"feat_imp_{title.replace(' ', '_')}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    idx = np.argsort(importances)[-top_n:]
    top_imp = importances[idx]
    top_names = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    bars = ax.barh(top_names, top_imp, color=sns.color_palette("Blues_r", top_n))
    ax.set_xlabel("Mean decrease in impurity", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="y", labelsize=8)
    # Le etichette numeriche aiutano il confronto fine tra feature vicine, cosa
    # utile quando si devono discutere risultati quantitativi nel testo della tesi.
    for bar, val in zip(bars, top_imp):
        ax.text(bar.get_width() + top_imp.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_f1_comparison(
    results_dict: dict,
    title: str = "F1-weighted comparison",
    output_path: Path | None = None,
) -> None:
    """
    Salva un confronto tra F1-weighted di più modelli e obiettivi.

    L'F1 pesato è una sintesi pratica quando le classi non sono bilanciate,
    perché evita che una buona accuratezza sulle classi maggioritarie nasconda
    prestazioni scarse su classi meno frequenti.
    """
    if output_path is None:
        output_path = FIGURES_DIR / "f1_comparison.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    objectives = list(results_dict.keys())
    models = list(next(iter(results_dict.values())).keys())
    x = np.arange(len(objectives))
    width = 0.25
    palette = sns.color_palette("Set2", len(models))

    fig, ax = plt.subplots(figsize=(max(7, len(objectives) * 2), 5))
    for i, (model, color) in enumerate(zip(models, palette)):
        f1_vals = [
            results_dict[obj].get(model, {}).get("report", {})
                .get("weighted avg", {}).get("f1-score", 0.0)
            for obj in objectives
        ]
        rects = ax.bar(x + i * width, f1_vals, width, label=model, color=color)
        for rect, val in zip(rects, f1_vals):
            # Annotare i valori evita di dover stimare a occhio differenze piccole
            # ma spesso rilevanti quando si confrontano modelli simili.
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x + width)
    ax.set_xticklabels(objectives, fontsize=10)
    ax.set_ylabel("F1-weighted", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_psd(
    signal: np.ndarray,
    fs: float,
    label: str = "",
    output_path: Path | None = None,
) -> None:
    """Salva la densità spettrale di potenza con il metodo di Welch.

    La PSD permette di capire dove si concentra l'energia del segnale; la scala
    logaritmica in ordinata rende leggibili insieme componenti forti e deboli.
    """
    if output_path is None:
        output_path = FIGURES_DIR / f"psd_{label}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    freqs, psd = welch(signal, fs=fs, nperseg=min(1024, len(signal)))
    fig, ax = plt.subplots(figsize=(8, 4))
    # La semilogy è preferibile perché le PSD fisiologiche coprono spesso più
    # ordini di grandezza e una scala lineare nasconderebbe le bande minori.
    ax.semilogy(freqs, psd, lw=1.2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title(f"Power Spectral Density — {label}")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
