"""
Funzioni di valutazione e visualizzazione dei risultati dei classificatori.

Questo modulo raccoglie le utility per trasformare l'output grezzo di
train_evaluate() in tabelle riepilogative, grafici delle matrici di confusione,
diagrammi di feature importance e curve ROC.
Separare la valutazione dalla logica di addestramento facilita il confronto
dei risultati tra obiettivi diversi (OBJ1, OBJ2, OBJ3) senza duplicare codice.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend non interattivo: necessario su server/CI senza display
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIGURES_DIR, TABLES_DIR


# ── Tabella riepilogativa ─────────────────────────────────────────────────────

def summarise_results(results_dict: dict) -> pd.DataFrame:
    """
    Converte l'output di train_evaluate() in un DataFrame compatto.

    Ogni riga rappresenta un modello (RF, SVM, LR) con le metriche aggregate:
    accuracy, F1 macro e F1 weighted. Il F1 weighted è la metrica principale
    perché tiene conto dello sbilanciamento delle classi (rilevante per NYHA).
    """
    rows = []
    for model_name, data in results_dict.items():
        rep = data["report"]
        rows.append({
            "model": model_name,
            "accuracy": rep.get("accuracy", float("nan")),
            "macro_f1": rep.get("macro avg", {}).get("f1-score", float("nan")),
            "weighted_f1": rep.get("weighted avg", {}).get("f1-score", float("nan")),
            "macro_precision": rep.get("macro avg", {}).get("precision", float("nan")),
            "macro_recall": rep.get("macro avg", {}).get("recall", float("nan")),
        })
    # Ordinamento per F1 weighted decrescente: il modello migliore appare prima
    return pd.DataFrame(rows).sort_values("weighted_f1", ascending=False).reset_index(drop=True)


def print_report(results_dict: dict, objective_name: str = "") -> None:
    """Stampa a schermo un riepilogo formattato di tutti i modelli."""
    print(f"\n{'='*60}")
    print(f" Risultati — {objective_name}")
    print(f"{'='*60}")
    df = summarise_results(results_dict)
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print()


# ── Matrici di confusione ──────────────────────────────────────────────────────

def save_confusion_matrices(
    results_dict: dict,
    output_dir: Path = FIGURES_DIR,
    prefix: str = "",
    label_names: list[str] | None = None,
) -> None:
    """
    Salva una matrice di confusione PNG per ciascun modello in results_dict.

    La matrice di confusione è lo strumento diagnostico principale per capire
    quali classi vengono confuse tra loro (es. NYHA II vs III, TUG vs GAIT).
    Utile per identificare pattern di errore sistematici non visibili dall'F1.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, data in results_dict.items():
        cm = data["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(max(6, cm.shape[0]), max(5, cm.shape[0] - 1)))
        xticklabels = label_names if label_names else "auto"
        yticklabels = label_names if label_names else "auto"
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=xticklabels, yticklabels=yticklabels,
        )
        ax.set_xlabel("Classe predetta")
        ax.set_ylabel("Classe reale")
        title = f"{prefix} {model_name} — Matrice di Confusione".strip()
        ax.set_title(title)
        plt.tight_layout()
        fname = f"{prefix}_{model_name}_cm.png".lstrip("_")
        fig.savefig(output_dir / fname, dpi=150)
        plt.close(fig)


# ── Feature importance ────────────────────────────────────────────────────────

def save_feature_importance(
    rf_pipeline,
    feature_cols: list[str] | None = None,
    output_path: Path | None = None,
    top_n: int = 25,
    feature_names: list[str] | None = None,  # alias per compatibilità
) -> pd.DataFrame:
    """
    Estrae e visualizza le feature importance dal Random Forest nella pipeline.

    L'importanza usata è la Mean Decrease in Impurity (indice di Gini):
    misura quanto ogni feature riduce l'impurità media nei nodi dell'albero.
    È veloce da calcolare ma può sovrastimare le feature ad alta cardinalità.
    Le top-N feature identificate guidano l'interpretazione clinica dei risultati.

    Restituisce un DataFrame con colonne (feature, importance) ordinato per
    importanza decrescente.
    """
    # Supporto a entrambe le varianti del parametro per retrocompatibilità
    if feature_cols is None and feature_names is not None:
        feature_cols = feature_names
    if output_path is None:
        output_path = FIGURES_DIR / "feature_importance.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Estrae il classificatore dalla pipeline scikit-learn
    rf = rf_pipeline.named_steps.get("classifier")
    if rf is None or not hasattr(rf, "feature_importances_"):
        return pd.DataFrame()

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n // 3)))
    sns.barplot(data=imp, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title(f"Top-{top_n} Feature Importance (Random Forest — Indice di Gini)")
    ax.set_xlabel("Riduzione media dell'impurità")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return imp


# ── Curve ROC (classificatori binari) ────────────────────────────────────────

def plot_roc_curves(
    results_dict: dict,
    y_true: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """
    Traccia le curve ROC per i classificatori binari.

    Richiede che results_dict contenga la chiave 'y_prob' (score di probabilità).
    Utile per OBJ2 (REST vs ACTIVITY) dove una soglia adattiva può migliorare
    le prestazioni rispetto alla soglia di default 0.5.
    """
    from sklearn.metrics import roc_curve, auc

    if output_path is None:
        output_path = FIGURES_DIR / "roc_curves.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    # Linea diagonale = classificatore casuale (AUC = 0.5) — riferimento
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Casuale (AUC=0.50)")
    for model_name, data in results_dict.items():
        y_prob = data.get("y_prob")
        if y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})", linewidth=1.5)

    ax.set_xlabel("Tasso di falsi positivi (1 - Specificità)")
    ax.set_ylabel("Tasso di veri positivi (Sensibilità)")
    ax.set_title("Curve ROC")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── Esportazione risultati ────────────────────────────────────────────────────

def save_results_table(
    results_dict: dict,
    output_path: Path | None = None,
    prefix: str = "",
) -> None:
    """Esporta la tabella riepilogativa delle metriche in formato CSV."""
    if output_path is None:
        output_path = TABLES_DIR / f"{prefix}_summary.csv".lstrip("_")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summarise_results(results_dict).to_csv(output_path, index=False)

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIGURES_DIR, TABLES_DIR


# ── Summary table ─────────────────────────────────────────────────────────────

def summarise_results(results_dict: dict) -> pd.DataFrame:
    """
    Convert train_evaluate() output into a tidy summary DataFrame.

    Each row = one model, with accuracy, macro-F1, weighted-F1.
    """
    rows = []
    for model_name, data in results_dict.items():
        rep = data["report"]
        rows.append({
            "model": model_name,
            "accuracy": rep.get("accuracy", float("nan")),
            "macro_f1": rep.get("macro avg", {}).get("f1-score", float("nan")),
            "weighted_f1": rep.get("weighted avg", {}).get("f1-score", float("nan")),
            "macro_precision": rep.get("macro avg", {}).get("precision", float("nan")),
            "macro_recall": rep.get("macro avg", {}).get("recall", float("nan")),
        })
    return pd.DataFrame(rows).sort_values("weighted_f1", ascending=False).reset_index(drop=True)


def print_report(results_dict: dict, objective_name: str = "") -> None:
    """Print a formatted summary of all models."""
    print(f"\n{'='*60}")
    print(f" Results — {objective_name}")
    print(f"{'='*60}")
    df = summarise_results(results_dict)
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print()


# ── Confusion matrix plots ─────────────────────────────────────────────────────

def save_confusion_matrices(
    results_dict: dict,
    output_dir: Path = FIGURES_DIR,
    prefix: str = "",
    label_names: list[str] | None = None,
) -> None:
    """Save one confusion matrix PNG per model in *results_dict*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, data in results_dict.items():
        cm = data["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(max(6, cm.shape[0]), max(5, cm.shape[0] - 1)))
        xticklabels = label_names if label_names else "auto"
        yticklabels = label_names if label_names else "auto"
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=xticklabels, yticklabels=yticklabels,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        title = f"{prefix} {model_name} — Confusion Matrix".strip()
        ax.set_title(title)
        plt.tight_layout()
        fname = f"{prefix}_{model_name}_cm.png".lstrip("_")
        fig.savefig(output_dir / fname, dpi=150)
        plt.close(fig)


# ── Feature importance ────────────────────────────────────────────────────────

def save_feature_importance(
    rf_pipeline,
    feature_cols: list[str] | None = None,
    output_path: Path | None = None,
    top_n: int = 25,
    feature_names: list[str] | None = None,  # alias for feature_cols
) -> pd.DataFrame:
    """
    Extract and plot feature importances from the Random Forest estimator
    inside a fitted Pipeline.

    Returns the importance DataFrame (feature, importance).
    """
    # Support both 'feature_cols' and 'feature_names' as parameter name
    if feature_cols is None and feature_names is not None:
        feature_cols = feature_names
    if output_path is None:
        output_path = FIGURES_DIR / "feature_importance.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rf = rf_pipeline.named_steps.get("classifier")
    if rf is None or not hasattr(rf, "feature_importances_"):
        return pd.DataFrame()

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n // 3)))
    sns.barplot(data=imp, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title(f"Top-{top_n} Feature Importances (Random Forest)")
    ax.set_xlabel("Mean Decrease in Impurity")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return imp


# ── ROC / AUC (binary) ────────────────────────────────────────────────────────

def plot_roc_curves(
    results_dict: dict,
    y_true: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """
    Plot ROC curves for binary classifiers.

    Requires that results_dict contains 'y_prob' key (probability scores).
    """
    from sklearn.metrics import roc_curve, auc

    if output_path is None:
        output_path = FIGURES_DIR / "roc_curves.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    for model_name, data in results_dict.items():
        y_prob = data.get("y_prob")
        if y_prob is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})", linewidth=1.5)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── Save table ────────────────────────────────────────────────────────────────

def save_results_table(
    results_dict: dict,
    output_path: Path | None = None,
    prefix: str = "",
) -> None:
    """Export the summary metrics table to CSV."""
    if output_path is None:
        output_path = TABLES_DIR / f"{prefix}_summary.csv".lstrip("_")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summarise_results(results_dict).to_csv(output_path, index=False)
