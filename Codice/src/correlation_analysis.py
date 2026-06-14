"""
Analisi di correlazione tra feature accelerometriche (ACC) e intervalli R-R ECG.

Il modulo raccoglie funzioni per quantificare l'associazione tra segnali,
confrontare condizioni sperimentali e produrre visualizzazioni riassuntive.
L'idea metodologica è combinare misure parametriche e non parametriche, così
da ottenere conclusioni più robuste rispetto a normalità imperfetta e outlier,
situazioni frequenti nei dati fisiologici reali.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIGURES_DIR, TEST_LABELS


# ── Pairwise correlations ─────────────────────────────────────────────────────

def compute_correlations(
    df: pd.DataFrame,
    ecg_col: str = "ecg_mean_rr",
    acc_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Calcola correlazioni di Pearson e Spearman tra *ecg_col* e ciascuna
    colonna in *acc_cols*.

    Usare entrambe è utile perché Pearson misura associazioni lineari,
    mentre Spearman è più robusta a outlier e non richiede una relazione
    strettamente lineare: la lettura congiunta rende l'interpretazione più
    affidabile su feature biometriche spesso rumorose.

    Returns
    -------
    pd.DataFrame con colonne:
        feature, pearson_r, pearson_p, spearman_r, spearman_p, n
    """
    if acc_cols is None:
        acc_cols = [c for c in df.columns if c.startswith("acc_")]

    results = []
    for col in acc_cols:
        sub = df[[ecg_col, col]].dropna()
        if len(sub) < 5:
            # Con campioni troppo piccoli il coefficiente sarebbe instabile
            # e poco difendibile in un'analisi statistica di tesi.
            continue
        x, y = sub[ecg_col].values, sub[col].values
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        results.append({
            "feature": col,
            "pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp,
            "n": len(sub),
        })

    # L'ordinamento per valore assoluto mette in alto le associazioni più forti
    # indipendentemente dal segno: in questa fase interessa l'intensità del legame.
    result_df = pd.DataFrame(results).sort_values("pearson_r", key=abs, ascending=False)
    return result_df.reset_index(drop=True)


def correlation_matrix(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """
    Restituisce la matrice completa di correlazione di Pearson per *cols*.

    La matrice è utile come vista globale esplorativa: permette di individuare
    ridondanze tra feature prima di eventuali passi di selezione o modellazione.
    """
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()
    return df[cols].corr(method="pearson")


# ── Linear regression ─────────────────────────────────────────────────────────

def linear_regression_summary(
    df: pd.DataFrame,
    target_col: str = "ecg_mean_rr",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Stima una regressione lineare semplice OLS per ogni feature rispetto
    a *target_col*.

    Returns
    -------
    pd.DataFrame con colonne:
        feature, slope, intercept, r_squared, p_value, std_err, n
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.startswith("acc_")]

    rows = []
    for col in feature_cols:
        sub = df[[target_col, col]].dropna()
        if len(sub) < 5:
            continue
        x, y = sub[col].values, sub[target_col].values
        # Una feature costante non porta informazione discriminante e renderebbe
        # ill-posed la regressione; scartarla evita errori e risultati fuorvianti.
        if np.amax(x) == np.amin(x):
            continue
        result = stats.linregress(x, y)
        rows.append({
            "feature": col,
            "slope": result.slope,
            "intercept": result.intercept,
            "r_squared": result.rvalue**2,
            "p_value": result.pvalue,
            "std_err": result.stderr,
            "n": len(sub),
        })

    return pd.DataFrame(rows).sort_values("r_squared", ascending=False).reset_index(drop=True)


# ── Rest vs Activity comparison ───────────────────────────────────────────────

def compare_rest_vs_activity(
    df_rest: pd.DataFrame,
    df_activity: pd.DataFrame,
    col: str,
) -> dict:
    """
    Confronta la distribuzione di *col* tra finestre di riposo e di attività.

    Vengono riportati sia il t-test di Welch sia il Mann-Whitney U: il primo
    è sensibile a differenze tra medie in un quadro parametrico, il secondo
    è la controparte non parametrica più adatta quando le feature HRV non sono
    gaussiane o presentano outlier. Usarli insieme rende il confronto più solido.

    Returns
    -------
    dict con chiavi: col, n_rest, n_activity, mean_rest, mean_activity,
                     t_stat, t_p, mw_stat, mw_p
    """
    a = df_rest[col].dropna().values
    b = df_activity[col].dropna().values
    if len(a) < 3 or len(b) < 3:
        return {}
    # equal_var=False applica il test di Welch, preferibile quando la varianza
    # tra condizioni fisiologiche può differire in modo sostanziale.
    t_stat, t_p = ttest_ind(a, b, equal_var=False)
    mw_stat, mw_p = mannwhitneyu(a, b, alternative="two-sided")
    return {
        "col": col,
        "n_rest": len(a), "n_activity": len(b),
        "mean_rest": float(np.mean(a)), "mean_activity": float(np.mean(b)),
        "delta_mean": float(np.mean(b) - np.mean(a)),
        "t_stat": float(t_stat), "t_p": float(t_p),
        "mw_stat": float(mw_stat), "mw_p": float(mw_p),
    }


def activity_vs_rr_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per ogni test motorio confronta ecg_mean_rr rispetto alle finestre REST.

    Si assume che il DataFrame contenga la colonna 'test_label' e che REST
    rappresenti il baseline di riferimento. Questo confronto serve a verificare
    se lo sforzo indotto dai diversi protocolli modifica il ritmo cardiaco medio.

    Returns
    -------
    Un DataFrame riassuntivo dei confronti.
    """
    if "REST" not in df["test_label"].unique():
        return pd.DataFrame()

    rest_df = df[df["test_label"] == "REST"]
    rows = []
    for lbl in TEST_LABELS:
        act_df = df[df["test_label"] == lbl]
        if len(act_df) < 3:
            continue
        row = compare_rest_vs_activity(rest_df, act_df, "ecg_mean_rr")
        if row:
            row["test_label"] = lbl
            rows.append(row)
    return pd.DataFrame(rows)


# ── Visualisation helpers ─────────────────────────────────────────────────────

def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: Path | None = None,
    title: str = "Feature Correlation Matrix",
    figsize: tuple = (16, 14),
) -> None:
    """
    Salva una heatmap Seaborn di *corr_matrix* in *output_path*.

    La visualizzazione della sola metà utile della matrice migliora la leggibilità
    ed evita di ripetere due volte la stessa informazione simmetrica.
    """
    fig, ax = plt.subplots(figsize=figsize)
    # Mascherare il triangolo superiore concentra l'attenzione sui pattern reali
    # invece che su duplicazioni visive inevitabili nelle matrici di correlazione.
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix, mask=mask, annot=False, fmt=".2f",
        cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        linewidths=0.3, ax=ax,
    )
    ax.set_title(title, fontsize=14, pad=12)
    plt.tight_layout()
    if output_path is None:
        output_path = FIGURES_DIR / "correlation_heatmap.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_scatter_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = "ecg_mean_rr",
    hue_col: str = "test_label",
    output_path: Path | None = None,
) -> None:
    """
    Crea uno scatter plot con retta di regressione e colori per *hue_col*.

    Distinguere i punti per test_label permette di capire se la relazione globale
    è omogenea oppure trainata da uno specifico contesto sperimentale.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df[[x_col, y_col, hue_col]].dropna()
    palette = sns.color_palette("tab10", n_colors=sub[hue_col].nunique())
    for i, (lbl, grp) in enumerate(sub.groupby(hue_col)):
        ax.scatter(grp[x_col], grp[y_col], label=lbl, alpha=0.5, s=15,
                   color=palette[i % len(palette)])
    # La regressione complessiva fornisce una sintesi immediata del trend medio,
    # anche se i colori restano fondamentali per intercettare eventuali sottogruppi.
    if len(sub) > 4:
        m, b, *_ = stats.linregress(sub[x_col].values, sub[y_col].values)
        xs = np.linspace(sub[x_col].min(), sub[x_col].max(), 200)
        ax.plot(xs, m * xs + b, "k--", linewidth=1.5, label="Regression")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"{y_col}  vs  {x_col}")
    plt.tight_layout()
    if output_path is None:
        output_path = FIGURES_DIR / f"scatter_{x_col}_vs_{y_col}.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
