"""
Modelli di classificazione per tre obiettivi della tesi:
  1. Tipo di attività  (STAIR / 6MWT / TUG / VELO / GAIT_ANALYSIS)
  2. Livello di sforzo (REST vs ACTIVITY — problema binario)
  3. Stato clinico     (classe NYHA oppure livello di fragilità EFS)

Ogni obiettivo usa una pipeline dedicata che prepara le feature, addestra
più classificatori e ne valuta la capacità di generalizzazione tramite
cross-validation stratificata. La scelta di confrontare Random Forest, SVM
e Logistic Regression permette di mettere a paragone un modello non lineare,
uno a margine massimo e un baseline lineare facilmente interpretabile.
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, StratifiedGroupKFold, cross_val_predict,
    GridSearchCV,
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from .config import MODELS_DIR, TEST_LABELS

# Sopra questa soglia il kernel RBF diventa costoso in memoria e tempo:
# LinearSVC mantiene un baseline competitivo ma scala meglio su molte finestre.
_SVM_LINEAR_THRESHOLD = 10_000

# ── Hyperparameter grids ───────────────────────────────────────────────────────
RF_PARAM_GRID = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
}

SVM_PARAM_GRID = {
    "classifier__C": [0.1, 1, 10],
    "classifier__gamma": ["scale", "auto"],
}

LR_PARAM_GRID = {
    "classifier__C": [0.01, 0.1, 1, 10],
}


# ── Helper functions ──────────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estrae X (feature), y (etichette) e groups (ID paziente) dal DataFrame.

    Le righe senza target vengono eliminate perché non contribuirebbero
    all'addestramento supervisionato. I valori mancanti delle feature non
    vengono rimossi qui: si preferisce imputarli nella pipeline con la mediana,
    così da non perdere finestre utili e da ridurre l'effetto di outlier
    tipico delle misure fisiologiche.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,) — etichette intere
    groups : np.ndarray — array di patient_id per split group-aware
    """
    sub = df[feature_cols + [target_col, "patient_id"]].dropna(subset=[target_col])
    X = sub[feature_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(sub[target_col].astype(str))
    groups = sub["patient_id"].values
    return X, y, groups, le


def _make_pipeline(classifier, use_smote: bool = False) -> Pipeline:
    """
    Costruisce la pipeline Imputer → Scaler → (SMOTE →) Classifier.

    Standardizzare prima del classificatore è importante soprattutto per
    SVM e regressione logistica: senza scaling, feature con scale diverse
    dominerebbero il modello più per unità di misura che per contenuto clinico.
    """
    steps = [
        # La mediana è più robusta della media quando le feature HRV/ACC
        # presentano code o valori anomali dovuti a artefatti di acquisizione.
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ]
    if use_smote:
        # SMOTE viene applicato solo quando richiesto: serve soprattutto per
        # target clinici sbilanciati, evitando che il modello impari quasi solo
        # le classi più frequenti.
        steps.insert(2, ("smote", SMOTE(random_state=42)))
        return ImbPipeline(steps)
    return Pipeline(steps)


def _cross_val_report(
    pipe,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_names: list[str],
    cv_folds: int = 3,
    group_aware: bool = False,
) -> dict:
    """
    Esegue la cross-validation e restituisce report di classificazione
    e matrice di confusione.

    Parameters
    ----------
    group_aware : se True usa StratifiedGroupKFold, così tutte le finestre
                  dello stesso paziente restano nello stesso split. Questo
                  evita data leakage, particolarmente critico quando il target
                  è clinico e definito a livello paziente (es. NYHA).
    """
    if group_aware:
        cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=cv, n_jobs=1)
    else:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        # Su Windows è più prudente evitare parallelismo annidato durante la CV:
        # la stabilità sperimentale conta più di un piccolo guadagno di velocità.
        y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)
    report = classification_report(y, y_pred, target_names=label_names,
                                    output_dict=True, zero_division=0)
    cm = confusion_matrix(y, y_pred)
    return {"report": report, "confusion_matrix": cm, "y_pred": y_pred, "y_true": y}


# ── Main training function ─────────────────────────────────────────────────────

def train_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_encoder: LabelEncoder,
    use_smote: bool = False,
    cv_folds: int = 3,
    tune_hyperparams: bool = False,
    group_aware: bool = False,
) -> dict:
    """
    Addestra Random Forest, SVM e Logistic Regression con cross-validation.

    Parameters
    ----------
    group_aware : se True usa StratifiedGroupKFold; è una scelta necessaria
                  quando il target è legato al paziente, perché la vera
                  generalizzazione va misurata su pazienti non visti.
    """
    label_names = list(label_encoder.classes_)
    n_samples = len(y)
    large_dataset = n_samples > _SVM_LINEAR_THRESHOLD

    if large_dataset:
        print(f"  [info] n={n_samples}: using LinearSVC (faster than RBF SVM on large data)")
        # Per molte finestre un'SVM con kernel RBF tende a scalare circa O(n^2)
        # in memoria; LinearSVC è più adatta quando conta la scalabilità.
        svm_clf = LinearSVC(max_iter=2000, random_state=42, class_weight="balanced", dual="auto")
        svm_grid = {"classifier__C": [0.01, 0.1, 1, 10]}
    else:
        svm_clf = SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced")
        svm_grid = SVM_PARAM_GRID

    classifiers = {
        "RandomForest": (
            # I pesi bilanciati compensano squilibri di classe senza alterare i dati
            # originali; è utile soprattutto per classi cliniche rare.
            # n_jobs=-1: parallelizzare gli alberi è sicuro perché la CV esterna
            # rimane seriale, evitando problemi di parallelismo annidato.
            RandomForestClassifier(n_estimators=200, random_state=42,
                                   class_weight="balanced", n_jobs=-1),
            RF_PARAM_GRID,
        ),
        "SVM": (svm_clf, svm_grid),
        "LogisticRegression": (
            # Anche qui class_weight='balanced' riduce il bias verso la classe
            # maggioritaria, rendendo più affidabile il confronto tra modelli.
            LogisticRegression(max_iter=1000, random_state=42,
                               class_weight="balanced"),
            LR_PARAM_GRID,
        ),
    }

    results = {}
    for name, (clf, param_grid) in classifiers.items():
        pipe = _make_pipeline(clf, use_smote=use_smote)

        best_params = None
        if tune_hyperparams:
            cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
            gs = GridSearchCV(pipe, param_grid, cv=cv_inner, scoring="f1_weighted",
                               n_jobs=1, refit=True, error_score="raise")
            gs.fit(X, y)
            pipe = gs.best_estimator_
            best_params = gs.best_params_

        report_data = _cross_val_report(pipe, X, y, groups, label_names,
                                        cv_folds, group_aware=group_aware)
        report_data["best_params"] = best_params
        results[name] = report_data
        print(f"  [{name}] F1-weighted: "
              f"{report_data['report']['weighted avg']['f1-score']:.3f}")

    return results


# ── Objective-specific pipelines ───────────────────────────────────────────────

def classify_activity_type(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> dict:
    """
    Obiettivo 1: classificazione multiclasse del tipo di test a partire da
    feature ACC + ECG.

    Si mantengono solo le etichette note di TEST_LABELS, perché includere
    segmenti ambigui o fuori protocollo introdurrebbe rumore supervisionato.
    """
    df_clean = df[df["test_label"].isin(TEST_LABELS)].copy()
    if feature_cols is None:
        feature_cols = [c for c in df_clean.columns
                        if c.startswith("ecg_") or c.startswith("acc_")]
    X, y, groups, le = build_feature_matrix(df_clean, feature_cols, "test_label")
    print(f"[Objective 1 — Activity Type] samples={len(y)}, classes={le.classes_}")
    return train_evaluate(X, y, groups, le)


def classify_effort_level(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> dict:
    """
    Obiettivo 2: classificazione binaria REST vs ACTIVITY.

    Se la colonna 'is_activity' non esiste, viene ricostruita dal test_label:
    è una scelta coerente con il protocollo sperimentale, dove i test motori
    rappresentano sforzo e REST il baseline fisiologico.
    """
    df_work = df.copy()
    if "is_activity" not in df_work.columns:
        df_work["is_activity"] = df_work["test_label"].isin(TEST_LABELS).astype(int)
    df_work = df_work[df_work["is_activity"].notna()]

    if feature_cols is None:
        feature_cols = [c for c in df_work.columns
                        if c.startswith("ecg_") or c.startswith("acc_")]
    X, y, groups, le = build_feature_matrix(df_work, feature_cols, "is_activity")
    print(f"[Objective 2 — Effort Level] samples={len(y)}, classes={le.classes_}")
    return train_evaluate(X, y, groups, le)


def classify_clinical_state(
    df: pd.DataFrame,
    target_col: str = "nyha",
    feature_cols: list[str] | None = None,
    threshold: float | None = None,
    use_smote: bool = True,
) -> dict:
    """
    Obiettivo 3: classificazione dello stato clinico (es. classe NYHA o EFS).

    Se *threshold* è valorizzato, il target numerico viene binarizzato:
    0 se valore <= threshold, 1 se valore > threshold. Questa opzione è utile
    quando interessa distinguere rischio basso/alto più che stimare tutte
    le categorie ordinali.

    Parameters
    ----------
    use_smote : abilita oversampling delle classi minoritarie. Per target come
                NYHA è sensato, perché le classi estreme sono spesso rare;
                per OBJ1 e OBJ2 in genere il bilanciamento è meno critico.
    """
    df_work = df.copy()
    if threshold is not None and target_col in df_work.columns:
        df_work[target_col] = (df_work[target_col] > threshold).astype(int)

    if feature_cols is None:
        feature_cols = [c for c in df_work.columns
                        if c.startswith("ecg_") or c.startswith("acc_")
                        or c in ("age", "gender", "bmi", "days_post_surgery")]
    X, y, groups, le = build_feature_matrix(df_work, feature_cols, target_col)
    print(f"[Objective 3 — Clinical ({target_col})] samples={len(y)}, classes={le.classes_}")
    # La valutazione group-aware è obbligatoria: finestre dello stesso paziente
    # in train e test darebbero una stima troppo ottimistica delle performance.
    return train_evaluate(X, y, groups, le, use_smote=use_smote, group_aware=True)


def classify_clinical_state_patient_level(
    df: pd.DataFrame,
    target_col: str = "nyha",
    feature_cols: list[str] | None = None,
    threshold: float | None = None,
    agg_funcs: tuple[str, ...] = ("mean", "std", "median"),
) -> dict:
    """
    Variante dell'obiettivo 3: aggregazione patient-level prima della
    classificazione.

    Poiché NYHA è un'etichetta del paziente e non della singola finestra,
    aggregare le feature per paziente è metodologicamente più corretto:
    si riduce la ridondanza tra finestre dello stesso soggetto e non si gonfia
    artificialmente la numerosità campionaria effettiva.

    Parameters
    ----------
    agg_funcs : funzioni di aggregazione applicate alle feature per paziente.
                Default: mean + std + median, così si catturano livello medio,
                variabilità e robustezza centrale del segnale.

    Returns
    -------
    Stessa struttura di train_evaluate(), ma con LOOCV
    (Leave-One-Patient-Out).
    """
    from sklearn.model_selection import LeaveOneGroupOut

    df_work = df.copy()
    if threshold is not None and target_col in df_work.columns:
        df_work[target_col] = (df_work[target_col] > threshold).astype(int)

    if feature_cols is None:
        feature_cols = [c for c in df_work.columns
                        if c.startswith("ecg_") or c.startswith("acc_")
                        or c in ("age", "gender", "bmi", "days_post_surgery")]

    # Senza target clinico la finestra non è utilizzabile per apprendimento supervisionato.
    df_work = df_work.dropna(subset=[target_col])

    # L'aggregazione per paziente allinea l'unità statistica dell'input
    # con l'unità clinica dell'etichetta.
    valid_feat_cols = [col for col in feature_cols if col in df_work.columns]
    agg_dict = {col: list(agg_funcs) for col in valid_feat_cols}
    feat_df = df_work.groupby("patient_id").agg(agg_dict)
    feat_df.columns = ["_".join(c) for c in feat_df.columns]

    # Il target è unico per paziente: prendere il primo valore non nullo
    # evita di duplicare informazione già costante a livello soggetto.
    target_series = df_work.groupby("patient_id")[target_col].first()

    patient_df = feat_df.join(target_series)
    agg_feat_cols = [c for c in patient_df.columns if c != target_col]

    X = patient_df[agg_feat_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(patient_df[target_col].astype(str))
    groups = np.arange(len(y))  # ogni paziente è il proprio gruppo: necessario per LOOCV
    label_names = list(le.classes_)
    n_patients = len(y)

    print(f"[Objective 3 — Patient-level ({target_col})] "
          f"patients={n_patients}, classes={le.classes_}, "
          f"features={len(agg_feat_cols)}")

    # Con pochi pazienti la LOOCV sfrutta quasi tutti i dati a ogni iterazione
    # ed è una delle poche strategie che mantiene una stima di generalizzazione
    # ancora informativa senza sprecare campioni.
    logo = LeaveOneGroupOut()
    results = {}

    classifiers_cfg = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"),
    }

    for name, clf in classifiers_cfg.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])
        y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=logo, n_jobs=1)
        report = classification_report(y, y_pred, target_names=label_names,
                                        output_dict=True, zero_division=0)
        cm = confusion_matrix(y, y_pred)
        f1 = report["weighted avg"]["f1-score"]
        print(f"  [{name}] F1-weighted (LOOCV): {f1:.3f}")
        results[name] = {"report": report, "confusion_matrix": cm,
                         "y_pred": y_pred, "y_true": y, "best_params": None}

    return results


# ── Model persistence ─────────────────────────────────────────────────────────

def save_model(model, name: str, output_dir: Path = MODELS_DIR) -> Path:
    """Salva un modello già addestrato in formato pickle dentro *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(path: Path):
    """Carica da disco un modello serializzato con pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)
