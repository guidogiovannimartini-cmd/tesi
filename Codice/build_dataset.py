"""
Build the full feature dataset for all patients in parallel and save to CSV.

Usage:
    python build_dataset.py [--workers N] [--force]

Options:
    --workers N   Number of parallel workers (default: all CPU cores)
    --force       Recompute even if the output CSV already exists
"""

import argparse
import sys
import warnings
import time
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from joblib import Parallel, delayed

from src.data_loader import (
    list_patients, get_patient_sessions,
    load_patient, get_test_segments, load_subject_info,
)
from src.preprocessing import preprocess_ecg, preprocess_acc
from src.feature_extraction import extract_all_features
from src.config import TABLES_DIR

OUTPUT_CSV = TABLES_DIR / "features_all.csv"


def _process_one(pid: str, session: int, subject_info: pd.DataFrame) -> pd.DataFrame | None:
    """Process one patient/session — returns a DataFrame or None on error."""
    try:
        d = load_patient(pid, session)
        e = d["ecg"]; a = d["acc"]
        ep = preprocess_ecg(e["signal"], e["fs"])
        ap = preprocess_acc(a["signal_x"], a["signal_y"], a["signal_z"], a["fs"])
        segs = get_test_segments(
            e["ann_samples"], e["ann_labels"],
            a["ann_samples"], a["ann_labels"],
            len(e["signal"]), len(a["signal_x"]),
        )
        srow = subject_info.loc[pid] if pid in subject_info.index else None
        df = extract_all_features(pid, session, ep, ap, segs, srow)
        return df if len(df) > 0 else None
    except Exception as ex:
        print(f"  [SKIP] {pid}_{session}: {ex}", flush=True)
        return None


def build(n_workers: int = -1, force: bool = False) -> pd.DataFrame:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_CSV.exists() and not force:
        print(f"Dataset già presente: {OUTPUT_CSV}  (usa --force per ricostruire)")
        return pd.read_csv(OUTPUT_CSV)

    subject_info = load_subject_info()
    patients = list_patients()
    jobs = [(pid, sess) for pid in patients for sess in get_patient_sessions(pid)]
    print(f"Pazienti: {len(patients)}  sessioni totali: {len(jobs)}  workers: {n_workers}")

    t0 = time.time()
    results = Parallel(n_jobs=n_workers, verbose=5)(
        delayed(_process_one)(pid, sess, subject_info) for pid, sess in jobs
    )
    elapsed = time.time() - t0

    dfs = [r for r in results if r is not None]
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDataset salvato: {OUTPUT_CSV}")
    print(f"Shape: {full_df.shape}  |  Tempo: {elapsed:.1f}s")
    print(f"Finestre per test:\n{full_df['test_label'].value_counts().to_string()}")
    return full_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=-1,
                        help="Numero di worker paralleli (-1 = tutti i core)")
    parser.add_argument("--force", action="store_true",
                        help="Ricostruisce anche se il CSV esiste già")
    args = parser.parse_args()
    build(n_workers=args.workers, force=args.force)
