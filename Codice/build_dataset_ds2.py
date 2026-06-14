"""
Build the DS2 feature dataset (SCG-RHC Wearable Seismocardiogram) and save to CSV.

Each record (~10 min, 500 Hz) is processed as a single "RHC" segment.
Features: same HRV + ACC pipeline as DS (src/feature_extraction.py).
Labels added per window: cdecomp, nyhac, age, gender, bmi, sbp, dbp.

Usage:
    python build_dataset_ds2.py [--workers N] [--force]

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

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data_loader_ds2 import list_records_ds2, load_record_ds2, DS2_FS
from src.preprocessing import preprocess_ecg, preprocess_acc
from src.feature_extraction import extract_all_features
from src.config import TABLES_DIR

OUTPUT_CSV = TABLES_DIR / "features_ds2.csv"


def _process_one(record_id: str) -> pd.DataFrame | None:
    """Process one DS2 record — returns a DataFrame or None on error."""
    try:
        rec = load_record_ds2(record_id)

        # Skip records with missing ECG or unlabelled CDecomp
        if rec["ecg"] is None:
            print(f"  [SKIP] {record_id}: no patch_ECG channel", flush=True)
            return None
        if rec["cdecomp"] < 0:
            print(f"  [SKIP] {record_id}: CDecomp unknown", flush=True)
            return None

        # Fall back to zero arrays for missing ACC axes
        n = len(rec["ecg"])
        acc_lat = rec["acc_lat"] if rec["acc_lat"] is not None else np.zeros(n, np.float32)
        acc_hf  = rec["acc_hf"]  if rec["acc_hf"]  is not None else np.zeros(n, np.float32)
        acc_dv  = rec["acc_dv"]  if rec["acc_dv"]  is not None else np.zeros(n, np.float32)

        # Preprocess — both ECG and ACC at DS2_FS = 500 Hz
        ep = preprocess_ecg(rec["ecg"], fs=DS2_FS)
        ap = preprocess_acc(acc_lat, acc_hf, acc_dv, fs=DS2_FS)

        # Single segment = whole recording (no motor tests — continuous RHC session)
        segments = {
            "RHC": {
                "ecg_start": 0,
                "ecg_end":   n,
                "acc_start": 0,
                "acc_end":   n,
            }
        }

        df = extract_all_features(
            record_id, 1, ep, ap, segments, subject_row=None
        )
        if len(df) == 0:
            return None

        # ── Attach DS2 clinical labels ─────────────────────────────────────────
        meta = rec["meta"]
        df["cdecomp"]      = rec["cdecomp"]
        df["nyhac"]        = rec["nyhac"]
        df["fine_align"]   = int(rec["fine_align"])

        height = meta.get("height", None)
        weight = meta.get("weight", None)
        df["age"]    = meta.get("age",    float("nan"))
        df["gender"] = 1 if str(meta.get("gender", "")).lower().startswith("m") else 0
        df["height"] = float(height) if height is not None else float("nan")
        df["weight"] = float(weight) if weight is not None else float("nan")
        if height and weight and float(height) > 0:
            df["bmi"] = float(weight) / (float(height) / 100.0) ** 2
        else:
            df["bmi"] = float("nan")
        df["sbp"] = meta.get("sbp", float("nan"))
        df["dbp"] = meta.get("dbp", float("nan"))

        return df

    except Exception as ex:
        print(f"  [SKIP] {record_id}: {ex}", flush=True)
        return None


def build(n_workers: int = -1, force: bool = False) -> pd.DataFrame:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_CSV.exists() and not force:
        print(f"Dataset DS2 già presente: {OUTPUT_CSV}  (usa --force per ricostruire)")
        return pd.read_csv(OUTPUT_CSV)

    records = list_records_ds2()
    print(f"Record DS2: {len(records)}   workers: {n_workers}")

    t0 = time.time()
    results = Parallel(n_jobs=n_workers, verbose=5)(
        delayed(_process_one)(rid) for rid in records
    )
    elapsed = time.time() - t0

    dfs = [r for r in results if r is not None]
    if not dfs:
        print("Nessun record processato con successo.")
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nDataset DS2 salvato: {OUTPUT_CSV}")
    print(f"Shape: {full_df.shape}  |  Tempo: {elapsed:.1f}s")
    print(f"Record unici: {full_df['patient_id'].nunique()}")
    print(f"Distribuzione CDecomp:\n{full_df['cdecomp'].value_counts().to_string()}")
    return full_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=-1,
                        help="Numero di worker paralleli (-1 = tutti i core)")
    parser.add_argument("--force", action="store_true",
                        help="Ricostruisce anche se il CSV esiste già")
    args = parser.parse_args()
    build(n_workers=args.workers, force=args.force)
