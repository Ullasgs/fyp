# fake_stream_and_train.py
"""
Generate fake continuous sensor streams (ESP32-like), build a labeled synthetic dataset
from the provided plant ranges (Excel or parsed CSV), train a classifier (Random Forest)
to predict the best plant for a reading, and demonstrate predictions on streaming fake values.

Outputs:
 - prints train/validation accuracy and top-3 deterministic vs ML comparison for streaming samples
 - saves trained model to ./trained_rf.pkl
 - saves synthetic dataset to ./synthetic_dataset.csv

Usage:
  python3 fake_stream_and_train.py --excel /mnt/data/Hydroponics_Plant_Range.xlsx
or
  python3 fake_stream_and_train.py --csv /mnt/data/parsed_plant_metadata_v2.csv

Dependencies (install on Pi):
  python3 -m pip install pandas scikit-learn joblib openpyxl numpy

Notes:
 - This script uses the deterministic range-based scorer to label the synthetic samples:
     for each synthetic reading we compute the deterministic scores for all plants and set the
     label to the top-scoring plant (ties broken arbitrarily).
 - The ML model learns to predict that top plant. This gives a fast classifier suitable for
   low-latency inference on the Pi. You can keep using the deterministic recommender alongside
   the ML model for explainability.
"""

from __future__ import annotations
import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score
from joblib import dump, load

# ---------------------------
# Utilities: parse range strings and build metadata
# ---------------------------
_num_re = re.compile(r"[-+]?\d*\.?\d+")

def parse_range_string(s: Any) -> Tuple[float, float]:
    """Parse '5.5-6.5' or '6.0' or '400-700' → (low, high)"""
    if pd.isna(s):
        raise ValueError("empty")
    s_str = str(s).strip()
    nums = _num_re.findall(s_str)
    if not nums:
        raise ValueError(f"no numbers in '{s}'")
    if len(nums) == 1:
        v = float(nums[0])
        return v, v
    low = float(nums[0]); high = float(nums[1])
    if low > high:
        low, high = high, low
    return low, high

# Map common sheet column names to canonical feature names
_DEFAULT_COL_MAP = {
    'ideal_ph': 'ph', 'ph': 'ph',
    'ec (mS/cm)': 'ec', 'ec': 'ec',
    'tds(ppm)': 'tds', 'tds (ppm)': 'tds', 'tds': 'tds',
    'temp(c)': 'temperature', 'temp': 'temperature', 'temperature': 'temperature'
}

def build_metadata_from_sheet(df: pd.DataFrame, tolerance_expand: float = 0.10,
                              col_map: Dict[str, str] = None) -> pd.DataFrame:
    """Return canonical metadata df with columns: plant, <feature>_min/_opt_low/_opt_high/_max"""
    col_map = col_map or _DEFAULT_COL_MAP
    cols_norm = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=cols_norm)
    lower_to_orig = {c.lower(): c for c in df.columns}
    rows = []
    for idx, row in df.iterrows():
        # detect plant name column
        plant_name = None
        for c in ("Plant", "plant", "Name", "name", "Species", "species"):
            if c in row and pd.notna(row[c]):
                plant_name = str(row[c]).strip()
                break
        if not plant_name:
            plant_name = str(idx)
        meta = {"plant": plant_name}
        for lower_col, feat in col_map.items():
            if lower_col in lower_to_orig:
                orig = lower_to_orig[lower_col]
                try:
                    low, high = parse_range_string(row[orig])
                except Exception:
                    continue
                opt_low, opt_high = low, high
                span = max(1e-9, opt_high - opt_low)
                expand = span * tolerance_expand
                if opt_low == opt_high:
                    expand = max(0.1, abs(opt_low) * tolerance_expand)
                minv = opt_low - expand
                maxv = opt_high + expand
                meta[f"{feat}_min"] = float(minv)
                meta[f"{feat}_opt_low"] = float(opt_low)
                meta[f"{feat}_opt_high"] = float(opt_high)
                meta[f"{feat}_max"] = float(maxv)
        if any(k.endswith("_min") for k in meta.keys()):
            rows.append(meta)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def load_metadata(csv_path: Optional[str] = None, excel_path: Optional[str] = None,
                  tolerance_expand: float = 0.10) -> pd.DataFrame:
    """Load canonical metadata (CSV preferred, else parse Excel)."""
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        # Ensure columns are canonical (assume parsed CSV already canonical)
        return df
    if excel_path and Path(excel_path).exists():
        xls = pd.ExcelFile(excel_path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(excel_path, sheet_name=sheet)
            parsed = build_metadata_from_sheet(df, tolerance_expand=tolerance_expand)
            if not parsed.empty:
                return parsed
        raise FileNotFoundError("No parsable sheet found in Excel.")
    raise FileNotFoundError("No metadata file found.")

# ---------------------------
# Deterministic scorer (used to label synthetic data)
# ---------------------------
def feature_score(x: float, bounds: Tuple[float, float, float, float]) -> float:
    minv, opt_low, opt_high, maxv = bounds
    if x < minv or x > maxv:
        return 0.0
    if opt_low <= x <= opt_high:
        return 1.0
    if minv <= x < opt_low:
        denom = (opt_low - minv)
        if denom == 0:
            return 0.0
        return max(0.0, 1.0 - (opt_low - x) / denom)
    if opt_high < x <= maxv:
        denom = (maxv - opt_high)
        if denom == 0:
            return 0.0
        return max(0.0, 1.0 - (x - opt_high) / denom)
    return 0.0

def plant_score_for_readings(plant_row: pd.Series, readings: Dict[str, float],
                             feature_weights: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, float]]:
    """Return combined score and per-feature dict for one plant row given readings."""
    if feature_weights is None:
        feature_weights = {}
    total_weight = 0.0
    weighted_sum = 0.0
    perfs = {}
    for col in plant_row.index:
        if not col.endswith("_min"):
            continue
        feat = col[:-4]
        try:
            bounds = (plant_row[f"{feat}_min"], plant_row[f"{feat}_opt_low"],
                      plant_row[f"{feat}_opt_high"], plant_row[f"{feat}_max"])
        except Exception:
            continue
        if any(pd.isna(b) for b in bounds):
            continue
        if feat not in readings:
            continue
        x = float(readings[feat])
        s = feature_score(x, bounds)
        w = float(feature_weights.get(feat, 1.0))
        perfs[feat] = s
        weighted_sum += w * s
        total_weight += w
    if total_weight == 0:
        return 0.0, perfs
    return weighted_sum / total_weight, perfs

# ---------------------------
# Synthetic dataset generation
# ---------------------------
def generate_synthetic_sample_from_plant(plant_row: pd.Series, feature_noise_frac: float = 0.15) -> Dict[str, float]:
    """
    Generate a single synthetic reading that is plausible for the given plant_row.
    For each feature available in the plant_row, we sample from a distribution centered near the opt band,
    with some chance to sample outside (so dataset contains both good and borderline samples).
    """
    sample = {}
    for col in plant_row.index:
        if not col.endswith("_min"):
            continue
        feat = col[:-4]
        try:
            opt_low = float(plant_row[f"{feat}_opt_low"])
            opt_high = float(plant_row[f"{feat}_opt_high"])
            minv = float(plant_row[f"{feat}_min"])
            maxv = float(plant_row[f"{feat}_max"])
        except Exception:
            continue
        # sample strategy:
        # with prob 0.75 sample inside opt band uniformly; with prob 0.25 sample from expanded tails or noise.
        if random.random() < 0.75:
            if opt_high == opt_low:
                # single value; Gaussian around that
                center = opt_low
                sigma = max(0.01, abs(center) * feature_noise_frac)
                val = random.gauss(center, sigma)
            else:
                val = random.uniform(opt_low, opt_high)
        else:
            # sample outside but within min-max with more variance
            val = random.uniform(minv, maxv)
        # clamp
        val = max(minv, min(maxv, val))
        sample[feat] = float(val)
    return sample

def synthesize_dataset(metadata: pd.DataFrame, samples_per_plant: int = 500,
                       feature_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    For each plant in metadata, generate samples_per_plant synthetic readings.
    Label each generated reading with the deterministic top-scoring plant (argmax over all plants).
    Returns DataFrame where columns are features + 'label' (plant name).
    """
    rows = []
    feature_weights = feature_weights or {}
    all_features = sorted({c[:-4] for c in metadata.columns if c.endswith("_min")})
    for _, plant_row in metadata.iterrows():
        plant_name = plant_row["plant"]
        for _ in range(samples_per_plant):
            base = generate_synthetic_sample_from_plant(plant_row)
            # Optionally add small gaussian noise to all features for variety
            noisy = {}
            for f in all_features:
                if f in base:
                    noisy[f] = float(base[f] * (1.0 + random.gauss(0, 0.02)))  # 2% gaussian jitter
                else:
                    # if this plant lacks this feature, sample a plausible value across global ranges
                    # we'll sample later; for now leave absent (scorer will ignore)
                    pass
            # now compute deterministic scores for all plants and set label to argmax
            scores = []
            for _, other_row in metadata.iterrows():
                score, _ = plant_score_for_readings(other_row, noisy, feature_weights)
                scores.append((other_row["plant"], score))
            # choose top
            scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
            top_plant = scores_sorted[0][0]
            # For training stability, if top score is zero (no plant matches), assign the generating plant
            if scores_sorted[0][1] == 0.0:
                top_plant = plant_name
            # build final row: fill missing features with NaN (we'll impute later)
            row = {f: noisy.get(f, float("nan")) for f in all_features}
            row["label"] = top_plant
            rows.append(row)
    df = pd.DataFrame(rows)
    # Impute any missing features by column median (simple)
    for f in [c for c in df.columns if c != "label"]:
        if df[f].isna().any():
            med = df[f].median(skipna=True)
            if math.isnan(med):
                med = 0.0
            df[f].fillna(med, inplace=True)
    return df

# ---------------------------
# Fake streaming generator
# ---------------------------
def fake_stream_generator(interval_sec: float = 1.0, seed: Optional[int] = None):
    """
    Yields artificial readings similar to ESP32 JSON, e.g. {"tds": 600.0, "ph": 6.0, "temperature": 22.0}
    Variation: generates readings across practical ranges. Random but reproducible if seed provided.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # Some practical ranges for features (broad)
    # We'll produce values around midpoints commonly found in hydroponics systems.
    while True:
        # pH typical 5.5-7.0
        ph = round(random.uniform(5.0, 7.5) + random.gauss(0, 0.05), 3)
        # TDS typical 200-2000 ppm
        tds = round(random.uniform(200, 1600) + random.gauss(0, 20), 1)
        # Temperature typical 15-30 C
        temperature = round(random.uniform(15.0, 28.0) + random.gauss(0, 0.2), 2)
        # EC roughly derived from TDS roughly EC (mS/cm) ≈ TDS / 700 (very approximate)
        ec = round(tds / 700.0 + random.gauss(0, 0.01), 3)
        yield {"tds": float(tds), "ph": float(ph), "temperature": float(temperature), "ec": float(ec)}
        time.sleep(interval_sec)

# ---------------------------
# Train classifier
# ---------------------------

def train_classifier_from_synthetic(df_synth: pd.DataFrame, random_state: int = 42):
    """
    Train RandomForest on the synthetic dataset.
    If any class has fewer than 2 samples, do NOT stratify the train_test_split
    (stratification requires at least 2 samples per class).
    """
    X = df_synth.drop(columns=["label"])
    y = df_synth["label"].astype(str)

    # show class distribution
    class_counts = y.value_counts()
    print("[INFO] Label distribution (counts):")
    print(class_counts.to_string())

    min_count = int(class_counts.min())
    if min_count < 2:
        print(f"[WARN] Minimum class count is {min_count} (<2). Skipping stratified split.")
        stratify_arg = None
    else:
        stratify_arg = y

    # perform train/test split without forcing stratify if unsuitable
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=random_state, stratify=stratify_arg
    )

    # Train classifier
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[TRAIN] RandomForest accuracy (top-1) on synthetic holdout: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Top-3 accuracy if possible
    try:
        top3 = top_k_accuracy_score(y_test, clf.predict_proba(X_test), k=3, labels=clf.classes_)
        print(f"[TRAIN] Top-3 accuracy on synthetic holdout: {top3:.4f}")
    except Exception:
        pass

    return clf, clf.classes_, X_test, y_test


# ---------------------------
# Demo / main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Fake stream + train ML recommender using provided plant ranges.")
    parser.add_argument("--csv", help="parsed metadata CSV (preferred)", default="/mnt/data/parsed_plant_metadata_v2.csv")
    parser.add_argument("--excel", help="metadata Excel path (fallback)", default="/mnt/data/Hydroponics_Plant_Range.xlsx")
    parser.add_argument("--samples_per_plant", type=int, default=400, help="synthetic samples per plant")
    parser.add_argument("--save_model", default="trained_rf.pkl")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--stream_count", type=int, default=10, help="how many fake streaming samples to show as demo")
    args = parser.parse_args()

    # Load metadata
    try:
        metadata = load_metadata(csv_path=args.csv if Path(args.csv).exists() else None,
                                 excel_path=args.excel if Path(args.excel).exists() else None,
                                 tolerance_expand=0.10)
    except Exception as e:
        print("Failed to load metadata:", e)
        return

    print(f"Loaded metadata for {len(metadata)} plants. Features detected:",
          sorted({c[:-4] for c in metadata.columns if c.endswith("_min")}))

    # Synthesize dataset
    print(f"Generating synthetic dataset ({args.samples_per_plant} samples per plant)...")
    df_synth = synthesize_dataset(metadata, samples_per_plant=args.samples_per_plant)
    print("Synthetic dataset shape:", df_synth.shape)
    # Save dataset
    df_synth.to_csv("synthetic_dataset.csv", index=False)
    print("Saved synthetic_dataset.csv")

    # Train classifier
    clf, classes, X_test, y_test = train_classifier_from_synthetic(df_synth, random_state=args.seed)
    dump(clf, args.save_model)
    print(f"Saved trained model to {args.save_model}")

    # Compare deterministic recommender vs ML on several fake streaming samples
    print("\n=== Streaming demo: comparing deterministic scorer and ML classifier ===")
    gen = fake_stream_generator(interval_sec=0.0, seed=args.seed + 1)
    all_features = sorted({c[:-4] for c in metadata.columns if c.endswith("_min")})
    feature_weights = {"ph": 2.0, "tds": 1.5, "ec": 1.5, "temperature": 1.0}
    for i in range(args.stream_count):
        sample = next(gen)
        # Keep only features our metadata uses (and in consistent order for ML)
        sample_for_model = {f: sample.get(f, 0.0) for f in all_features}
        # deterministic ranking
        det_scores = []
        for _, row in metadata.iterrows():
            s, perfs = plant_score_for_readings(row, sample_for_model, feature_weights)
            det_scores.append((row["plant"], float(s), perfs))
        det_scores_sorted = sorted(det_scores, key=lambda x: x[1], reverse=True)
        det_topk = det_scores_sorted[:3]

        # ML prediction
        Xvec = np.array([sample_for_model[f] for f in all_features]).reshape(1, -1)
        ml_pred = clf.predict(Xvec)[0]
        # get ml top-3 via predict_proba
        probs = clf.predict_proba(Xvec)[0]
        # map classes to probs
        cls_probs = list(zip(clf.classes_, probs))
        cls_probs_sorted = sorted(cls_probs, key=lambda x: x[1], reverse=True)
        ml_topk = cls_probs_sorted[:3]

        # print comparison
        print(f"\nSample #{i+1}: {json.dumps(sample_for_model)}")
        print("Deterministic top-3:")
        for rank, (pname, sc, perfs) in enumerate(det_topk, start=1):
            pf = ", ".join(f"{k}:{v:.2f}" for k, v in perfs.items())
            print(f"  {rank}. {pname} — score {sc:.3f} — {pf}")
        print("ML top-3 (name:prob):")
        for rank, (pname, prob) in enumerate(ml_topk, start=1):
            print(f"  {rank}. {pname} — {prob:.3f}")

    print("\nDone demo.")

if __name__ == "__main__":
    main()
