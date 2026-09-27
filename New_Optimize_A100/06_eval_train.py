"""
06_eval_train.py
================
Computes the competition F_0.5 score on the TRAINING set using the
matches produced by 04_train_lgbm.py.

Competition scoring rules (exact):
  - F_0.5 is computed per Source 1 entity, then macro-averaged.
  - Singletons (S1 with no true matches):
      score 1.0 if you predict empty list  (correct)
      score 0.0 if you predict any match   (false merge)
  - For entities with true matches:
      per-entity P = |predicted & true| / |predicted|   (0 if predicted empty)
      per-entity R = |predicted & true| / |true|
      per-entity F_0.5 = (1.25 * P * R) / (0.25 * P + R)
                         0.0 if both P and R are 0

Usage:
    python 06_eval_train.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl

import config

config.ensure_output_dirs()
config.print_config()
print()


# ===========================================================================
# 1. Load ground truth
# ===========================================================================

def load_ground_truth() -> dict[str, set[str]]:
    """Returns {s1_entity_id: set_of_matched_s2_s3_ids}."""
    path = config.TRAIN_GT_PARQUET if config.TRAIN_GT_PARQUET.exists() else config.TRAIN_GT_TSV
    if not path.exists():
        print(f"[ERROR] Ground truth not found at {path}")
        sys.exit(1)

    if str(path).endswith(".parquet"):
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(str(path), separator="\t", infer_schema_length=0,
                         null_values=["", "NULL", "NaN"])

    df = df.rename({c: c.strip().lower() for c in df.columns})
    s1_col    = df.columns[0]
    match_col = df.columns[1]

    gt: dict[str, set[str]] = {}
    for row in df.to_dicts():
        s1_id     = str(row[s1_col]).strip()
        raw_match = str(row[match_col]).strip() if row[match_col] else ""
        matched   = {m.strip() for m in raw_match.split(",") if m.strip()} if raw_match else set()
        gt[s1_id] = matched

    print(f"[GT] Loaded {len(gt):,} S1 entities from ground truth")
    n_with_match = sum(1 for v in gt.values() if v)
    n_singleton  = sum(1 for v in gt.values() if not v)
    print(f"     {n_with_match:,} with true matches  |  {n_singleton:,} true singletons")
    return gt


# ===========================================================================
# 2. Load predictions
# ===========================================================================

def load_train_predictions() -> dict[str, set[str]]:
    """
    Loads the matching_lgbm.tsv produced by 04_train_lgbm.py and converts
    it from (query_entity_id, candidate_s1_entity_id) format into
    {s1_entity_id: set_of_matched_query_ids}.
    """
    match_path = config.OUTPUT_DIR / "matching_lgbm.tsv"

    if not match_path.exists():
        print(f"[ERROR] {match_path.name} not found - run 04_train_lgbm.py first")
        sys.exit(1)

    pred_map: dict[str, set[str]] = {}
    with open(match_path, encoding="utf-8") as fh:
        header = fh.readline()   # skip header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            q_id  = parts[0].strip()   # S2 or S3
            s1_id = parts[1].strip()   # S1
            pred_map.setdefault(s1_id, set()).add(q_id)

    print(f"[PRED] Loaded predictions: {len(pred_map):,} S1 entities have at least one match")
    total_matches = sum(len(v) for v in pred_map.values())
    print(f"       {total_matches:,} total predicted match pairs")
    return pred_map


# ===========================================================================
# 3. Load all train S1 entity IDs (we must score every one)
# ===========================================================================

def load_all_s1_ids() -> list[str]:
    path = config.TRAIN_SOURCE1_PARQUET if config.TRAIN_SOURCE1_PARQUET.exists() \
           else config.TRAIN_SOURCE1_TSV
    if str(path).endswith(".parquet"):
        df = pl.read_parquet(path)
    else:
        df = pl.read_csv(str(path), separator="\t", infer_schema_length=0)
    df = df.rename({c: c.strip().lower() for c in df.columns})
    return df["entity_id"].cast(pl.Utf8).to_list()


# ===========================================================================
# 4. Per-entity F_0.5 (exact competition formula)
# ===========================================================================

def entity_f05(true_set: set[str], pred_set: set[str]) -> float:
    """
    Computes F_0.5 for one S1 entity.

    Singleton rule:
      true_set empty, pred_set empty -> 1.0
      true_set empty, pred_set non-empty -> 0.0

    Standard rule:
      F_0.5 = (1.25 * P * R) / (0.25 * P + R)
      0.0 if P = R = 0.
    """
    # Singleton handling
    if not true_set:
        return 1.0 if not pred_set else 0.0

    # Non-singleton
    tp = len(true_set & pred_set)

    if not pred_set:
        # Missed everything - precision is undefined, treat as 0
        precision = 0.0
    else:
        precision = tp / len(pred_set)

    recall = tp / len(true_set)

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0

    return (1.25 * precision * recall) / denom


# ===========================================================================
# 5. Macro F_0.5 + breakdown
# ===========================================================================

def compute_macro_f05(
    all_s1_ids: list[str],
    gt: dict[str, set[str]],
    pred_map: dict[str, set[str]],
):
    """
    Compute macro-averaged F_0.5 over all S1 entities (including singletons).
    Also reports precision, recall, and detailed breakdown.
    """
    scores        = []
    tp_total      = 0
    fp_total      = 0
    fn_total      = 0
    singleton_correct   = 0
    singleton_incorrect = 0
    match_entity_scores = []

    for s1_id in all_s1_ids:
        true_set = gt.get(s1_id, set())
        pred_set = pred_map.get(s1_id, set())

        f = entity_f05(true_set, pred_set)
        scores.append(f)

        if not true_set:
            # Singleton
            if not pred_set:
                singleton_correct += 1
            else:
                singleton_incorrect += 1
        else:
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            tp_total += tp
            fp_total += fp
            fn_total += fn
            match_entity_scores.append(f)

    macro_f05   = sum(scores) / len(scores) if scores else 0.0
    # Micro precision/recall over non-singleton entities only
    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    # Macro over match entities only (excludes singletons)
    macro_match_f05 = sum(match_entity_scores) / len(match_entity_scores) \
                      if match_entity_scores else 0.0

    return {
        "macro_f05":            macro_f05,
        "macro_f05_non_single": macro_match_f05,
        "micro_precision":      micro_p,
        "micro_recall":         micro_r,
        "n_entities":           len(all_s1_ids),
        "n_singletons_correct": singleton_correct,
        "n_singletons_wrong":   singleton_incorrect,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "scores":               scores,
    }


# ===========================================================================
# 6. Run and print report
# ===========================================================================

t0 = time.perf_counter()

all_s1_ids = load_all_s1_ids()
print(f"[S1]   {len(all_s1_ids):,} S1 entities in train set\n")

gt       = load_ground_truth()
pred_map = load_train_predictions()

# Add every S1 to gt if not present (they are singletons by definition)
for s1_id in all_s1_ids:
    if s1_id not in gt:
        gt[s1_id] = set()

print()
result = compute_macro_f05(all_s1_ids, gt, pred_map)

print("=" * 60)
print("  TRAIN SET EVALUATION (F_0.5 - competition metric)")
print("=" * 60)
print()
print(f"  Macro F_0.5 (ALL entities, incl. singletons)  : {result['macro_f05']:.4f}  <-- LEADERBOARD METRIC")
print(f"  Macro F_0.5 (non-singleton entities only)      : {result['macro_f05_non_single']:.4f}")
print(f"  Micro Precision (TP / (TP+FP))                 : {result['micro_precision']:.4f}")
print(f"  Micro Recall    (TP / (TP+FN))                 : {result['micro_recall']:.4f}")
print()
print(f"  TP: {result['tp']:,}   FP: {result['fp']:,}   FN: {result['fn']:,}")
print()
print(f"  Total S1 entities scored  : {result['n_entities']:,}")
print(f"  Singletons correct (pred empty = true empty) : {result['n_singletons_correct']:,}")
print(f"  Singletons wrong   (predicted a match)       : {result['n_singletons_wrong']:,}")
print()

# Score distribution
scores = result["scores"]
buckets = [0.0, 0.25, 0.5, 0.75, 0.9, 1.01]
labels  = ["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-0.90", "0.90-1.00"]
print("  Per-entity F_0.5 distribution:")
for lo, hi, label in zip(buckets, buckets[1:], labels):
    count = sum(1 for s in scores if lo <= s < hi)
    bar   = "#" * (count * 30 // max(len(scores), 1))
    print(f"    {label}: {count:>6,}  {bar}")

print()
print(f"  Elapsed: {time.perf_counter()-t0:.1f}s")
print()
print("[NOTE] This score is on the TRAINING set and will be higher than")
print("       the leaderboard score (which uses the held-out test set).")
print("       Use it to compare iterations, not as the final number.")
