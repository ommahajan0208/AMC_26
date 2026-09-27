"""
05_predict_and_submit.py
========================
Phase 5 - Inference on Test Set + Submission Formatting

Loads the trained LightGBM model, scores the test feature pairs,
then formats the output exactly as the submission format requires:

  matching_results.tsv:
    source1_entity_id   |   matched_entity_ids (comma-separated S2/S3 IDs)

  Every S1 entity from test_source1 must have a row.
  Entities with no accepted match get an empty matched_entity_ids field.

Also writes candidate_pairs_submission.tsv for the optional
blocking recall check in the validator.

Usage:
    python 05_predict_and_submit.py
"""

import csv
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

import config

config.ensure_output_dirs()
config.print_config()
print()

# ===========================================================================
# 1. Load model
# ===========================================================================

if not config.LGBM_MODEL_PATH.exists():
    print(f"[ERROR] Model not found at {config.LGBM_MODEL_PATH} - run 04_train_lgbm.py first")
    sys.exit(1)

print(f"[LOAD] Loading model from {config.LGBM_MODEL_PATH.name} ...")
with open(config.LGBM_MODEL_PATH, "rb") as f:
    model = pickle.load(f)
print("  Model loaded.")

# ===========================================================================
# 2. Load test features
# ===========================================================================

if not config.FEATURES_TEST_PARQUET.exists():
    print(f"[ERROR] {config.FEATURES_TEST_PARQUET.name} not found - run 03_feature_engineering.py first")
    sys.exit(1)

print(f"\n[LOAD] Loading test features from {config.FEATURES_TEST_PARQUET.name} ...")
t0 = time.perf_counter()
feat_pl = pl.read_parquet(config.FEATURES_TEST_PARQUET)
print(f"  {feat_pl.height:,} candidate pairs loaded in {time.perf_counter()-t0:.1f}s")

# Fill missing feature cols and cast to float32 - no pyarrow needed
for f in config.FEATURE_COLS:
    if f not in feat_pl.columns:
        feat_pl = feat_pl.with_columns(pl.lit(0.0).cast(pl.Float32).alias(f))
feat_pl = feat_pl.with_columns([
    pl.col(f).cast(pl.Float32).fill_nan(0.0).fill_null(0.0) for f in config.FEATURE_COLS
])

X      = feat_pl.select(config.FEATURE_COLS).to_numpy().astype(np.float32)
q_ids  = feat_pl.select("query_entity_id").to_numpy().ravel().tolist()
s1_ids = feat_pl.select("candidate_s1_entity_id").to_numpy().ravel().tolist()

# ===========================================================================
# 3. Score
# ===========================================================================

print(f"\n[SCORE] Scoring {len(q_ids):,} candidate pairs ...")
t0    = time.perf_counter()
probs = model.predict_proba(X)[:, 1]
print(f"  Scored in {time.perf_counter()-t0:.1f}s")

accept_mask    = probs > config.T_HIGH
uncertain_mask = (probs >= config.T_LOW) & (probs <= config.T_HIGH)
reject_mask    = probs < config.T_LOW

print(f"  REJECT:    {reject_mask.sum():,} ({reject_mask.mean()*100:.1f}%)")
print(f"  UNCERTAIN: {uncertain_mask.sum():,} ({uncertain_mask.mean()*100:.1f}%)")
print(f"  ACCEPT:    {accept_mask.sum():,} ({accept_mask.mean()*100:.1f}%)")

# Include uncertain + accepted (conservative - better recall)
match_mask   = probs >= config.T_LOW
n_total      = len(q_ids)
print(f"\n  Accepted (including uncertain): {match_mask.sum():,} pairs")

# ===========================================================================
# 4. Load all test S1 entity IDs
# ===========================================================================

print(f"\n[LOAD] Loading all test S1 entity IDs ...")
if not config.TEST_SOURCE1_PARQUET.exists():
    # Fallback to TSV
    s1_df = pl.read_csv(
        str(config.TEST_SOURCE1_TSV),
        separator="\t",
        infer_schema_length=0,
        null_values=["", "NULL", "NaN"],
    )
else:
    s1_df = pl.read_parquet(config.TEST_SOURCE1_PARQUET)

s1_df = s1_df.rename({c: c.strip().lower() for c in s1_df.columns})
s1_ids_test = s1_df["entity_id"].cast(pl.Utf8).to_list()
print(f"  {len(s1_ids_test):,} S1 entities in test set")

# ===========================================================================
# 5. Build match mapping: S1 -> list of matched S2/S3 IDs
# ===========================================================================

print(f"\n[BUILD] Building match mapping ...")
# Group by candidate_s1_entity_id: which S2/S3 entities matched to each S1
match_map: dict[str, list[str]] = {s1: [] for s1 in s1_ids_test}

for i in range(n_total):
    if match_mask[i]:
        s1_id = s1_ids[i]
        q_id  = q_ids[i]
        if s1_id in match_map:
            match_map[s1_id].append(q_id)

# Use s1_ids_test (all test S1 IDs in order) for the final output
matched_count   = sum(1 for v in match_map.values() if v)
unmatched_count = sum(1 for v in match_map.values() if not v)
print(f"  S1 entities with matches:    {matched_count:,}")
print(f"  S1 entities without matches: {unmatched_count:,} (will get empty row)")

# ===========================================================================
# 6. Write matching_results.tsv
# ===========================================================================

print(f"\n[WRITE] Writing {config.SUBMISSION_PATH.name} ...")
t0 = time.perf_counter()
with config.SUBMISSION_PATH.open("w", encoding="utf-8", newline="\n") as fh:
    fh.write("source1_entity_id\tmatched_entity_ids\n")
    for s1_id in s1_ids_test:
        matched = match_map.get(s1_id, [])
        seen    = set()
        unique  = [x for x in matched if not (x in seen or seen.add(x))]
        fh.write(f"{s1_id}\t{','.join(unique)}\n")

size_kb = config.SUBMISSION_PATH.stat().st_size / 1e3
print(f"  Saved {len(s1_ids):,} rows ({size_kb:.1f} KB) in {time.perf_counter()-t0:.1f}s")

# ===========================================================================
# 7. Write candidate_pairs_submission.tsv (for validator)
# ===========================================================================

print(f"\n[WRITE] Writing candidate_pairs_submission.tsv ...")
cand_map: dict[str, list[str]] = {s1: [] for s1 in s1_ids_test}
for i in range(n_total):
    s1_id = s1_ids[i]
    q_id  = q_ids[i]
    if s1_id in cand_map:
        cand_map[s1_id].append(q_id)

with config.CANDIDATE_SUBMISSION_PATH.open("w", encoding="utf-8", newline="\n") as fh:
    fh.write("source1_entity_id\tcandidate_entity_ids\n")
    for s1_id in s1_ids_test:
        cands_list = cand_map.get(s1_id, [])
        seen   = set()
        unique = [x for x in cands_list if not (x in seen or seen.add(x))]
        fh.write(f"{s1_id}\t{','.join(unique)}\n")

print(f"  Saved candidate_pairs_submission.tsv")

# ===========================================================================
# 8. Quick sanity check
# ===========================================================================

print(f"\n[CHECK] Submission sanity check ...")
sample_with_match = [(s1, ids) for s1, ids in match_map.items() if ids][:3]
sample_empty      = [s1 for s1, ids in match_map.items() if not ids][:3]

if sample_with_match:
    print(f"  Sample matches:")
    for s1, ids in sample_with_match:
        print(f"    {s1} -> {', '.join(ids[:5])}{'...' if len(ids) > 5 else ''}")

if sample_empty:
    print(f"  Sample empty (singletons): {', '.join(sample_empty)}")

print(f"\n[DONE] Submission ready at: {config.SUBMISSION_PATH}")
print(f"       Run validate_submission.py to check format before submitting.")
