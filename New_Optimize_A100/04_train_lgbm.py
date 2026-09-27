"""
04_train_lgbm.py
================
Phase 4 - LightGBM Binary Classifier

Trains a gradient-boosted tree model on the 29 feature columns
produced by 03_feature_engineering.py.

Evaluation strategy:
  Eval 1: Stratified 80/20 split - reports F_0.5 on val set
  Eval 2: US-train / India-validate - proxy for France generalisation

In WHOLE_DATA = False mode, skips the final 100% retrain step
(too few rows to be meaningful, just validates the logic).

Outputs:
    output/lgbm_model.pkl
    output/threshold_sweep.csv
    output/feature_importance.tsv
    output/uncertain_pairs.tsv   (for reranker, if implemented)
    output/matching_lgbm.tsv     (Stage-1-only high-confidence matches)

Usage:
    python 04_train_lgbm.py
"""

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import csv

import numpy as np
import polars as pl
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit

import config

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("lightgbm required: pip install lightgbm")

config.ensure_output_dirs()
config.print_config()
print()

LABEL_COL = "label"
ID_COLS   = ["query_entity_id", "candidate_s1_entity_id"]


# ===========================================================================
# 1. Metric helpers
# ===========================================================================

def f_beta(y_true, y_pred, beta=0.5):
    return float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))


def threshold_sweep(y_true, y_prob, beta=0.5):
    thresholds = np.arange(0.30, 0.96, 0.01)
    # Returns: list of dicts, best_threshold, best_f05
    rows = []
    best_t, best_f = 0.5, 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f05    = f_beta(y_true, y_pred, beta)
        prec   = precision_score(y_true, y_pred, zero_division=0)
        rec    = recall_score(y_true, y_pred, zero_division=0)
        rows.append({"threshold": round(float(t), 2), "F_0.5": f05,
                     "precision": prec, "recall": rec})
        if f05 > best_f:
            best_f, best_t = f05, float(t)
    return best_t, best_f, rows


# ===========================================================================
# 2. Train helper
# ===========================================================================

def train_lgbm(X_train, y_train, X_val, y_val, params=None):
    """X_train/X_val are plain numpy float32 arrays. y_train/y_val are 1-D int arrays."""
    p = {**config.LGBM_PARAMS, **(params or {})}
    print(f"  Training LightGBM: {len(X_train):,} train | {len(X_val):,} val")
    print(f"  Positive rate - train: {y_train.mean()*100:.2f}%  val: {y_val.mean()*100:.2f}%")
    model = lgb.LGBMClassifier(**p)
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )
    except Exception as e:
        # GPU device may not be available (e.g., faiss-gpu but no lgb GPU build)
        if "gpu" in str(e).lower() or "device" in str(e).lower():
            print(f"  [WARN] LightGBM GPU failed ({e}), retrying on CPU")
            p_cpu = {k: v for k, v in p.items() if k not in ("device", "gpu_platform_id", "gpu_device_id")}
            model  = lgb.LGBMClassifier(**p_cpu)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
            )
        else:
            raise
    print(f"  Best iteration: {model.best_iteration_}")
    return model


# ===========================================================================
# 3. Main run
# ===========================================================================

feat_path = config.FEATURES_TRAIN_PARQUET

if not feat_path.exists():
    print(f"[ERROR] {feat_path.name} not found - run 03_feature_engineering.py first")
    sys.exit(1)

if config.LGBM_MODEL_PATH.exists():
    print(f"[SKIP] {config.LGBM_MODEL_PATH.name} already exists")
    print("       Delete it to retrain.")
    sys.exit(0)

print(f"[LOAD] Loading features from {feat_path.name} ...")
t0 = time.perf_counter()

# Load with Polars, convert to numpy for sklearn/lgbm compatibility
feat_pl = pl.read_parquet(feat_path)
# Cap rows for QUICK_RUN (use first N rows; not a true stratified sample but enough to test pipeline)
max_rows = config.effective_rows()
if max_rows > 0:
    feat_pl = feat_pl.slice(0, max_rows)
print(f"  {feat_pl.height:,} rows x {feat_pl.width} cols")

if LABEL_COL not in feat_pl.columns:
    print(f"[ERROR] '{LABEL_COL}' column missing - run 03_feature_engineering.py with ground truth")
    sys.exit(1)

# Fill missing feature columns with 0
missing_feats = [f for f in config.FEATURE_COLS if f not in feat_pl.columns]
if missing_feats:
    print(f"[WARN] Missing feature columns (filling 0): {missing_feats}")
    for f in missing_feats:
        feat_pl = feat_pl.with_columns(pl.lit(0.0).cast(pl.Float32).alias(f))

# Cast all feature cols to Float32 and fill nulls with 0 - stays in Polars, no pyarrow needed
feat_pl = feat_pl.with_columns([
    pl.col(f).cast(pl.Float32).fill_nan(0.0).fill_null(0.0) for f in config.FEATURE_COLS
])

# Extract as numpy arrays directly (no pandas / no pyarrow required)
X          = feat_pl.select(config.FEATURE_COLS).to_numpy().astype(np.float32)   # [N, F]
y          = feat_pl.select(LABEL_COL).to_numpy().ravel().astype(np.int32)       # [N]
# Keep ID columns as plain python lists for output writing
q_ids      = feat_pl.select("query_entity_id").to_numpy().ravel().tolist()
s1_ids     = feat_pl.select("candidate_s1_entity_id").to_numpy().ravel().tolist()
strat_col  = feat_pl.select("source_pair").to_numpy().ravel().astype(np.int32)   # for stratify

print(f"  Loaded in {time.perf_counter()-t0:.1f}s | positive rate: {y.mean()*100:.2f}%")

# ===========================================================================
# Eval 1 - Stratified 80/20 split
# ===========================================================================
print(f"\n[EVAL 1] Stratified 80/20 split ...")
splitter               = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, val_idx     = next(splitter.split(X, strat_col))

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]

model_eval = train_lgbm(X_train, y_train, X_val, y_val)

y_val_prob              = model_eval.predict_proba(X_val)[:, 1]
t_opt, best_f05, sweep_rows = threshold_sweep(y_val, y_val_prob)
print(f"  Eval 1 - best threshold: {t_opt:.2f} | F_0.5: {best_f05:.4f}")
print(f"  Strategy thresholds: T_low={config.T_LOW} | T_high={config.T_HIGH}")

top5 = sorted(sweep_rows, key=lambda r: r["F_0.5"], reverse=True)[:5]
print(f"  Top-5 thresholds:")
for r in top5:
    print(f"    t={r['threshold']:.2f}  F_0.5={r['F_0.5']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}")

# ===========================================================================
# Eval 2 - Cross-country generalisation (US -> India) - full run only
# ===========================================================================
if not config.WHOLE_DATA:
    print(f"\n[EVAL 2] Skipped in dev mode (too few rows per country)")

# ===========================================================================
# Final model - retrain on 100% data (skip in dev mode)
# ===========================================================================
if config.WHOLE_DATA:
    print(f"\n[TRAIN] Retraining on 100% data ...")
    model_full = train_lgbm(X, y, X_val, y_val)
else:
    print(f"\n[TRAIN] WHOLE_DATA=False - using eval model as final model (dev mode)")
    model_full = model_eval

# ===========================================================================
# Score all pairs and produce outputs
# ===========================================================================
n_total = len(q_ids)
print(f"\n[SCORE] Scoring all {n_total:,} candidate pairs ...")
# predict_proba on the full array - LightGBM internally parallelises with n_jobs
all_prob = model_full.predict_proba(X)[:, 1]

t_low, t_high  = config.T_LOW, config.T_HIGH
uncertain_mask = (all_prob >= t_low) & (all_prob <= t_high)
accept_mask    = all_prob > t_high
reject_mask    = all_prob < t_low

print(f"  REJECT:    {reject_mask.sum():,} ({reject_mask.mean()*100:.1f}%)")
print(f"  UNCERTAIN: {uncertain_mask.sum():,} ({uncertain_mask.mean()*100:.1f}%)")
print(f"  ACCEPT:    {accept_mask.sum():,} ({accept_mask.mean()*100:.1f}%)")

out_dir = config.OUTPUT_DIR
out_dir.mkdir(parents=True, exist_ok=True)


def write_tsv(path, header, rows_iter):
    """Write a TSV using the stdlib csv module - no pandas required."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows_iter)


# Uncertain pairs
uncertain_rows = [
    (q_ids[i], s1_ids[i], round(float(all_prob[i]), 6))
    for i in range(n_total) if uncertain_mask[i]
]
write_tsv(out_dir / "uncertain_pairs.tsv",
          ["query_entity_id", "candidate_s1_entity_id", "p_lgbm"],
          uncertain_rows)
print(f"  Saved {len(uncertain_rows):,} uncertain pairs to uncertain_pairs.tsv")

# High-confidence accepted matches
accept_rows = [
    (q_ids[i], s1_ids[i], round(float(all_prob[i]), 6))
    for i in range(n_total) if accept_mask[i]
]
write_tsv(out_dir / "matching_lgbm.tsv",
          ["query_entity_id", "candidate_s1_entity_id", "p_lgbm"],
          accept_rows)
print(f"  Saved {len(accept_rows):,} accepted matches to matching_lgbm.tsv")

# Feature importance
gain  = model_full.booster_.feature_importance(importance_type="gain")
split = model_full.booster_.feature_importance(importance_type="split")
fi_rows = sorted(zip(config.FEATURE_COLS, gain, split), key=lambda r: r[1], reverse=True)
write_tsv(out_dir / "feature_importance.tsv",
          ["feature", "importance_gain", "importance_split"],
          fi_rows)
print(f"\n  Top-10 features by gain:")
for feat, g, s in fi_rows[:10]:
    print(f"    {feat:<40} gain={g:>8}  split={s:>8}")

# Save model
with open(config.LGBM_MODEL_PATH, "wb") as f:
    pickle.dump(model_full, f)
print(f"\n  Model saved to {config.LGBM_MODEL_PATH.name}")

# Threshold sweep CSV
write_tsv(out_dir / "threshold_sweep.csv",
          ["threshold", "F_0.5", "precision", "recall"],
          [(r["threshold"], r["F_0.5"], r["precision"], r["recall"]) for r in sweep_rows])
print(f"  Threshold sweep saved to threshold_sweep.csv")

print(f"\n[DONE] Training complete. Val F_0.5 = {best_f05:.4f} at threshold = {t_opt:.2f}")
