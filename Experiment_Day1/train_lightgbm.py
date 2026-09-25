"""
train_lightgbm.py
=================
Phase 6 — LightGBM Stage 1 Classifier (runs on Notebook CPU, free tier)

Strategy reference: ER_Strategy.md §8 (Two-stage model), §9 (Validation)

What this module does
---------------------
1. Loads features_train.parquet produced by feature_engineering.py.
2. Trains a LightGBM binary classifier on the 30 feature columns.
3. Performs triple validation (ER_Strategy.md §9):
     Eval 1: Stratified 80/20 split by country → overall F_0.5
     Eval 2: US-train → India-validate → France generalisation proxy
4. Sweeps thresholds 0.30–0.95 on the validation set to maximise F_0.5
   and confirms / adjusts T_low=0.60, T_high=0.90.
5. Saves:
     lgbm_model.pkl         — trained LightGBM model
     uncertain_pairs.tsv   — pairs with 0.60 ≤ P_lgbm ≤ 0.90 (reranker input)
     matching_lgbm.tsv     — Stage-1-only matches (P_lgbm > T_high) for reference

Usage (called from main.ipynb)
-------------------------------
    import train_lightgbm

    results = train_lightgbm.run(
        features_path="output/features_train.parquet",
        candidates_path="output/candidate_pairs_train_with_tfidf.tsv",
        output_dir="output",
        t_low=0.60,
        t_high=0.90,
    )
    # results["val_f05"], results["t_opt"], results["uncertain_count"]
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature columns (must match feature_engineering.compute_pair_features keys)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Name (15)
    "bgem3_name_cosine",
    "bgem3_full_cosine",
    "name_jaro_winkler",
    "name_levenshtein_ratio",
    "name_token_jaccard",
    "name_token_sort_ratio",
    "name_char_trigram_jaccard",
    "core_name_jaro_winkler",
    "core_name_token_jaccard",
    "name_tfidf_cosine",
    "name_is_domain_both",
    "domain_base_jaro",
    "tld_match",
    "max_name_jaro",
    "name_dedup_ratio",
    # Address (10)
    "addr_jaro_winkler",
    "addr_token_jaccard",
    "addr_token_sort_ratio",
    "addr_numeric_overlap",
    "addr_tfidf_cosine",
    "addr_landmark_penalised_jaccard",
    "addr_is_null_either",
    "addr_is_null_both",
    "addr_numeric_exact_match",
    "addr_city_token_match",
    # Global (7)
    "faiss_rank",
    "tfidf_rank",
    "script_mismatch",
    "country_canonical_match",
    "source_pair",
    "tld_country_signal_match",
]

LABEL_COL = "label"
ID_COLS = ["query_entity_id", "candidate_s1_entity_id"]

# LightGBM defaults — tuned for F_0.5 (precision-heavy)
DEFAULT_LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "scale_pos_weight": 3,   # compensates for 1:3 positive-to-negative ratio
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}


# ===========================================================================
# 1. Metric helpers
# ===========================================================================


def f_beta(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5) -> float:
    """F_beta score.  beta=0.5 weights precision 2x over recall (ER_Strategy §0)."""
    return float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    beta: float = 0.5,
) -> Tuple[float, float, pd.DataFrame]:
    """Sweep thresholds and return (best_threshold, best_f05, sweep_df)."""
    if thresholds is None:
        thresholds = np.arange(0.30, 0.96, 0.01)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f05 = f_beta(y_true, y_pred, beta=beta)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        rows.append({"threshold": round(float(t), 2), "F_0.5": f05, "precision": prec, "recall": rec})
    sweep_df = pd.DataFrame(rows)
    best_row = sweep_df.loc[sweep_df["F_0.5"].idxmax()]
    return float(best_row["threshold"]), float(best_row["F_0.5"]), sweep_df


# ===========================================================================
# 2. Country label extraction from features
# ===========================================================================


def _country_label_series(df: pd.DataFrame) -> pd.Series:
    """Extract per-pair country label for stratification.

    Uses country_canonical_match column as a proxy: we do not store the actual
    country string in features_train.parquet by default, so we reconstruct a
    coarse country bucket from the candidate pairs TSV.

    Falls back to a single bucket if country info is not available.
    """
    if "country" in df.columns:
        return df["country"]
    # Fallback: use source_pair (0=S2, 1=S3) as stratification label
    return df.get("source_pair", pd.Series(np.zeros(len(df), dtype=int)))


# ===========================================================================
# 3. Training function
# ===========================================================================


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    lgb_params: Optional[dict] = None,
) -> "lgb.LGBMClassifier":
    """Train LightGBM and return fitted model."""
    if not _LGB_AVAILABLE:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

    params = {**DEFAULT_LGB_PARAMS, **(lgb_params or {})}
    log.info("Training LightGBM: %d train, %d val pairs", len(X_train), len(X_val))
    log.info("Positive rate — train: %.3f%%  val: %.3f%%",
             y_train.mean() * 100, y_val.mean() * 100)

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    log.info("Best iteration: %d", model.best_iteration_)
    return model


# ===========================================================================
# 4. Main run function
# ===========================================================================


def run(
    features_path: str,
    output_dir: str,
    candidates_path: Optional[str] = None,
    t_low: float = 0.60,
    t_high: float = 0.90,
    lgb_params: Optional[dict] = None,
    val_frac: float = 0.20,
) -> dict:
    """Full Phase 6 pipeline.

    Parameters
    ----------
    features_path:
        Path to features_train.parquet.
    output_dir:
        Directory for saved artefacts.
    candidates_path:
        Optional merged candidate pairs TSV (used to enrich uncertain_pairs
        output with query/s1 entity IDs).
    t_low:
        Low threshold — pairs with P < t_low are auto-rejected.
    t_high:
        High threshold — pairs with P > t_high are auto-accepted.
    lgb_params:
        Optional dict of overrides to DEFAULT_LGB_PARAMS.
    val_frac:
        Fraction held out for validation (Eval 1).

    Returns
    -------
    dict with keys: val_f05, t_opt, uncertain_count, model, sweep_df,
                    india_f05 (Eval 2), feature_importance
    """
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 4A. Load features
    # ------------------------------------------------------------------
    log.info("Loading features from %s ...", features_path)
    df = pd.read_parquet(features_path)
    log.info("Loaded %d rows, %d columns.", len(df), len(df.columns))

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"'{LABEL_COL}' column missing from {features_path}. "
            "Run feature_engineering.run() with ground_truth_path."
        )

    # Restrict to feature columns that actually exist in the file
    present_feats = [f for f in FEATURE_COLS if f in df.columns]
    missing = [f for f in FEATURE_COLS if f not in df.columns]
    if missing:
        log.warning("Missing feature columns (will use 0.0): %s", missing)
        for f in missing:
            df[f] = 0.0

    X = df[FEATURE_COLS].fillna(0.0).astype(np.float32)
    y = df[LABEL_COL].astype(int)

    # ------------------------------------------------------------------
    # 4B. Eval 1 — Stratified 80/20 split (country-stratified if possible)
    # ------------------------------------------------------------------
    log.info("=== Eval 1: Stratified 80/20 split ===")
    strat_label = _country_label_series(df)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
    train_idx, val_idx = next(splitter.split(X, strat_label))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = train(X_train, y_train, X_val, y_val, lgb_params)

    y_val_prob = model.predict_proba(X_val)[:, 1]
    t_opt, best_f05, sweep_df = threshold_sweep(y_val.values, y_val_prob)

    log.info("Eval 1 — best threshold: %.2f, F_0.5: %.4f", t_opt, best_f05)
    log.info("Strategy thresholds: T_low=%.2f, T_high=%.2f", t_low, t_high)

    # Print top thresholds for inspection
    top_rows = sweep_df.nlargest(5, "F_0.5")
    log.info("Top-5 thresholds by F_0.5:\n%s", top_rows.to_string(index=False))

    # ------------------------------------------------------------------
    # 4C. Eval 2 — US-train → India-validate (France generalisation proxy)
    # ------------------------------------------------------------------
    india_f05 = None
    if "country" in df.columns:
        log.info("=== Eval 2: US-train / India-validate ===")
        us_mask = df["country"] == "US"
        india_mask = df["country"] == "India"
        if us_mask.sum() > 0 and india_mask.sum() > 0:
            X_us = X[us_mask]
            y_us = y[us_mask]
            X_india = X[india_mask]
            y_india = y[india_mask]
            model_us = train(X_us, y_us, X_india, y_india, lgb_params)
            y_india_prob = model_us.predict_proba(X_india)[:, 1]
            _, india_f05, _ = threshold_sweep(y_india.values, y_india_prob)
            log.info("Eval 2 — India F_0.5 (US-trained): %.4f", india_f05)
            if india_f05 < 0.70:
                log.warning(
                    "Eval 2 India F_0.5 (%.4f) < 0.70 — France generalisation may be poor. "
                    "Consider adding more cross-country features or threshold tuning.",
                    india_f05,
                )
            elif india_f05 >= 0.80:
                log.info("Eval 2 India F_0.5 >= 0.80 — France generalisation looks good.")
        else:
            log.warning("Eval 2 skipped — not enough US or India rows in features.")
    else:
        log.info("Eval 2 skipped — 'country' column not present in features.")

    # ------------------------------------------------------------------
    # 4D. Retrain on 100% data for final model (Eval 3 setup)
    # ------------------------------------------------------------------
    log.info("Retraining on 100%% data for final model...")
    model_full = train(X, y, X_val, y_val, lgb_params)

    # ------------------------------------------------------------------
    # 4E. Predict on all pairs and produce uncertain / match outputs
    # ------------------------------------------------------------------
    log.info("Scoring all %d candidate pairs...", len(df))
    all_prob = model_full.predict_proba(X)[:, 1]
    df = df.copy()
    df["p_lgbm"] = all_prob

    uncertain_mask = (all_prob >= t_low) & (all_prob <= t_high)
    accept_mask = all_prob > t_high
    reject_mask = all_prob < t_low

    log.info(
        "Pair disposition: REJECT=%d (%.1f%%), UNCERTAIN=%d (%.1f%%), ACCEPT=%d (%.1f%%)",
        reject_mask.sum(), reject_mask.mean() * 100,
        uncertain_mask.sum(), uncertain_mask.mean() * 100,
        accept_mask.sum(), accept_mask.mean() * 100,
    )

    # Uncertain pairs — feed to reranker
    uncertain_df = df[uncertain_mask][ID_COLS + ["p_lgbm"]].copy()
    uncertain_path = os.path.join(output_dir, "uncertain_pairs.tsv")
    uncertain_df.to_csv(uncertain_path, sep="\t", index=False)
    log.info("Saved %d uncertain pairs to %s", len(uncertain_df), uncertain_path)

    # Stage-1-only matches (high confidence)
    match_df = df[accept_mask][ID_COLS + ["p_lgbm"]].copy()
    match_path = os.path.join(output_dir, "matching_lgbm.tsv")
    match_df.to_csv(match_path, sep="\t", index=False)
    log.info("Saved %d Stage-1 accepted matches to %s", len(match_df), match_path)

    # ------------------------------------------------------------------
    # 4F. Feature importance
    # ------------------------------------------------------------------
    fi = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance_gain": model_full.booster_.feature_importance(importance_type="gain"),
        "importance_split": model_full.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    fi_path = os.path.join(output_dir, "feature_importance.tsv")
    fi.to_csv(fi_path, sep="\t", index=False)
    log.info("Feature importance saved to %s", fi_path)
    log.info("Top-10 features by gain:\n%s", fi.head(10).to_string(index=False))

    # ------------------------------------------------------------------
    # 4G. Save model
    # ------------------------------------------------------------------
    model_path = os.path.join(output_dir, "lgbm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_full, f)
    log.info("Model saved to %s", model_path)

    # Threshold sweep CSV
    sweep_path = os.path.join(output_dir, "threshold_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    log.info("Threshold sweep saved to %s", sweep_path)

    return {
        "model": model_full,
        "val_f05": best_f05,
        "india_f05": india_f05,
        "t_opt": t_opt,
        "t_low_used": t_low,
        "t_high_used": t_high,
        "uncertain_count": int(uncertain_mask.sum()),
        "accept_count": int(accept_mask.sum()),
        "reject_count": int(reject_mask.sum()),
        "sweep_df": sweep_df,
        "feature_importance": fi,
        "model_path": model_path,
        "uncertain_path": uncertain_path,
    }


# ===========================================================================
# 5. Inference helper — used in main.ipynb Phase 7 / test inference
# ===========================================================================


def predict(
    features_path: str,
    model_path: str,
    t_low: float = 0.60,
    t_high: float = 0.90,
    output_dir: str = "output",
) -> pd.DataFrame:
    """Score a features parquet with a saved model and return scored DataFrame.

    Returns a DataFrame with all original columns plus:
        p_lgbm       — LightGBM match probability
        disposition  — "REJECT" / "UNCERTAIN" / "ACCEPT"
    """
    log.info("Loading model from %s ...", model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    log.info("Loading features from %s ...", features_path)
    df = pd.read_parquet(features_path)
    for feat in FEATURE_COLS:
        if feat not in df.columns:
            df[feat] = 0.0
    X = df[FEATURE_COLS].fillna(0.0).astype(np.float32)

    log.info("Scoring %d pairs...", len(df))
    df = df.copy()
    df["p_lgbm"] = model.predict_proba(X)[:, 1]

    def _disp(p):
        if p < t_low:
            return "REJECT"
        if p > t_high:
            return "ACCEPT"
        return "UNCERTAIN"

    df["disposition"] = df["p_lgbm"].apply(_disp)
    counts = df["disposition"].value_counts()
    log.info("Dispositions:\n%s", counts.to_string())

    os.makedirs(output_dir, exist_ok=True)
    uncertain_df = df[df["disposition"] == "UNCERTAIN"][ID_COLS + ["p_lgbm"]]
    uncertain_path = os.path.join(output_dir, "uncertain_pairs_test.tsv")
    uncertain_df.to_csv(uncertain_path, sep="\t", index=False)
    log.info("Saved %d uncertain test pairs to %s", len(uncertain_df), uncertain_path)

    return df


# ===========================================================================
# 6. CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LightGBM Stage 1 — Phase 6")
    parser.add_argument("--features", required=True, help="features_train.parquet path")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--t-low", type=float, default=0.60)
    parser.add_argument("--t-high", type=float, default=0.90)
    parser.add_argument("--val-frac", type=float, default=0.20)
    args = parser.parse_args()

    run(
        features_path=args.features,
        output_dir=args.output_dir,
        t_low=args.t_low,
        t_high=args.t_high,
        val_frac=args.val_frac,
    )
