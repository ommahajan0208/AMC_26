"""
config.py
=========
Central configuration for the Amazon ML Challenge 2026 pipeline.

THE ONLY FILE YOU NEED TO EDIT BETWEEN DEV AND FULL RUNS.

Set WHOLE_DATA = False  to process DEV_ROWS rows per source file (fast dev loop).
Set WHOLE_DATA = True   to process all 26M rows (~3-5 hour full run on A100).

QUICK_RUN = True  runs just QUICK_ROWS rows end-to-end for smoke-testing.
"""

from pathlib import Path

# ===========================================================================
# MAIN TOGGLES
# ===========================================================================

WHOLE_DATA = False   # <-- change to True for the full run
DEV_ROWS   = 2_000_000  # rows per source file when WHOLE_DATA = False
QUICK_RUN  = False   # <-- set True (or override from notebook) for 50-row smoke test
QUICK_ROWS = 50      # number of rows per file used in QUICK_RUN mode

# ===========================================================================
# PATHS
# ===========================================================================

# Root of the project (one level up from this file)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Original TSV files (read-only - never modified)
DATA_DIR  = ROOT_DIR / "student_resource" / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"

# All generated outputs land here
OUTPUT_DIR     = ROOT_DIR / "Experiment_Day2" / "output"
PARQUET_DIR    = OUTPUT_DIR / "parquet"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"

# Source TSV files
TRAIN_SOURCE1_TSV = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2_TSV = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3_TSV = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT_TSV      = TRAIN_DIR / "train_ground_truth.tsv"
TEST_SOURCE1_TSV  = TEST_DIR  / "test_source1.tsv"
TEST_SOURCE2_TSV  = TEST_DIR  / "test_source2.tsv"
TEST_SOURCE3_TSV  = TEST_DIR  / "test_source3.tsv"

# Converted Parquet files (created by 00_convert_to_parquet.py)
TRAIN_SOURCE1_PARQUET = PARQUET_DIR / "train_source1.parquet"
TRAIN_SOURCE2_PARQUET = PARQUET_DIR / "train_source2.parquet"
TRAIN_SOURCE3_PARQUET = PARQUET_DIR / "train_source3.parquet"
TRAIN_GT_PARQUET      = PARQUET_DIR / "train_ground_truth.parquet"
TEST_SOURCE1_PARQUET  = PARQUET_DIR / "test_source1.parquet"
TEST_SOURCE2_PARQUET  = PARQUET_DIR / "test_source2.parquet"
TEST_SOURCE3_PARQUET  = PARQUET_DIR / "test_source3.parquet"

# Pipeline stage outputs
CANDIDATES_TRAIN_PARQUET  = OUTPUT_DIR / "candidate_pairs_train.parquet"
CANDIDATES_TEST_PARQUET   = OUTPUT_DIR / "candidate_pairs_test.parquet"
FEATURES_TRAIN_PARQUET    = OUTPUT_DIR / "features_train.parquet"
FEATURES_TEST_PARQUET     = OUTPUT_DIR / "features_test.parquet"
LGBM_MODEL_PATH           = OUTPUT_DIR / "lgbm_model.pkl"
SUBMISSION_PATH           = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_SUBMISSION_PATH = OUTPUT_DIR / "candidate_pairs_submission.tsv"

# ===========================================================================
# MODEL SETTINGS
# ===========================================================================

EMBED_MODEL      = "BAAI/bge-m3"   # primary embedding model
EMBED_DIM        = 1024             # bge-m3 dense vector dimension
# A100 (80 GB VRAM) at bf16: bge-m3 takes ~1.2 GB, leaving 78 GB for activations.
# batch=1024, max_len=512 uses ~18 GB peak -> safe with headroom for FAISS later.
# If you hit OOM the script auto-halves the batch size and continues.
EMBED_BATCH_SIZE = 1024
EMBED_MAX_LEN    = 512             # max token length for bge-m3

# ===========================================================================
# BLOCKING SETTINGS - tuned for A100 + large dataset
# ===========================================================================

TOP_K         = 20       # candidates per query (FAISS + TF-IDF each)
FAISS_NLIST   = 4096     # IVF cells - appropriate for millions of vectors
FAISS_NPROBE  = 128      # cells probed per search (higher = better recall)
QUERY_BATCH   = 20_000   # vectors per FAISS search call on GPU

# Use faiss-gpu when available (requires: pip install faiss-gpu)
FAISS_USE_GPU = True

# ===========================================================================
# FEATURE ENGINEERING SETTINGS
# ===========================================================================

# Chunk size for feature computation. 200k is safe on Colab (~50 GB RAM).
FEATURE_CHUNK = 200_000

# ===========================================================================
# LIGHTGBM SETTINGS
# ===========================================================================

LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "n_estimators":      1000,
    "learning_rate":     0.05,
    "num_leaves":        127,
    "max_depth":         -1,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "scale_pos_weight":  3,
    "n_jobs":            -1,
    "random_state":      42,
    "verbose":           -1,
    # GPU tree building - uses CUDA via cuML/cuBLAS inside LightGBM
    "device":            "gpu",
    "gpu_platform_id":   0,
    "gpu_device_id":     0,
}

T_LOW  = 0.60   # below this: auto-reject
T_HIGH = 0.90   # above this: auto-accept

# ===========================================================================
# FEATURE COLUMNS (must stay in sync with 03_feature_engineering.py)
# ===========================================================================

FEATURE_COLS = [
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
    "faiss_rank",
    "tfidf_rank",
    "script_mismatch",
    "country_canonical_match",
    "source_pair",
    "tld_country_signal_match",
]


def effective_rows() -> int:
    """Return the row limit for the current run mode. 0 means no limit."""
    if QUICK_RUN:
        return QUICK_ROWS
    if WHOLE_DATA:
        return 0
    return DEV_ROWS


def ensure_output_dirs():
    """Create all output directories if they do not exist."""
    for d in [OUTPUT_DIR, PARQUET_DIR, EMBEDDINGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def print_config():
    """Print current config summary."""
    if QUICK_RUN:
        mode = f"QUICK RUN ({QUICK_ROWS} rows/file)"
    elif WHOLE_DATA:
        mode = "FULL DATA"
    else:
        mode = f"DEV MODE ({DEV_ROWS:,} rows/file)"
    print(f"[config] Mode         : {mode}")
    print(f"[config] Root         : {ROOT_DIR}")
    print(f"[config] Data dir     : {DATA_DIR}")
    print(f"[config] Output dir   : {OUTPUT_DIR}")
    print(f"[config] Embed model  : {EMBED_MODEL}")
    print(f"[config] Batch size   : {EMBED_BATCH_SIZE}")
    print(f"[config] TOP_K        : {TOP_K}")
    print(f"[config] FAISS GPU    : {FAISS_USE_GPU}")
