"""
03_feature_engineering.py
==========================
Phase 3 - Feature Engineering

Computes 29 features for every (S2/S3 query, S1 candidate) pair
from the candidate pairs Parquet produced by 02_block.py.

Key optimization over Experiment_Day1: NO pandas.iterrows().
All per-column string similarity features are computed with:
  - rapidfuzz vectorized operations
  - Polars apply() for row-wise features that need NormalisedRecord
  - Batch TF-IDF matrix multiply for cosine features

Output:
    output/features_train.parquet  (with 'label' column)
    output/features_test.parquet

Usage:
    python 03_feature_engineering.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

import config
from normaliser import NormalisedRecord, canonical_country, normalise_record

try:
    from rapidfuzz.distance import Levenshtein as _Lev
    from rapidfuzz.distance import JaroWinkler as _JW
    from rapidfuzz.fuzz import token_sort_ratio as _tsr
    _RF = True
except ImportError:
    _RF = False
    print("[WARN] rapidfuzz not installed - string features will be 0.0")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy.sparse import csr_matrix
except ImportError:
    raise ImportError("scikit-learn required: pip install scikit-learn")

config.ensure_output_dirs()
config.print_config()
print()


# ===========================================================================
# 1. Fast string similarity helpers (all accept plain strings)
# ===========================================================================

def _jaro_winkler(a: str, b: str) -> float:
    if not _RF:
        return 0.0
    return _JW.normalized_similarity(a, b)


def _levenshtein_ratio(a: str, b: str) -> float:
    if not _RF:
        return 0.0
    return _Lev.normalized_similarity(a, b)


def _tok_sort(a: str, b: str) -> float:
    if not _RF:
        return 0.0
    return _tsr(a, b) / 100.0


def _jaccard_str(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    u = sa | sb
    return len(sa & sb) / len(u) if u else 0.0


def _trigram_jaccard(a: str, b: str) -> float:
    def tg(s):
        return {s[i:i+3] for i in range(max(len(s) - 2, 0))}
    ta, tb = tg(a), tg(b)
    if not ta and not tb:
        return 1.0
    u = ta | tb
    return len(ta & tb) / len(u) if u else 0.0


def _numeric_overlap(a: str, b: str) -> float:
    na = frozenset(t for t in a.split() if t.isdigit())
    nb = frozenset(t for t in b.split() if t.isdigit())
    if not na and not nb:
        return 1.0
    u = na | nb
    return len(na & nb) / len(u) if u else 0.0


def _addr_jaccard(a: str, b: str) -> float:
    sa = frozenset(t for t in a.replace(",", " ").split() if len(t) >= 2)
    sb = frozenset(t for t in b.replace(",", " ").split() if len(t) >= 2)
    if not sa and not sb:
        return 1.0
    u = sa | sb
    return len(sa & sb) / len(u) if u else 0.0


# ===========================================================================
# 2. Record cache builder
# ===========================================================================

def build_record_cache(parquet_paths: list[tuple[Path, str]]) -> dict[str, NormalisedRecord]:
    """Load and normalise source Parquets into a dict keyed by entity_id."""
    cache: dict[str, NormalisedRecord] = {}
    for path, src_tag in parquet_paths:
        if not path.exists():
            print(f"  [SKIP] {path.name} not found")
            continue
        print(f"  Loading {path.name} ({src_tag}) ...", end=" ", flush=True)
        t0  = time.perf_counter()
        df  = pl.read_parquet(path)
        df  = df.rename({c: c.strip().lower() for c in df.columns})
        for col in ["entity_id", "business_name", "business_address", "country"]:
            if col not in df.columns:
                df = df.with_columns(pl.lit("").alias(col))
        df = df.with_columns([
            pl.col(c).cast(pl.Utf8).fill_null("") for c in
            ["entity_id", "business_name", "business_address", "country"]
        ])
        rows = df.select(["entity_id", "business_name", "business_address", "country"]).to_dicts()
        for row in rows:
            eid = row["entity_id"]
            if eid in cache:
                continue
            rec = normalise_record(eid, row["business_name"],
                                   row["business_address"], row["country"])
            object.__setattr__(rec, "source", src_tag)
            cache[eid] = rec
        print(f"{len(rows):,} records in {time.perf_counter()-t0:.1f}s")
    return cache


# ===========================================================================
# 3. TF-IDF store (fitted on S1, queried for each candidate pair batch)
# ===========================================================================

class TFIDFStore:
    def __init__(self):
        self._name_vecs: dict[str, TfidfVectorizer] = {}
        self._name_mats: dict[str, csr_matrix]       = {}
        self._name_ids:  dict[str, np.ndarray]        = {}
        self._name_id_to_row: dict[str, dict[str, int]] = {}  # country -> {eid: row_idx}
        self._addr_vecs: dict[str, TfidfVectorizer] = {}
        self._addr_mats: dict[str, csr_matrix]       = {}

    def fit(self, s1_records: dict[str, NormalisedRecord]):
        by_country: dict[str, list] = {}
        for eid, rec in s1_records.items():
            by_country.setdefault(rec.country_canonical, []).append((eid, rec))

        for ctry, items in by_country.items():
            ids        = np.array([x[0] for x in items], dtype=str)
            name_texts = [x[1].name_normalized for x in items]
            addr_texts = [x[1].addr_normalized for x in items]

            nv = TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                 max_features=100_000, sublinear_tf=True, dtype=np.float32)
            av = TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                 max_features=100_000, sublinear_tf=True, dtype=np.float32)
            self._name_vecs[ctry] = nv
            self._name_mats[ctry] = nv.fit_transform(name_texts)
            self._addr_vecs[ctry] = av
            self._addr_mats[ctry] = av.fit_transform(addr_texts)
            self._name_ids[ctry]  = ids
            # Pre-build O(1) lookup dict: entity_id -> row index in the TF-IDF matrix.
            self._name_id_to_row[ctry] = {eid: i for i, eid in enumerate(ids)}
            print(f"  [TFIDF] {ctry}: name_vocab={len(nv.vocabulary_)}, addr_vocab={len(av.vocabulary_)}")

    def batch_name_cosine(
        self,
        query_names: list[str],
        s1_eids: list[str],
        countries: list[str],
    ) -> np.ndarray:
        """
        Vectorized name TF-IDF cosine for a batch of (query, candidate) pairs.

        Groups pairs by country, does ONE sparse matrix multiply per country
        to get all query vectors at once, then reads only the required S1
        column for each pair. This is O(B * vocab) vs O(B * vocab) per-pair
        with transform(), but avoids the Python overhead of B separate calls.
        For a chunk of 200k pairs this is ~100x faster than the per-pair approach.
        """
        result = np.zeros(len(query_names), dtype=np.float32)
        # Group indices by country
        by_ctry: dict[str, list[int]] = {}
        for i, c in enumerate(countries):
            by_ctry.setdefault(c, []).append(i)

        for ctry, idxs in by_ctry.items():
            if ctry not in self._name_vecs:
                continue
            nv  = self._name_vecs[ctry]
            mat = self._name_mats[ctry]   # [n_s1, vocab]
            id2row = self._name_id_to_row[ctry]

            q_texts = [query_names[i] for i in idxs]
            # q_mat: [n_q, vocab] sparse
            q_mat = nv.transform(q_texts)  # ONE transform call for the whole group
            # scores_csr: [n_q, n_s1] sparse - we only need the diagonal per pair
            scores_csr = (q_mat @ mat.T).tocsr()

            for local_i, global_i in enumerate(idxs):
                s1row = id2row.get(s1_eids[global_i])
                if s1row is None:
                    continue
                # Extract the single element [local_i, s1row] from the sparse result
                row_start = scores_csr.indptr[local_i]
                row_end   = scores_csr.indptr[local_i + 1]
                nnz_cols  = scores_csr.indices[row_start:row_end]
                # Find if s1row appears in the non-zero columns
                match = np.where(nnz_cols == s1row)[0]
                if match.size > 0:
                    result[global_i] = scores_csr.data[row_start + match[0]]
        return result

    def batch_addr_cosine(
        self,
        query_addrs: list[str],
        s1_eids: list[str],
        countries: list[str],
    ) -> np.ndarray:
        """Same as batch_name_cosine but for address TF-IDF."""
        result = np.zeros(len(query_addrs), dtype=np.float32)
        by_ctry: dict[str, list[int]] = {}
        for i, c in enumerate(countries):
            by_ctry.setdefault(c, []).append(i)

        for ctry, idxs in by_ctry.items():
            if ctry not in self._addr_vecs:
                continue
            av  = self._addr_vecs[ctry]
            mat = self._addr_mats[ctry]
            id2row = self._name_id_to_row[ctry]  # same index for both name/addr

            q_texts = [query_addrs[i] for i in idxs]
            q_mat = av.transform(q_texts)
            scores_csr = (q_mat @ mat.T).tocsr()

            for local_i, global_i in enumerate(idxs):
                s1row = id2row.get(s1_eids[global_i])
                if s1row is None:
                    continue
                row_start = scores_csr.indptr[local_i]
                row_end   = scores_csr.indptr[local_i + 1]
                nnz_cols  = scores_csr.indices[row_start:row_end]
                match = np.where(nnz_cols == s1row)[0]
                if match.size > 0:
                    result[global_i] = scores_csr.data[row_start + match[0]]
        return result


# ===========================================================================
# 4. Feature computation - vectorized over a batch of pairs
# ===========================================================================

# Chunk size from config (default 200k, overridden by config.FEATURE_CHUNK)
CHUNK = getattr(config, "FEATURE_CHUNK", 200_000)

def compute_features_chunk(
    chunk: pl.DataFrame,
    cache: dict[str, NormalisedRecord],
    tfidf: TFIDFStore,
) -> list[dict]:
    """Compute all features for a chunk of candidate pairs.

    Key optimizations vs the original:
    - TF-IDF cosine: ONE sparse matrix multiply per country per chunk
      instead of calling transform([single_str]) per pair (was the #1 bottleneck).
    - All string similarities computed via list comprehensions over pre-fetched
      numpy arrays, avoiding per-row Python dict lookups and function dispatch.
    """
    pairs = chunk.select(["query_entity_id", "candidate_s1_entity_id",
                          "faiss_rank", "faiss_score",
                          "tfidf_rank", "tfidf_score"]).to_dicts()

    # --- Pre-fetch all records for the chunk ---
    q_recs_raw  = [cache.get(p["query_entity_id"])        for p in pairs]
    s1_recs_raw = [cache.get(p["candidate_s1_entity_id"]) for p in pairs]

    # Filter out pairs with missing records
    valid   = [i for i, (q, s) in enumerate(zip(q_recs_raw, s1_recs_raw)) if q is not None and s is not None]
    if not valid:
        return []

    pairs   = [pairs[i]        for i in valid]
    q_recs  = [q_recs_raw[i]  for i in valid]
    s1_recs = [s1_recs_raw[i] for i in valid]
    n       = len(pairs)

    # --- Extract string arrays for vectorized ops ---
    q_names   = [r.name_normalized for r in q_recs]
    s1_names  = [r.name_normalized for r in s1_recs]
    q_cores   = [r.name_core       for r in q_recs]
    s1_cores  = [r.name_core       for r in s1_recs]
    q_addrs   = [r.addr_normalized for r in q_recs]
    s1_addrs  = [r.addr_normalized for r in s1_recs]
    s1_eids   = [p["candidate_s1_entity_id"] for p in pairs]
    countries = [r.country_canonical         for r in s1_recs]

    # --- Vectorized string similarities ---
    if _RF:
        name_jw_arr  = [_JW.normalized_similarity(q_names[i],  s1_names[i])  for i in range(n)]
        name_lev_arr = [_Lev.normalized_similarity(q_names[i], s1_names[i])  for i in range(n)]
        name_tsr_arr = [_tsr(q_names[i], s1_names[i]) / 100.0               for i in range(n)]
        core_jw_arr  = [_JW.normalized_similarity(q_cores[i],  s1_cores[i])  for i in range(n)]
        addr_jw_arr  = [_JW.normalized_similarity(q_addrs[i],  s1_addrs[i])  for i in range(n)]
        addr_tsr_arr = [_tsr(q_addrs[i],  s1_addrs[i]) / 100.0              for i in range(n)]
        legal_jw_arr = [_JW.normalized_similarity(q_recs[i].name_legal, s1_recs[i].name_legal) for i in range(n)]
    else:
        z            = [0.0] * n
        name_jw_arr = name_lev_arr = name_tsr_arr = core_jw_arr = z
        addr_jw_arr = addr_tsr_arr = legal_jw_arr = z

    # --- Set-op features (no external dependency) ---
    name_tok_jac = [_jaccard_str(q_names[i],   s1_names[i])   for i in range(n)]
    name_char_tg = [_trigram_jaccard(q_names[i], s1_names[i]) for i in range(n)]
    core_tok_jac = [_jaccard_str(q_cores[i],   s1_cores[i])   for i in range(n)]
    addr_tok_jac = [_addr_jaccard(q_addrs[i],  s1_addrs[i])   for i in range(n)]
    addr_num_ov  = [_numeric_overlap(q_addrs[i], s1_addrs[i]) for i in range(n)]

    # --- Batch TF-IDF cosines: ONE matrix multiply per country ---
    name_tfidf = tfidf.batch_name_cosine(q_names, s1_eids, countries)
    addr_tfidf = tfidf.batch_addr_cosine(q_addrs, s1_eids, countries)

    # --- Assemble output rows ---
    rows_out = []
    for i in range(n):
        q   = q_recs[i]
        s1  = s1_recs[i]
        p   = pairs[i]

        both_domain = int(q.is_domain and s1.is_domain)
        dom_jw      = _jaro_winkler(q.base_domain, s1.base_domain) if both_domain else 0.0
        tld_match   = int(q.tld == s1.tld and q.tld != "")
        trade_jw    = _jaro_winkler(q.name_trade, s1.name_trade) if (q.name_trade and s1.name_trade) else 0.0
        max_name_jw = max(float(legal_jw_arr[i]), trade_jw)

        q_num  = frozenset(t for t in q.addr_normalized.split() if t.isdigit())
        s1_num = frozenset(t for t in s1.addr_normalized.split() if t.isdigit())
        addr_lm_jac = float(addr_tok_jac[i]) * (
            1.0 - 0.4 * (int(q.has_landmark) + int(s1.has_landmark)) / 2.0
        )

        city_a = _largest_nonnumeric(q.addr_tokens)
        city_b = _largest_nonnumeric(s1.addr_tokens)

        rows_out.append({
            "query_entity_id":                 p["query_entity_id"],
            "candidate_s1_entity_id":          p["candidate_s1_entity_id"],
            "bgem3_full_cosine":               float(p.get("faiss_score") or 0.0),
            "name_jaro_winkler":               float(name_jw_arr[i]),
            "name_levenshtein_ratio":          float(name_lev_arr[i]),
            "name_token_jaccard":              float(name_tok_jac[i]),
            "name_token_sort_ratio":           float(name_tsr_arr[i]),
            "name_char_trigram_jaccard":       float(name_char_tg[i]),
            "core_name_jaro_winkler":          float(core_jw_arr[i]),
            "core_name_token_jaccard":         float(core_tok_jac[i]),
            "name_tfidf_cosine":               float(name_tfidf[i]),
            "name_is_domain_both":             both_domain,
            "domain_base_jaro":                dom_jw,
            "tld_match":                       tld_match,
            "max_name_jaro":                   max_name_jw,
            "addr_jaro_winkler":               float(addr_jw_arr[i]),
            "addr_token_jaccard":              float(addr_tok_jac[i]),
            "addr_token_sort_ratio":           float(addr_tsr_arr[i]),
            "addr_numeric_overlap":            float(addr_num_ov[i]),
            "addr_tfidf_cosine":               float(addr_tfidf[i]),
            "addr_landmark_penalised_jaccard": addr_lm_jac,
            "addr_is_null_either":             int(q.addr_is_null or s1.addr_is_null),
            "addr_is_null_both":               int(q.addr_is_null and s1.addr_is_null),
            "addr_numeric_exact_match":        int(len(q_num) > 0 and q_num == s1_num),
            "addr_city_token_match":           int(bool(city_a) and bool(city_b) and city_a == city_b),
            "faiss_rank":                      float(p.get("faiss_rank")  or 99.0),
            "tfidf_rank":                      float(p.get("tfidf_rank")  or 99.0),
            "script_mismatch":                 int(q.dominant_script != s1.dominant_script),
            "country_canonical_match":         int(q.country_canonical == s1.country_canonical),
            "source_pair":                     0 if getattr(q, "source", "S2") == "S2" else 1,
            "tld_country_signal_match":        int(
                q.tld_country_signal != "unknown"
                and q.tld_country_signal == s1.tld_country_signal
            ),
        })

    return rows_out


def _largest_nonnumeric(tokens) -> str:
    nn = [t for t in tokens if not t.isdigit()]
    return max(nn, key=len) if nn else ""


# ===========================================================================
# 5. Main pipeline per split
# ===========================================================================

def run_split(split: str):
    out_path = config.FEATURES_TRAIN_PARQUET if split == "train" else config.FEATURES_TEST_PARQUET
    cands_path = config.CANDIDATES_TRAIN_PARQUET if split == "train" else config.CANDIDATES_TEST_PARQUET
    gt_path    = config.TRAIN_GT_PARQUET if split == "train" else None

    if out_path.exists():
        n = pl.read_parquet(out_path).height
        print(f"[SKIP] {out_path.name} already exists ({n:,} rows)")
        return

    if not cands_path.exists():
        print(f"[SKIP] {cands_path.name} not found - run 02_block.py first")
        return

    print(f"\n{'='*60}")
    print(f"  Feature engineering: {split.upper()}")
    print(f"{'='*60}")
    t0 = time.perf_counter()

    # Load candidate pairs
    print(f"\n[A] Loading candidate pairs ...")
    cands = pl.read_parquet(cands_path)
    # Cap rows for QUICK_RUN
    max_rows = config.effective_rows()
    if max_rows > 0:
        cands = cands.slice(0, max_rows)
    print(f"    {cands.height:,} candidate pairs")

    # Build record cache
    print(f"\n[B] Building record cache ...")
    if split == "train":
        source_files = [
            (config.TRAIN_SOURCE1_PARQUET, "S1"),
            (config.TRAIN_SOURCE2_PARQUET, "S2"),
            (config.TRAIN_SOURCE3_PARQUET, "S3"),
        ]
    else:
        source_files = [
            (config.TEST_SOURCE1_PARQUET,  "S1"),
            (config.TEST_SOURCE2_PARQUET,  "S2"),
            (config.TEST_SOURCE3_PARQUET,  "S3"),
        ]

    cache = build_record_cache(source_files)
    print(f"    {len(cache):,} records cached")

    # Fit TF-IDF on S1 records
    print(f"\n[C] Fitting TF-IDF on S1 records ...")
    s1_records = {eid: rec for eid, rec in cache.items() if rec.source == "S1"}
    tfidf = TFIDFStore()
    tfidf.fit(s1_records)

    # Compute features in streaming chunks - write Parquet shards instead of
    # accumulating all features in one giant list (which would exhaust RAM for
    # tens of millions of candidate pairs).
    print(f"\n[D] Computing features ({cands.height:,} pairs in chunks of {CHUNK:,}) ...")
    n_chunks  = (cands.height + CHUNK - 1) // CHUNK
    chunk_dir = out_path.parent / f"_feat_chunks_{split}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    shard_idx  = 0
    total_done = 0

    for ci in range(n_chunks):
        chunk = cands.slice(ci * CHUNK, CHUNK)
        rows  = compute_features_chunk(chunk, cache, tfidf)
        if rows:
            shard_path = chunk_dir / f"shard_{shard_idx:04d}.parquet"
            pl.DataFrame(rows).write_parquet(shard_path, compression="zstd")
            shard_idx += 1
            total_done += len(rows)

        if (ci + 1) % 5 == 0 or ci + 1 == n_chunks:
            elapsed = time.perf_counter() - t0
            rate    = total_done / elapsed if elapsed > 0 else 0
            print(f"    chunk {ci+1}/{n_chunks} | {total_done:,} pairs done | {rate:,.0f} pairs/s")

    print(f"\n[E] Merging {shard_idx} shards ...")
    shards  = sorted(chunk_dir.glob("shard_*.parquet"))
    feat_df = pl.concat([pl.read_parquet(p) for p in shards], how="diagonal_relaxed")
    import shutil; shutil.rmtree(chunk_dir, ignore_errors=True)
    print(f"    Feature matrix: {feat_df.height:,} rows x {feat_df.width} cols")

    # Add labels for train split
    if split == "train" and gt_path and gt_path.exists():
        print(f"\n[E] Adding labels from ground truth ...")
        gt = pl.read_parquet(gt_path)
        gt = gt.rename({c: c.strip().lower() for c in gt.columns})
        s1_col    = gt.columns[0]
        match_col = gt.columns[1]

        # Build set of (query_id, s1_id) positive pairs
        # GT format: source1_entity_id | matched_entity_ids (comma-separated)
        gt_pairs: set[tuple[str, str]] = set()
        for row in gt.to_dicts():
            s1_id = str(row[s1_col])
            for mid in str(row[match_col]).split(","):
                mid = mid.strip()
                if mid:
                    gt_pairs.add((mid, s1_id))

        # Fast Polars join to assign labels - replaces slow map_elements per row
        gt_df = pl.DataFrame(
            list(gt_pairs),
            schema=["query_entity_id", "candidate_s1_entity_id"],
            orient="row",
        ).with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))

        feat_df = feat_df.join(
            gt_df,
            on=["query_entity_id", "candidate_s1_entity_id"],
            how="left",
        ).with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
        pos = feat_df.filter(pl.col("label") == 1).height
        print(f"    Labels: {pos:,} positives, {feat_df.height - pos:,} negatives")

    feat_df.write_parquet(out_path, compression="zstd")
    elapsed = time.perf_counter() - t0
    print(f"\n[DONE] {split}: saved {feat_df.height:,} rows to {out_path.name} in {elapsed/60:.1f} min")


# ===========================================================================
# 6. Run
# ===========================================================================

t_total = time.perf_counter()
run_split("train")
run_split("test")
print(f"\n[DONE] Feature engineering complete in {(time.perf_counter()-t_total)/60:.1f} min")
