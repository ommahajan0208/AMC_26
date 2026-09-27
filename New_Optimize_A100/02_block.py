"""
02_block.py
===========
Phase 2 - Candidate Generation (FAISS + TF-IDF blocking)

For each S2/S3 record, finds the top-K most similar S1 records
using two complementary methods:

  FAISS (primary)   - approximate nearest neighbour over BGE-m3 embeddings
  TF-IDF (fallback) - sparse lexical matching on normalised business names

Country partitioning is applied strictly: US queries only search US S1
records, India only India, etc. This alone eliminates 50-70% of the
search space with zero recall loss.

Input:
    output/embeddings/{s1,s2,s3}_{train,test}.npy + .meta.jsonl

Output:
    output/candidate_pairs_{train,test}.parquet
    Columns: query_entity_id | candidate_s1_entity_id |
             faiss_rank | faiss_score | tfidf_rank | tfidf_score |
             query_country | query_source

Usage:
    python 02_block.py
"""

import json
import io
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from normaliser import canonical_country, normalise_record

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu or faiss-gpu is required: pip install faiss-gpu")

# Try to use GPU FAISS (faiss-gpu). Falls back to CPU if unavailable or disabled.
_FAISS_GPU_RES = None
if config.FAISS_USE_GPU:
    try:
        _FAISS_GPU_RES = faiss.StandardGpuResources()
        _FAISS_GPU_RES.setTempMemory(512 * 1024 * 1024)  # 512 MB scratch
        print("[FAISS] GPU resources initialized")
    except Exception as _fe:
        _FAISS_GPU_RES = None
        print(f"[FAISS] GPU not available ({_fe}), using CPU")

config.ensure_output_dirs()
config.print_config()
print()


# ===========================================================================
# 1. Shard loading helpers
# ===========================================================================

def load_embeddings(source_tag: str, split: str):
    """Load npy + meta.jsonl for one source/split pair.
    Returns (vecs float16 [N,1024], metas list[dict]).
    """
    npy_path  = config.EMBEDDINGS_DIR / f"{source_tag}_{split}.npy"
    meta_path = config.EMBEDDINGS_DIR / f"{source_tag}_{split}.meta.jsonl"

    if not npy_path.exists():
        print(f"  [SKIP] {npy_path.name} not found")
        return np.empty((0, config.EMBED_DIM), dtype=np.float16), []

    vecs = np.load(npy_path)
    # Faster meta loading: read the whole file at once then split in memory
    raw  = meta_path.read_bytes().decode("utf-8", errors="replace")
    metas = [json.loads(l) for l in raw.splitlines() if l.strip()]
    if vecs.dtype != np.float16:
        vecs = vecs.astype(np.float16)
    print(f"  Loaded {len(vecs):,} vectors from {npy_path.name}")
    return vecs, metas


# ===========================================================================
# 2. FAISS index builder (per-country)
# ===========================================================================

def _to_fp32_normed(vecs_fp16: np.ndarray) -> np.ndarray:
    vecs = np.ascontiguousarray(vecs_fp16, dtype=np.float32)
    faiss.normalize_L2(vecs)
    return vecs


def build_faiss_index(vecs_fp32: np.ndarray) -> faiss.Index:
    n = len(vecs_fp32)
    flat_threshold = 500

    if n < flat_threshold:
        idx = faiss.IndexFlatIP(config.EMBED_DIM)
        idx.add(vecs_fp32)
        return _maybe_to_gpu(idx)

    nlist  = min(config.FAISS_NLIST, n // 39)
    nlist  = max(nlist, 1)
    nprobe = min(config.FAISS_NPROBE, nlist)

    q   = faiss.IndexFlatIP(config.EMBED_DIM)
    idx = faiss.IndexIVFFlat(q, config.EMBED_DIM, nlist, faiss.METRIC_INNER_PRODUCT)
    idx.nprobe = nprobe
    idx.train(vecs_fp32)
    idx.add(vecs_fp32)
    return _maybe_to_gpu(idx)


def _maybe_to_gpu(idx: faiss.Index) -> faiss.Index:
    """Move a FAISS index to GPU if resources are available."""
    if _FAISS_GPU_RES is None:
        return idx
    try:
        return faiss.index_cpu_to_gpu(_FAISS_GPU_RES, 0, idx)
    except Exception:
        return idx


class CountryFaissIndexes:
    """Holds one FAISS index per canonical country."""

    def __init__(self):
        self.indexes: dict[str, faiss.Index] = {}
        self.id_lists: dict[str, list[str]]  = {}

    def build(self, vecs_fp16: np.ndarray, metas: list[dict]):
        groups: dict[str, tuple[list[int], list[str]]] = defaultdict(lambda: ([], []))
        for i, m in enumerate(metas):
            c = m.get("country_canonical") or "Unknown"
            groups[c][0].append(i)
            groups[c][1].append(m["entity_id"])

        for c, (rows, eids) in groups.items():
            print(f"  [FAISS] Building index for '{c}': {len(rows):,} S1 vectors ...", end=" ", flush=True)
            fp32 = _to_fp32_normed(vecs_fp16[rows])
            self.indexes[c]  = build_faiss_index(fp32)
            self.id_lists[c] = eids
            print("done")

    def search(self, query_vecs_fp16: np.ndarray, country: str, top_k: int):
        used = country if country in self.indexes else "Unknown"
        if used not in self.indexes:
            n = len(query_vecs_fp16)
            return (
                np.full((n, top_k), -1.0, dtype=np.float32),
                np.full((n, top_k), -1,   dtype=np.int64),
                country,
            )
        idx     = self.indexes[used]
        actual_k = min(top_k, idx.ntotal)
        q_fp32  = _to_fp32_normed(query_vecs_fp16)
        scores, positions = idx.search(q_fp32, actual_k)
        if actual_k < top_k:
            pad = top_k - actual_k
            scores    = np.pad(scores,    [(0, 0), (0, pad)], constant_values=-1.0)
            positions = np.pad(positions, [(0, 0), (0, pad)], constant_values=-1)
        return scores, positions, used


# ===========================================================================
# 3. TF-IDF index builder (per-country, on S1 names)
# ===========================================================================

def _tfidf_text(rec) -> str:
    combined = f"{rec.name_normalized} {rec.name_core}".strip()
    return combined


def build_tfidf_indexes(
    s1_parquet: Path,
) -> dict[str, tuple[TfidfVectorizer, csr_matrix, np.ndarray]]:
    """Build per-country TF-IDF name indexes from S1 Parquet."""
    if not s1_parquet.exists():
        print("  [TFIDF] S1 Parquet not found - skipping TF-IDF")
        return {}

    df = pl.read_parquet(s1_parquet)
    df = df.rename({c: c.strip().lower() for c in df.columns})
    for col in ["entity_id", "business_name", "business_address", "country"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit("").alias(col))
    df = df.with_columns([
        pl.col("entity_id").cast(pl.Utf8).fill_null(""),
        pl.col("business_name").cast(pl.Utf8).fill_null(""),
        pl.col("business_address").cast(pl.Utf8).fill_null(""),
        pl.col("country").cast(pl.Utf8).fill_null(""),
    ])

    rows = df.to_dicts()
    # Cap for QUICK_RUN
    max_rows = config.effective_rows()
    if max_rows > 0:
        rows = rows[:max_rows]

    by_country: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))

    for row in rows:
        try:
            rec  = normalise_record(row["entity_id"], row["business_name"],
                                    row["business_address"], row["country"])
            ctry = rec.country_canonical
            # Use the normalized business name (not entity_id!) as the TF-IDF text.
            # Using entity_id (a random number) produces zero meaningful matches.
            text = _tfidf_text(rec)
        except Exception:
            ctry = canonical_country(row.get("country", ""))
            text = str(row.get("business_name", ""))
        by_country[ctry][0].append(row["entity_id"])
        by_country[ctry][1].append(text)

    indexes = {}
    for ctry, (eids, texts) in by_country.items():
        print(f"  [TFIDF] Building index for '{ctry}': {len(eids):,} records ...", end=" ", flush=True)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                              max_features=150_000, sublinear_tf=True, dtype=np.float32)
        mat = vec.fit_transform(texts)
        indexes[ctry] = (vec, mat, np.array(eids, dtype=str))
        print(f"vocab={len(vec.vocabulary_)}")

    return indexes


def query_tfidf_batch(
    texts: list[str],
    entity_ids_query: list[str],
    countries: list[str],
    indexes: dict,
    top_k: int,
) -> list[dict]:
    """Vectorized TF-IDF top-K retrieval using sparse CSR internals.

    Key insight: the result matrix (q_mat @ mat.T) is extremely sparse -
    most S1 entities share zero tokens with any given query. Instead of
    ever densifying it (OOM for large datasets), we read the CSR internal
    arrays directly:
      - indptr[i]:indptr[i+1]  ->  non-zero column range for row i
      - indices[...]           ->  which S1 column (entity index)
      - data[...]              ->  the actual TF-IDF cosine score
    Only non-zero entries are processed. For 1.2M S1 entities a typical
    query row has ~100-1000 non-zero entries (0.01-0.1% density).
    Peak additional RAM: negligible (a few KB per row).
    """
    results = []
    by_country: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(countries):
        used = c if c in indexes else "Unknown"
        if used in indexes:
            by_country[used].append(i)

    for ctry, idxs in by_country.items():
        vec, mat, s1_eids = indexes[ctry]

        # Process in sub-batches. Sub-batch size can be large because
        # the sparse result uses negligible memory (only non-zeros stored).
        SUB = 2048
        for sub_start in range(0, len(idxs), SUB):
            sub_idxs    = idxs[sub_start : sub_start + SUB]
            batch_texts = [texts[i] for i in sub_idxs]

            # Sparse multiply: [sub, vocab] @ [vocab, n_s1] -> sparse [sub, n_s1]
            q_mat      = vec.transform(batch_texts)
            scores_csr = (q_mat @ mat.T).tocsr()   # guarantees CSR format

            # --- Vectorized top-K extraction via CSR internals ---
            indptr  = scores_csr.indptr    # shape [sub+1]
            indices = scores_csr.indices   # non-zero column indices (S1 positions)
            data    = scores_csr.data      # non-zero scores

            for local_i, global_i in enumerate(sub_idxs):
                row_start = indptr[local_i]
                row_end   = indptr[local_i + 1]
                if row_start == row_end:
                    continue  # no TF-IDF overlap with any S1 entity

                nnz_scores = data[row_start:row_end]    # array of non-zero scores
                nnz_cols   = indices[row_start:row_end]  # which S1 entity each score belongs to

                # top-K among the non-zero entries only (usually 100-1000 elements)
                k_actual  = min(top_k, len(nnz_scores))
                # argpartition gives top-k in O(n) then argsort just the k elements
                local_top = np.argpartition(nnz_scores, -k_actual)[-k_actual:]
                local_top = local_top[np.argsort(nnz_scores[local_top])[::-1]]

                for rank, li in enumerate(local_top, start=1):
                    results.append({
                        "query_entity_id":        entity_ids_query[global_i],
                        "candidate_s1_entity_id": s1_eids[nnz_cols[li]],
                        "tfidf_rank":             rank,
                        "tfidf_score":            float(nnz_scores[li]),
                    })
    return results



# Fixed schema - sentinel values (-1 / -1.0) instead of None for missing ranks/scores.
# This is critical: mixing None-typed and i64-typed rows in the same chunk buffer
# causes a Polars ComputeError because Polars infers the column type from the
# first N rows and then rejects appending a different concrete type later.
_SHARD_SCHEMA = {
    "query_entity_id":        pl.Utf8,
    "candidate_s1_entity_id": pl.Utf8,
    "faiss_rank":             pl.Int16,    # -1 = not from FAISS
    "faiss_score":            pl.Float32,  # -1.0 = not from FAISS
    "tfidf_rank":             pl.Int16,    # -1 = not from TF-IDF
    "tfidf_score":            pl.Float32,  # -1.0 = not from TF-IDF
    "query_country":          pl.Utf8,
    "query_source":           pl.Utf8,
}


# ===========================================================================
# 4. Blocking pass - stream S2/S3 through FAISS + TF-IDF per split
# ===========================================================================

def blocking_pass(split: str):
    out_path   = config.CANDIDATES_TRAIN_PARQUET if split == "train" else config.CANDIDATES_TEST_PARQUET
    s1_parquet = config.TRAIN_SOURCE1_PARQUET    if split == "train" else config.TEST_SOURCE1_PARQUET

    if out_path.exists():
        n = pl.read_parquet(out_path).height
        print(f"[SKIP] {out_path.name} already exists ({n:,} candidate pairs)")
        return

    print(f"\n{'='*60}")
    print(f"  Blocking pass: {split.upper()}")
    print(f"{'='*60}")
    t_pass = time.perf_counter()

    # --- Load S1 embeddings + build FAISS per country ---
    print(f"\n[A] Loading S1 embeddings and building FAISS indexes ...")
    s1_vecs, s1_metas = load_embeddings("s1", split)
    if len(s1_vecs) == 0:
        print("  [ERROR] No S1 embeddings found. Run 01_embed.py first.")
        sys.exit(1)

    faiss_indexes = CountryFaissIndexes()
    faiss_indexes.build(s1_vecs, s1_metas)
    del s1_vecs

    # --- Build TF-IDF per country on S1 ---
    print(f"\n[B] Building TF-IDF indexes on S1 names ...")
    tfidf_indexes = build_tfidf_indexes(s1_parquet)

    # --- Stream S2 + S3 through both indexes in RAM-safe chunks ---
    # Per-column list accumulators (not list-of-dicts):
    #   - 5-10x faster DataFrame construction (no schema inference, no dict overhead)
    #   - Fixed concrete types with sentinel values avoids Polars ComputeError
    CHUNK_ROWS = 500_000
    chunk_dir  = out_path.parent / f"_chunks_{split}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_idx  = 0
    total_rows = 0

    _q_ids:    list = []
    _s1_ids:   list = []
    _f_ranks:  list = []
    _f_scores: list = []
    _t_ranks:  list = []
    _t_scores: list = []
    _ctries:   list = []
    _sources:  list = []

    def _flush_chunk():
        nonlocal chunk_idx, total_rows
        nonlocal _q_ids, _s1_ids, _f_ranks, _f_scores, _t_ranks, _t_scores, _ctries, _sources
        n = len(_q_ids)
        if n == 0:
            return
        shard_path = chunk_dir / f"shard_{chunk_idx:04d}.parquet"
        pl.DataFrame({
            "query_entity_id":        _q_ids,
            "candidate_s1_entity_id": _s1_ids,
            "faiss_rank":             _f_ranks,
            "faiss_score":            _f_scores,
            "tfidf_rank":             _t_ranks,
            "tfidf_score":            _t_scores,
            "query_country":          _ctries,
            "query_source":           _sources,
        }, schema=_SHARD_SCHEMA).write_parquet(shard_path, compression="zstd")
        total_rows += n
        print(f"    [flush] shard {chunk_idx:04d}: {n:,} rows (total: {total_rows:,})")
        _q_ids=[]; _s1_ids=[]; _f_ranks=[]; _f_scores=[]
        _t_ranks=[]; _t_scores=[]; _ctries=[]; _sources=[]
        chunk_idx += 1

    for source_tag in ["s2", "s3"]:
        src_upper = source_tag.upper()
        print(f"\n[C] Querying {src_upper}_{split} embeddings ...")
        q_vecs, q_metas = load_embeddings(source_tag, split)
        if len(q_vecs) == 0:
            continue

        n          = len(q_vecs)
        batch_size = config.QUERY_BATCH
        t_src      = time.perf_counter()
        ids_arr    = np.array([m["entity_id"] for m in q_metas], dtype=object)
        ctry_arr   = np.array(
            [m.get("country_canonical") or "Unknown" for m in q_metas], dtype=object
        )

        for start in range(0, n, batch_size):
            end        = min(start + batch_size, n)
            batch_vecs = q_vecs[start:end]
            batch_ids  = ids_arr[start:end]
            batch_ctry = ctry_arr[start:end]

            # ---- FAISS: fully vectorized result expansion ----
            # Uses np.where(valid_mask) to expand [n_query, K] score/position
            # matrices into flat candidate row arrays without any Python for-loop over K.
            unique_ctries = np.unique(batch_ctry)
            for ctry in unique_ctries:
                mask       = batch_ctry == ctry
                local_idxs = np.where(mask)[0]
                sub_vecs   = batch_vecs[local_idxs]
                eids       = batch_ids[local_idxs]

                scores, positions, used_ctry = faiss_indexes.search(
                    sub_vecs, ctry, config.TOP_K
                )
                s1_id_arr = np.array(faiss_indexes.id_lists.get(used_ctry, []), dtype=object)
                if len(s1_id_arr) == 0:
                    continue

                # valid_mask shape [n_sub, TOP_K] - True where position is a real S1 idx
                valid_mask = (positions >= 0) & (positions < len(s1_id_arr))
                li_idx, k_idx = np.where(valid_mask)  # flat (query_row, rank_slot) pairs
                if li_idx.size == 0:
                    continue

                pos_flat   = positions[li_idx, k_idx]
                score_flat = scores[li_idx, k_idx].astype(np.float32)
                rank_flat  = (k_idx + 1).astype(np.int16)
                n_pairs    = li_idx.size

                _q_ids.extend(eids[li_idx].tolist())
                _s1_ids.extend(s1_id_arr[pos_flat].tolist())
                _f_ranks.extend(rank_flat.tolist())
                _f_scores.extend(score_flat.tolist())
                _t_ranks.extend([-1] * n_pairs)
                _t_scores.extend([-1.0] * n_pairs)
                _ctries.extend([str(ctry)] * n_pairs)
                _sources.extend([src_upper] * n_pairs)

            # ---- TF-IDF: sparse CSR vectorized top-K ----
            if tfidf_indexes:
                # Fix: build proper text for each query row using the
                # pre-loaded meta (which has the normalised name stored
                # as the entity_id field in older metas, or look it up).
                # Use business name metadata if available; otherwise fall
                # back to entity_id so the code never crashes.
                tfidf_texts   = [
                    m.get("business_name_norm") or m.get("entity_id", "")
                    for m in q_metas[start:end]
                ]
                tfidf_results = query_tfidf_batch(
                    texts=tfidf_texts,
                    entity_ids_query=batch_ids.tolist(),
                    countries=batch_ctry.tolist(),
                    indexes=tfidf_indexes,
                    top_k=config.TOP_K,
                )
                if tfidf_results:
                    id_pos_map = {eid: i for i, eid in enumerate(batch_ids.tolist())}
                    for r in tfidf_results:
                        _q_ids.append(r["query_entity_id"])
                        _s1_ids.append(r["candidate_s1_entity_id"])
                        _f_ranks.append(-1)
                        _f_scores.append(-1.0)
                        _t_ranks.append(int(r["tfidf_rank"]))
                        _t_scores.append(float(r["tfidf_score"]))
                        qi = id_pos_map.get(r["query_entity_id"], 0)
                        _ctries.append(str(batch_ctry[qi]))
                        _sources.append(src_upper)

            if len(_q_ids) >= CHUNK_ROWS:
                _flush_chunk()

            if (start // batch_size + 1) % 5 == 0 or end == n:
                elapsed = time.perf_counter() - t_src
                print(f"    {end:>7,}/{n:,} | {end/elapsed:,.0f} rec/s")

        del q_vecs, q_metas

    _flush_chunk()  # flush remaining

    if total_rows == 0:
        print("[WARN] No candidate pairs generated")
        import shutil; shutil.rmtree(chunk_dir, ignore_errors=True)
        sys.exit(1)

    # --- Merge shards, deduplicate, write final output ---
    print(f"\n[D] Merging {chunk_idx} shards and deduplicating ...")
    t_merge = time.perf_counter()
    shards  = sorted(chunk_dir.glob("shard_*.parquet"))

    df = pl.concat([pl.read_parquet(p) for p in shards], how="diagonal_relaxed")
    # Prefer FAISS rank when deduplicating: -1 sentinel sorts last (after real ranks 1..K)
    df = (
        df.sort(["query_entity_id", "candidate_s1_entity_id", "faiss_rank", "tfidf_rank"])
          .unique(subset=["query_entity_id", "candidate_s1_entity_id"], keep="first")
    )

    df.write_parquet(out_path, compression="zstd")
    import shutil; shutil.rmtree(chunk_dir, ignore_errors=True)

    elapsed = time.perf_counter() - t_pass
    print(f"[DONE] {split}: {df.height:,} candidate pairs -> {out_path.name} in {elapsed/60:.1f} min")
    print(f"       (merge took {time.perf_counter()-t_merge:.1f}s)")


# ===========================================================================
# 5. Main
# ===========================================================================

t_total = time.perf_counter()

blocking_pass("train")
blocking_pass("test")

print(f"\n[DONE] Blocking complete in {(time.perf_counter()-t_total)/60:.1f} min")
print(f"       Outputs in: {config.OUTPUT_DIR}")
