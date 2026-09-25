"""
build_faiss_and_block.py
========================
SageMaker Training Job entry point — FAISS Index Build + Blocking (Phase 4)

Strategy reference: ER_Strategy.md §2F, §3, §9

What this job does
------------------
1. Reads S1 embedding shards (from train_embeddings.py) in /opt/ml/input/data/.
2. Per canonical country (US / India / France / Unknown), builds a FAISS
   IndexIVFFlat index over the S1 vectors:
       nlist  = 4096  (number of Voronoi cells)
       nprobe = 64    (cells probed at query time; recall@20 ≈ 95%+)
       metric = METRIC_INNER_PRODUCT on L2-normalised fp32 vectors
                → equivalent to cosine similarity
3. Loads S2/S3 embedding shards; for each record queries the same-country S1
   index and emits the top-K=20 candidate S1 entity IDs.
4. Streams all candidate pairs to candidate_pairs_{split}.tsv in
   /opt/ml/output/data/.
   Columns: query_entity_id | candidate_s1_entity_id | faiss_rank | faiss_score
5. (Train split, --eval-recall) Loads train_ground_truth.tsv and reports
   blocking recall against matchable S1 entities only:
       denominator = unique S1 IDs in ground truth (2,083,574)
       NOT total S1 = 2,206,821  (singletons have no GT row — §9 note)

Instance target: ml.g4dn.4xlarge (1×T4 16 GB VRAM, 16 vCPU, 64 GB RAM)
  FAISS index build and search run in CPU RAM — VRAM is unused here.

Memory budget (fp16 → fp32 in FAISS)
  fp16 S1 train shards : 2,206,821 × 1024 × 2 B ≈  4.25 GB (loaded once)
  fp32 S1 train FAISS  : 2,206,821 × 1024 × 4 B ≈  8.50 GB (per-country split)
  fp16 S1 test shards  : 1,732,544 × 1024 × 2 B ≈  3.33 GB
  All fit comfortably in 64 GB. Per-country sharding reduces peak further.

Shard naming (produced by train_embeddings.py):
  {source}_{split}_emb_shard{N:04d}.npy        float16 ndarray [K, 1024]
  {source}_{split}_emb_shard{N:04d}.meta.jsonl  K JSON lines per shard

Usage (called by SageMaker; run twice — once for train, once for test):
  python build_faiss_and_block.py --job-mode train [options]
  python build_faiss_and_block.py --job-mode test  [options]

Key options:
  --job-mode      train|test                         (default: train)
  --top-k         top-K candidates per query         (default: 20 — locked Q6/Q32)
  --nlist         IVF Voronoi cells                  (default: 4096)
  --nprobe        IVF cells probed at search time    (default: 64)
  --query-batch   vectors per FAISS search call      (default: 10000)
  --eval-recall   measure blocking recall vs GT      (train only)
  --gt-file       ground truth filename              (default: train_ground_truth.tsv)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np

# ---------------------------------------------------------------------------
# FAISS — required
# ---------------------------------------------------------------------------
try:
    import faiss  # type: ignore
except ImportError as exc:
    raise ImportError(
        "faiss-cpu is required: pip install faiss-cpu>=1.7.4"
    ) from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SageMaker path constants (overridable via CLI for local testing)
# ---------------------------------------------------------------------------
_SM_INPUT_DIR  = os.environ.get("SM_CHANNEL_DATA",    "/opt/ml/input/data")
_SM_OUTPUT_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------
_DIM             = 1024    # BGE-m3 dense embedding dimension
_IVF_MIN_MULT    = 39      # IVF rule of thumb: ≥ 39 × nlist training vectors
_FLAT_THRESHOLD  = 500     # Below this, skip IVF and use IndexFlatIP


# ===========================================================================
# 1. SHARD LOADING
# ===========================================================================

def _iter_shard_pairs(prefix: str, input_dir: Path) -> list[tuple[Path, Path]]:
    """
    Return sorted (npy_path, meta_jsonl_path) pairs matching
    {input_dir}/{prefix}_emb_shard*.npy.

    Raises FileNotFoundError if a .npy shard is missing its companion .meta.jsonl.
    """
    npy_files = sorted(
        input_dir.glob(f"{prefix}_emb_shard*.npy"),
        key=lambda p: p.name,
    )
    pairs: list[tuple[Path, Path]] = []
    for npy_path in npy_files:
        meta_path = npy_path.with_suffix("").with_suffix(".meta.jsonl")
        # Handle both .npy → .meta.jsonl and .meta.jsonl suffix styles
        if not meta_path.exists():
            # Fallback: strip .npy, append .meta.jsonl
            meta_path = npy_path.parent / (npy_path.stem + ".meta.jsonl")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing companion meta file for {npy_path.name}: "
                f"expected {meta_path.name}"
            )
        pairs.append((npy_path, meta_path))
    return pairs


def _load_meta_jsonl(meta_path: Path) -> list[dict]:
    """Read one .meta.jsonl shard into a list of dicts."""
    metas: list[dict] = []
    with meta_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                metas.append(json.loads(line))
    return metas


def load_shards(
    source: str,    # "s1" / "s2" / "s3"
    split: str,     # "train" / "test"
    input_dir: Path,
) -> tuple[np.ndarray, list[dict]]:
    """
    Concatenate all shards for (source, split) into:
        vecs  — float16 ndarray [N, 1024]
        metas — list of N dicts (entity_id, country_canonical, …)

    Returns (empty array, []) if no shards exist for this prefix.
    """
    prefix      = f"{source}_{split}"
    shard_pairs = _iter_shard_pairs(prefix, input_dir)

    if not shard_pairs:
        log.warning("No shards found for prefix='%s' in %s", prefix, input_dir)
        return np.empty((0, _DIM), dtype=np.float16), []

    log.info("Loading %d shard(s) for '%s' …", len(shard_pairs), prefix)
    t0 = time.perf_counter()

    all_vecs:  list[np.ndarray] = []
    all_metas: list[dict]       = []

    for npy_path, meta_path in shard_pairs:
        vecs = np.load(npy_path)
        if vecs.dtype != np.float16:
            vecs = vecs.astype(np.float16)

        metas = _load_meta_jsonl(meta_path)

        if len(vecs) != len(metas):
            raise ValueError(
                f"Shape mismatch in {npy_path.name}: "
                f"vecs={len(vecs)}  metas={len(metas)}"
            )

        all_vecs.append(vecs)
        all_metas.extend(metas)

    combined = np.concatenate(all_vecs, axis=0)  # float16 [N, 1024]
    log.info(
        "  → %d vectors [%s]  in %.1fs  (%.0f MB fp16)",
        len(combined), prefix,
        time.perf_counter() - t0,
        combined.nbytes / 1e6,
    )
    return combined, all_metas


# ===========================================================================
# 2. FAISS INDEX BUILDER
# ===========================================================================

def _to_fp32_normalised(vecs_fp16: np.ndarray) -> np.ndarray:
    """
    Cast fp16 → fp32 and L2-normalise in-place.
    After normalisation, inner product == cosine similarity.
    """
    vecs = np.ascontiguousarray(vecs_fp16, dtype=np.float32)
    faiss.normalize_L2(vecs)   # in-place, no copy
    return vecs


def _build_ivf_index(
    vecs_fp32: np.ndarray,
    nlist: int,
    nprobe: int,
) -> faiss.Index:
    """
    Build a FAISS IndexIVFFlat (cosine via L2-normalised IP).
    Falls back to IndexFlatIP when the vector count is too small for IVF.

    nlist = 4096, nprobe = 64 per strategy §2F gives recall@20 ≈ 95%+.
    """
    n = len(vecs_fp32)

    if n < _FLAT_THRESHOLD:
        log.warning(
            "  %d vectors < flat_threshold=%d — using IndexFlatIP (brute-force)",
            n, _FLAT_THRESHOLD,
        )
        idx = faiss.IndexFlatIP(_DIM)
        idx.add(vecs_fp32)
        return idx

    min_train = _IVF_MIN_MULT * nlist
    effective_nlist = nlist
    if n < min_train:
        # Shrink nlist so we always have ≥ 39 × nlist training vectors
        effective_nlist = max(1, n // _IVF_MIN_MULT)
        log.warning(
            "  %d vectors < %d (39×nlist) — reducing nlist %d → %d",
            n, min_train, nlist, effective_nlist,
        )

    quantizer = faiss.IndexFlatIP(_DIM)
    index = faiss.IndexIVFFlat(
        quantizer, _DIM, effective_nlist, faiss.METRIC_INNER_PRODUCT
    )
    index.nprobe = min(nprobe, effective_nlist)   # nprobe ≤ nlist

    log.info(
        "  Training IVF index: nlist=%d  nprobe=%d  n=%d",
        effective_nlist, index.nprobe, n,
    )
    t0 = time.perf_counter()
    index.train(vecs_fp32)
    log.info("  Trained in %.1fs — adding vectors …", time.perf_counter() - t0)

    t0 = time.perf_counter()
    index.add(vecs_fp32)
    log.info("  Added %d vectors in %.1fs", n, time.perf_counter() - t0)

    return index


class CountryIndexes:
    """
    One FAISS index per canonical country, each backed by its own entity ID list.

    FAISS internal position → entity_id via country_to_ids[country][position].

    Strategy §2F: US / India / France / Unknown shards.
      - At train time France has 0 S1 vectors → no France index exists.
      - At test time France has ~259K S1 vectors → France index is built.
      - Unknown catches any records with unrecognised country canonical.
    """

    def __init__(self) -> None:
        self.country_to_index: dict[str, faiss.Index] = {}
        self.country_to_ids:   dict[str, list[str]]   = {}

    def build(
        self,
        vecs_fp16: np.ndarray,
        metas: list[dict],
        nlist: int,
        nprobe: int,
    ) -> None:
        """Partition S1 vectors by country_canonical and build one index each."""
        # --- Group row indices by country ---
        groups: dict[str, tuple[list[int], list[str]]] = defaultdict(
            lambda: ([], [])
        )
        for i, meta in enumerate(metas):
            ctry = meta.get("country_canonical") or "Unknown"
            row_idxs, entity_ids = groups[ctry]
            row_idxs.append(i)
            entity_ids.append(meta["entity_id"])

        log.info(
            "S1 country distribution: %s",
            {c: len(v[0]) for c, v in sorted(groups.items())},
        )

        # --- Build one FAISS index per country ---
        for ctry, (row_idxs, entity_ids) in groups.items():
            n = len(row_idxs)
            log.info(
                "Building FAISS index for country='%s'  (%d vectors) …", ctry, n
            )
            ctry_vecs = _to_fp32_normalised(vecs_fp16[row_idxs])
            self.country_to_index[ctry] = _build_ivf_index(
                ctry_vecs, nlist, nprobe
            )
            self.country_to_ids[ctry] = entity_ids

        log.info(
            "All FAISS indexes ready: %s",
            {c: self.country_to_index[c].ntotal for c in self.country_to_index},
        )

    def search(
        self,
        query_vecs_fp16: np.ndarray,   # [N, 1024] float16
        country: str,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Search the index for `country`.  Falls back to "Unknown" if the country
        has no dedicated index (e.g. France at train time).

        Returns:
            scores      — float32 [N, top_k]  cosine similarities
            positions   — int64   [N, top_k]  FAISS internal row indices
            used_country — which country index was actually searched
        """
        used_country = country if country in self.country_to_index else "Unknown"

        if used_country not in self.country_to_index:
            # Truly no usable index — return empty results
            n = len(query_vecs_fp16)
            log.warning(
                "No index available for country='%s' (and no Unknown fallback) — "
                "returning empty candidates for %d records.",
                country, n,
            )
            return (
                np.full((n, top_k), -1.0, dtype=np.float32),
                np.full((n, top_k), -1,   dtype=np.int64),
                country,
            )

        if used_country != country:
            log.debug(
                "No index for '%s' — falling back to '%s'", country, used_country
            )

        index     = self.country_to_index[used_country]
        actual_k  = min(top_k, index.ntotal)
        q_fp32    = _to_fp32_normalised(query_vecs_fp16)
        scores, positions = index.search(q_fp32, actual_k)

        # Pad to top_k if the index is smaller than K
        if actual_k < top_k:
            pad = top_k - actual_k
            scores    = np.pad(scores,    [(0, 0), (0, pad)], constant_values=-1.0)
            positions = np.pad(positions, [(0, 0), (0, pad)], constant_values=-1)

        return scores, positions, used_country

    @property
    def total_s1_records(self) -> int:
        return sum(idx.ntotal for idx in self.country_to_index.values())


# ===========================================================================
# 3. CANDIDATE PAIR WRITER (streaming TSV)
# ===========================================================================

_TSV_HEADER = [
    "query_entity_id",        # S2 or S3 entity ID (queried against S1)
    "candidate_s1_entity_id", # matched S1 candidate
    "faiss_rank",             # 1-based rank in top-K
    "faiss_score",            # cosine similarity (L2-normalised inner product)
    "query_country",          # canonical country of the query record
    "query_source",           # "S2" or "S3"
]


class CandidatePairWriter:
    """
    Streams candidate pairs to a TSV file one batch at a time.
    Never accumulates all rows in memory.
    """

    def __init__(self, output_path: Path) -> None:
        self._fh     = output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh, delimiter="\t", lineterminator="\n")
        self._writer.writerow(_TSV_HEADER)
        self.total_pairs = 0

    def write_batch(
        self,
        query_ids:   list[str],
        scores:      np.ndarray,   # [N, K] float32
        positions:   np.ndarray,   # [N, K] int64
        s1_ids:      list[str],    # country-specific entity_id list (FAISS position → id)
        query_country: str,
        source_tag:  str,          # "S2" / "S3"
    ) -> None:
        for i, qid in enumerate(query_ids):
            for rank_0 in range(scores.shape[1]):
                pos   = int(positions[i, rank_0])
                score = float(scores[i, rank_0])
                if pos < 0 or pos >= len(s1_ids):
                    continue   # FAISS padding or out-of-range
                self._writer.writerow([
                    qid,
                    s1_ids[pos],
                    rank_0 + 1,           # 1-based rank
                    f"{score:.6f}",
                    query_country,
                    source_tag,
                ])
                self.total_pairs += 1

    def close(self) -> None:
        self._fh.close()


# ===========================================================================
# 4. BLOCKING PASS — stream S2/S3 shards through per-country FAISS indexes
# ===========================================================================

def _blocking_pass(
    source_tag:      str,           # "s2" / "s3"
    split:           str,           # "train" / "test"
    input_dir:       Path,
    country_indexes: CountryIndexes,
    writer:          CandidatePairWriter,
    top_k:           int,
    query_batch:     int,
) -> int:
    """
    Load all shards for (source_tag, split), query the per-country FAISS indexes,
    and stream the resulting candidate pairs through `writer`.

    Returns total number of query records processed.
    """
    prefix      = f"{source_tag}_{split}"
    shard_pairs = _iter_shard_pairs(prefix, input_dir)

    if not shard_pairs:
        log.warning("No shards found for '%s' — skipping.", prefix)
        return 0

    log.info("=== Blocking pass: %s  (%d shards) ===", prefix, len(shard_pairs))
    t_pass  = time.perf_counter()
    total   = 0
    logged  = 0

    for shard_idx, (npy_path, meta_path) in enumerate(shard_pairs):
        vecs  = np.load(npy_path).astype(np.float16)   # [K, 1024]
        metas = _load_meta_jsonl(meta_path)

        if len(vecs) != len(metas):
            raise ValueError(
                f"Shard {npy_path.name}: vec count {len(vecs)} ≠ "
                f"meta count {len(metas)}"
            )

        # Accumulate records per country, flush when the per-country buffer
        # hits query_batch.  This keeps individual FAISS searches uniformly
        # batched while respecting the per-country index routing.
        buffers: dict[str, tuple[list[int], list[str]]] = defaultdict(
            lambda: ([], [])
        )

        def _flush(ctry: str) -> None:
            row_idxs, query_ids = buffers[ctry]
            if not row_idxs:
                return
            q_vecs = vecs[np.array(row_idxs, dtype=np.int64)]
            scores, positions, used_ctry = country_indexes.search(
                q_vecs, ctry, top_k
            )
            writer.write_batch(
                query_ids    = query_ids,
                scores       = scores,
                positions    = positions,
                s1_ids       = country_indexes.country_to_ids.get(used_ctry, []),
                query_country = ctry,
                source_tag   = source_tag.upper(),
            )
            row_idxs.clear()
            query_ids.clear()

        for i, meta in enumerate(metas):
            ctry = meta.get("country_canonical") or "Unknown"
            row_idxs, query_ids = buffers[ctry]
            row_idxs.append(i)
            query_ids.append(meta["entity_id"])

            if len(row_idxs) >= query_batch:
                _flush(ctry)

            total += 1

        # Flush tail of each country buffer for this shard
        for ctry in list(buffers):
            _flush(ctry)

        # Progress every 5 shards (= up to 2.5 M records)
        if total - logged >= 2_500_000 or shard_idx + 1 == len(shard_pairs):
            elapsed = time.perf_counter() - t_pass
            rate    = total / elapsed if elapsed > 0 else 0
            log.info(
                "  [%s] shard %d/%d | %7d records | %.0f rec/s | %.1f min",
                prefix, shard_idx + 1, len(shard_pairs),
                total, rate, elapsed / 60,
            )
            logged = total

    elapsed = time.perf_counter() - t_pass
    log.info(
        "=== %s done: %d records in %.1f min  (%.0f rec/s) "
        "| cumulative pairs: %d ===",
        prefix, total, elapsed / 60,
        total / elapsed if elapsed > 0 else 0,
        writer.total_pairs,
    )
    return total


# ===========================================================================
# 5. BLOCKING RECALL EVALUATION  (train split only)
# ===========================================================================

def evaluate_blocking_recall(
    candidate_pairs_path: Path,
    gt_path:              Path,
) -> dict:
    """
    Compute blocking recall against the ground truth.

    Denominator = unique matchable S1 entity IDs in ground truth.
    Singletons (S1 entities with no GT row) are excluded from the denominator.
    See ER_Strategy.md §9 for rationale.

    Streams through the candidate pairs file in chunks to avoid OOM on
    the full ~150–180M row output.

    Returns a dict of recall metrics that is also written to the manifest.
    """
    # --- Load ground truth ---
    log.info("Loading ground truth: %s", gt_path)
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError("pandas is required for --eval-recall") from exc

    gt = pd.read_csv(gt_path, sep="\t", dtype=str)
    gt.columns = [c.strip().lower() for c in gt.columns]

    # Detect columns flexibly (s1_entity_id / s2_entity_id or similar naming)
    s1_col = next(
        (c for c in gt.columns if "s1" in c and "entity" in c),
        next((c for c in gt.columns if "s1" in c), None),
    )
    s23_col = next(
        (c for c in gt.columns
         if ("s2" in c or "s3" in c) and "entity" in c),
        next((c for c in gt.columns
              if "s2" in c or "s3" in c or "match" in c), None),
    )
    if s1_col is None or s23_col is None:
        # Last-resort: first two columns
        s1_col, s23_col = gt.columns[0], gt.columns[1]
        log.warning(
            "GT column detection ambiguous — assuming s1='%s'  s23='%s'",
            s1_col, s23_col,
        )

    log.info("GT columns: s1='%s'  s23='%s'  rows=%d", s1_col, s23_col, len(gt))

    # --- Build GT lookup: s23_id → {correct s1_ids} ---
    gt_lookup: dict[str, set[str]] = defaultdict(set)
    for row in gt.itertuples(index=False):
        s23_id = getattr(row, s23_col, None)
        s1_id  = getattr(row, s1_col,  None)
        if s23_id and s1_id:
            gt_lookup[s23_id].add(s1_id)

    matchable_s1_ids: set[str] = set(gt[s1_col].dropna().unique())
    gt_pairs_total              = sum(len(v) for v in gt_lookup.values())

    log.info(
        "Matchable S1 entities (denominator): %d  |  GT pairs: %d",
        len(matchable_s1_ids), gt_pairs_total,
    )

    # --- Stream candidate pairs; check against GT lookup ---
    # Only track s23_ids that appear in GT to bound memory.
    gt_s23_ids = set(gt_lookup.keys())

    recovered_s1:      set[str] = set()
    gt_pairs_recovered: int     = 0

    log.info(
        "Streaming candidate pairs for recall evaluation: %s",
        candidate_pairs_path,
    )
    t0 = time.perf_counter()
    rows_scanned = 0

    chunksize = 2_000_000
    for chunk in pd.read_csv(
        candidate_pairs_path,
        sep="\t",
        dtype=str,
        chunksize=chunksize,
    ):
        # Filter to query IDs that actually appear in GT
        chunk = chunk[chunk["query_entity_id"].isin(gt_s23_ids)]

        for row in chunk.itertuples(index=False):
            s23_id = row.query_entity_id
            s1_id  = row.candidate_s1_entity_id
            if s23_id in gt_lookup and s1_id in gt_lookup[s23_id]:
                gt_pairs_recovered += 1
                recovered_s1.add(s1_id)

        rows_scanned += len(chunk)

    pair_recall   = gt_pairs_recovered / gt_pairs_total   if gt_pairs_total   else 0.0
    entity_recall = len(recovered_s1)  / len(matchable_s1_ids) if matchable_s1_ids else 0.0

    log.info("=" * 65)
    log.info("BLOCKING RECALL REPORT  (%.1f min to evaluate)",
             (time.perf_counter() - t0) / 60)
    log.info("  Matchable S1 entities (denominator) : %10d",
             len(matchable_s1_ids))
    log.info("  GT pairs total                       : %10d", gt_pairs_total)
    log.info("  GT pairs recovered in top-%d         : %10d  (pair recall = %.4f)",
             20, gt_pairs_recovered, pair_recall)
    log.info("  S1 entities recovered                : %10d", len(recovered_s1))
    log.info("  ENTITY-LEVEL RECALL                  :     %.4f  (target ≥ 0.95)",
             entity_recall)

    target_met = entity_recall >= 0.95
    if target_met:
        log.info("  ✓ Recall target met — proceed to feature engineering.")
    else:
        log.warning(
            "  ⚠ Recall %.4f < 0.95 — add blocking round 2 (sorted name tokens, "
            "numeric address tokens) per ER_Strategy.md §3.",
            entity_recall,
        )
    log.info("=" * 65)

    return {
        "matchable_s1_count":    len(matchable_s1_ids),
        "gt_pairs_total":        gt_pairs_total,
        "gt_pairs_recovered":    gt_pairs_recovered,
        "pair_recall":           round(pair_recall,   4),
        "s1_entities_recovered": len(recovered_s1),
        "entity_recall":         round(entity_recall, 4),
        "recall_target_met":     target_met,
    }


# ===========================================================================
# 6. CLI + MAIN
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ER Pipeline Phase 4 — FAISS build + blocking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--job-mode",
        choices=["train", "test"],
        default="train",
        help="Which split to process. Run two separate jobs: train then test.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K S1 candidates per S2/S3 query. K=20 locked — Q6/Q32.",
    )
    p.add_argument(
        "--nlist",
        type=int,
        default=4096,
        help="FAISS IVF number of Voronoi cells. Strategy §2F: 4096.",
    )
    p.add_argument(
        "--nprobe",
        type=int,
        default=64,
        help="FAISS IVF cells probed per search. Strategy §2F: 64 → recall@20 ≈ 95%+.",
    )
    p.add_argument(
        "--query-batch",
        type=int,
        default=10_000,
        help="Number of query vectors per FAISS search call.",
    )
    p.add_argument(
        "--eval-recall",
        action="store_true",
        default=False,
        help=(
            "After blocking, compute entity-level recall vs ground truth. "
            "Only meaningful for --job-mode train. Requires pandas."
        ),
    )
    p.add_argument(
        "--gt-file",
        default="train_ground_truth.tsv",
        help=(
            "Ground truth filename, relative to --input-dir. "
            "Only used when --eval-recall is set."
        ),
    )
    p.add_argument(
        "--input-dir",
        default=_SM_INPUT_DIR,
        help="Directory containing .npy shards and ground truth TSV.",
    )
    p.add_argument(
        "--output-dir",
        default=_SM_OUTPUT_DIR,
        help="Directory to write candidate_pairs_{split}.tsv and manifest.",
    )
    return p.parse_args()


def main() -> None:
    args       = parse_args()
    split      = args.job_mode
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("ER Pipeline — Phase 4: FAISS Index Build + Blocking")
    log.info(
        "  job_mode=%s  top_k=%d  nlist=%d  nprobe=%d  query_batch=%d",
        split, args.top_k, args.nlist, args.nprobe, args.query_batch,
    )
    log.info("  input_dir  = %s", input_dir)
    log.info("  output_dir = %s", output_dir)
    log.info("=" * 70)

    t_job = time.perf_counter()

    # -----------------------------------------------------------------------
    # Phase A — Load S1 embeddings
    # -----------------------------------------------------------------------
    log.info("[Phase A] Loading S1 embeddings (split='%s') …", split)
    s1_vecs, s1_metas = load_shards("s1", split, input_dir)

    if len(s1_vecs) == 0:
        log.error("No S1 vectors loaded — cannot build FAISS index. Aborting.")
        sys.exit(1)

    log.info(
        "  S1 total: %d records  |  %.0f MB (fp16)",
        len(s1_vecs), s1_vecs.nbytes / 1e6,
    )

    # -----------------------------------------------------------------------
    # Phase B — Build per-country FAISS indexes over S1
    # -----------------------------------------------------------------------
    log.info("[Phase B] Building per-country FAISS indexes …")
    t_b = time.perf_counter()

    country_indexes = CountryIndexes()
    country_indexes.build(
        vecs_fp16 = s1_vecs,
        metas     = s1_metas,
        nlist     = args.nlist,
        nprobe    = args.nprobe,
    )

    # Release the fp16 array — FAISS holds its own fp32 copy internally
    del s1_vecs, s1_metas

    log.info(
        "[Phase B] Done in %.1f min  |  %d S1 records indexed across %d country shards.",
        (time.perf_counter() - t_b) / 60,
        country_indexes.total_s1_records,
        len(country_indexes.country_to_index),
    )

    # -----------------------------------------------------------------------
    # Phase C — Stream S2 + S3 through FAISS → candidate pairs
    # -----------------------------------------------------------------------
    output_path = output_dir / f"candidate_pairs_{split}.tsv"
    log.info("[Phase C] Streaming S2/S3 → candidate pairs: %s", output_path)

    writer = CandidatePairWriter(output_path)

    total_s2 = _blocking_pass(
        source_tag      = "s2",
        split           = split,
        input_dir       = input_dir,
        country_indexes = country_indexes,
        writer          = writer,
        top_k           = args.top_k,
        query_batch     = args.query_batch,
    )
    total_s3 = _blocking_pass(
        source_tag      = "s3",
        split           = split,
        input_dir       = input_dir,
        country_indexes = country_indexes,
        writer          = writer,
        top_k           = args.top_k,
        query_batch     = args.query_batch,
    )

    writer.close()

    output_mb = output_path.stat().st_size / 1e6
    log.info(
        "[Phase C] Done: %d S2 + %d S3 = %d query records → %d candidate pairs  "
        "(%.0f MB, %.1f pairs/query)",
        total_s2, total_s3, total_s2 + total_s3,
        writer.total_pairs, output_mb,
        writer.total_pairs / (total_s2 + total_s3) if (total_s2 + total_s3) > 0 else 0,
    )

    # -----------------------------------------------------------------------
    # Phase D — Blocking recall evaluation (train + --eval-recall only)
    # -----------------------------------------------------------------------
    recall_results: dict = {}

    if args.eval_recall:
        if split != "train":
            log.warning(
                "--eval-recall only applies to the train split (no ground truth "
                "for test). Skipping."
            )
        else:
            gt_path = input_dir / args.gt_file
            if not gt_path.exists():
                log.error(
                    "Ground truth not found at %s — skipping recall eval.", gt_path
                )
            else:
                recall_results = evaluate_blocking_recall(output_path, gt_path)

    # -----------------------------------------------------------------------
    # Phase E — Write manifest
    # -----------------------------------------------------------------------
    elapsed_hr = (time.perf_counter() - t_job) / 3600

    manifest = {
        "phase":           "4-faiss-blocking",
        "job_mode":        split,
        "top_k":           args.top_k,
        "nlist":           args.nlist,
        "nprobe":          args.nprobe,
        "query_batch":     args.query_batch,
        "s1_total":        country_indexes.total_s1_records,
        "s2_queries":      total_s2,
        "s3_queries":      total_s3,
        "total_queries":   total_s2 + total_s3,
        "total_candidate_pairs": writer.total_pairs,
        "output_file":     output_path.name,
        "output_mb":       round(output_mb, 1),
        "elapsed_hrs":     round(elapsed_hr, 3),
        "country_index_sizes": {
            c: country_indexes.country_to_index[c].ntotal
            for c in sorted(country_indexes.country_to_index)
        },
        **recall_results,
    }

    manifest_path = output_dir / f"manifest_faiss_{split}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written: %s", manifest_path)

    log.info(
        "=== Phase 4 complete: %.2f hrs total  |  %d candidate pairs ===",
        elapsed_hr, writer.total_pairs,
    )


if __name__ == "__main__":
    main()
