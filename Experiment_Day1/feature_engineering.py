"""
feature_engineering.py
=======================
Phase 5 — Feature Engineering (runs on Notebook CPU, free tier)

Strategy reference: ER_Strategy.md §5 (domain features), §6 (normalisation
flow), §7 (30-feature set per candidate pair)

What this module does
---------------------
1. Loads candidate_pairs_{split}_with_tfidf.tsv (union of FAISS + TF-IDF
   candidates from Phases 3+4).
2. Loads and normalises all source TSVs once, caching NormalisedRecord per
   entity_id so normaliser.normalise_record() is called exactly once per
   entity.
3. Computes all 30 features per candidate pair as described in §7.
4. For the train split: merges with train_ground_truth.tsv to add ``label``
   (1=match, 0=non-match).
5. Saves features_train.parquet and features_test.parquet.

Usage (called from main.ipynb)
-------------------------------
    import feature_engineering

    train_df = feature_engineering.run(
        split="train",
        data_dir="../student_resource/dataset",
        candidates_path="output/candidate_pairs_train_with_tfidf.tsv",
        output_path="output/features_train.parquet",
        ground_truth_path="../student_resource/dataset/train/train_ground_truth.tsv",
    )

    test_df = feature_engineering.run(
        split="test",
        data_dir="../student_resource/dataset",
        candidates_path="output/candidate_pairs_test_with_tfidf.tsv",
        output_path="output/features_test.parquet",
    )
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from normaliser import NormalisedRecord, canonical_country, normalise_record

# ---------------------------------------------------------------------------
# Optional: rapidfuzz
# ---------------------------------------------------------------------------
try:
    from rapidfuzz.distance import Levenshtein as _Lev
    from rapidfuzz.fuzz import token_sort_ratio as _tsr
    from rapidfuzz.fuzz import WRatio as _wratio
    import rapidfuzz.distance as _rfd

    def _jaro_winkler(a: str, b: str) -> float:
        return _rfd.JaroWinkler.normalized_similarity(a, b)

    def _levenshtein_ratio(a: str, b: str) -> float:
        return _Lev.normalized_similarity(a, b)

    def _tok_sort_ratio(a: str, b: str) -> float:
        return _tsr(a, b) / 100.0

    _RAPIDFUZZ = True
except ImportError:
    _RAPIDFUZZ = False

    def _jaro_winkler(a: str, b: str) -> float:  # type: ignore[misc]
        return 0.0

    def _levenshtein_ratio(a: str, b: str) -> float:  # type: ignore[misc]
        return 0.0

    def _tok_sort_ratio(a: str, b: str) -> float:  # type: ignore[misc]
        return 0.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CHUNK_SIZE = 100_000  # pairs computed per batch


# ===========================================================================
# 1. String similarity helpers
# ===========================================================================


def _jaccard(set_a: frozenset, set_b: frozenset) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _char_trigram_jaccard(a: str, b: str) -> float:
    def trigrams(s: str) -> set:
        return {s[i:i+3] for i in range(max(len(s) - 2, 0))}
    tg_a = trigrams(a)
    tg_b = trigrams(b)
    if not tg_a and not tg_b:
        return 1.0
    union = tg_a | tg_b
    return len(tg_a & tg_b) / len(union) if union else 0.0


def _numeric_tokens(text: str) -> frozenset:
    """Extract purely numeric tokens from text."""
    return frozenset(t for t in text.split() if t.isdigit())


def _dominant_script(text: str) -> str:
    counts: dict[str, int] = {}
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            try:
                block = unicodedata.name(ch, "").split(" ")[0]
            except Exception:
                block = "UNKNOWN"
            counts[block] = counts.get(block, 0) + 1
    if not counts:
        return "unknown"
    return max(counts, key=counts.__getitem__).lower()


def _script_mismatch(rec_a: NormalisedRecord, rec_b: NormalisedRecord) -> int:
    return int(rec_a.dominant_script != rec_b.dominant_script)


def _numeric_overlap(num_a: frozenset, num_b: frozenset) -> float:
    if not num_a and not num_b:
        return 1.0
    union = num_a | num_b
    return len(num_a & num_b) / len(union) if union else 0.0


def _addr_landmark_penalised_jaccard(rec_a: NormalisedRecord, rec_b: NormalisedRecord) -> float:
    base = _jaccard(rec_a.addr_tokens, rec_b.addr_tokens)
    landmark_ratio = (
        int(rec_a.has_landmark) + int(rec_b.has_landmark)
    ) / 2.0
    return base * (1.0 - 0.4 * landmark_ratio)


def _name_dedup_ratio(rec: NormalisedRecord) -> float:
    tokens = rec.name_normalized.split()
    if not tokens:
        return 1.0
    return len(set(tokens)) / len(tokens)


def _addr_city_token_match(rec_a: NormalisedRecord, rec_b: NormalisedRecord) -> int:
    """1 if largest non-numeric addr token matches between both records."""
    def largest_non_numeric(tokens: frozenset) -> str:
        non_num = [t for t in tokens if not t.isdigit()]
        if not non_num:
            return ""
        return max(non_num, key=len)
    city_a = largest_non_numeric(rec_a.addr_tokens)
    city_b = largest_non_numeric(rec_b.addr_tokens)
    if not city_a or not city_b:
        return 0
    return int(city_a == city_b)


# ===========================================================================
# 2. TF-IDF cosine per-country (name + address, separate)
# ===========================================================================


class _TFIDFStore:
    """Lazy per-country TF-IDF matrices for name and address cosine features."""

    def __init__(self) -> None:
        self._name_vecs: dict[str, TfidfVectorizer] = {}
        self._name_mats: dict[str, object] = {}
        self._name_ids: dict[str, np.ndarray] = {}
        self._addr_vecs: dict[str, TfidfVectorizer] = {}
        self._addr_mats: dict[str, object] = {}
        self._addr_ids: dict[str, np.ndarray] = {}

    def fit(self, s1_records: Dict[str, NormalisedRecord]) -> None:
        """Fit TF-IDF models per country over all S1 records."""
        by_country: dict[str, list] = {}
        for entity_id, rec in s1_records.items():
            c = rec.country_canonical
            by_country.setdefault(c, []).append((entity_id, rec))

        for country, items in by_country.items():
            ids = np.array([x[0] for x in items], dtype=str)
            name_texts = [x[1].name_normalized for x in items]
            addr_texts = [x[1].addr_normalized for x in items]

            nv = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, dtype=np.float32)
            nm = nv.fit_transform(name_texts)
            av = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, dtype=np.float32)
            am = av.fit_transform(addr_texts)

            self._name_vecs[country] = nv
            self._name_mats[country] = nm
            self._name_ids[country] = ids
            self._addr_vecs[country] = av
            self._addr_mats[country] = am
            self._addr_ids[country] = ids

    def name_cosine(self, query_rec: NormalisedRecord, s1_rec: NormalisedRecord) -> float:
        c = s1_rec.country_canonical
        if c not in self._name_vecs:
            return 0.0
        q_vec = self._name_vecs[c].transform([query_rec.name_normalized])
        ids = self._name_ids[c]
        idx_arr = np.where(ids == s1_rec.entity_id)[0]
        if len(idx_arr) == 0:
            return 0.0
        s1_vec = self._name_mats[c][idx_arr[0]]
        score = float((q_vec @ s1_vec.T).toarray()[0][0])
        return score

    def addr_cosine(self, query_rec: NormalisedRecord, s1_rec: NormalisedRecord) -> float:
        c = s1_rec.country_canonical
        if c not in self._addr_vecs:
            return 0.0
        q_vec = self._addr_vecs[c].transform([query_rec.addr_normalized])
        ids = self._addr_ids[c]
        idx_arr = np.where(ids == s1_rec.entity_id)[0]
        if len(idx_arr) == 0:
            return 0.0
        s1_vec = self._addr_mats[c][idx_arr[0]]
        score = float((q_vec @ s1_vec.T).toarray()[0][0])
        return score


# ===========================================================================
# 3. 30-feature computation per pair
# ===========================================================================


def compute_pair_features(
    q_rec: NormalisedRecord,
    s1_rec: NormalisedRecord,
    faiss_rank: Optional[float],
    faiss_score: Optional[float],
    tfidf_rank: Optional[float],
    tfidf_store: Optional[_TFIDFStore] = None,
    bgem3_name_cosine: Optional[float] = None,
    bgem3_full_cosine: Optional[float] = None,
) -> dict:
    """Compute all 30 features for a single (query, S1) candidate pair.

    BGE-m3 cosines (bgem3_name_cosine, bgem3_full_cosine) come directly from
    the candidate pairs TSV (produced by build_faiss_and_block.py, which stores
    the FAISS inner-product score as faiss_score). They are passed in directly
    rather than recomputed here.

    When bgem3_full_cosine is None (e.g. TF-IDF-only pairs), faiss_score is
    used as a fallback; when both are absent, the value is 0.0.
    """
    # ---- Name features --------------------------------------------------
    q_name = q_rec.name_normalized
    s1_name = s1_rec.name_normalized
    q_core = q_rec.name_core
    s1_core = s1_rec.name_core

    name_jaro = _jaro_winkler(q_name, s1_name)
    name_lev = _levenshtein_ratio(q_name, s1_name)
    name_tok_jaccard = _jaccard(
        frozenset(q_name.split()), frozenset(s1_name.split())
    )
    name_tok_sort = _tok_sort_ratio(q_name, s1_name)
    name_char_tg = _char_trigram_jaccard(q_name, s1_name)
    core_jaro = _jaro_winkler(q_core, s1_core)
    core_tok_jaccard = _jaccard(
        frozenset(q_core.split()), frozenset(s1_core.split())
    )
    name_tfidf_cos = (
        tfidf_store.name_cosine(q_rec, s1_rec) if tfidf_store else 0.0
    )
    both_domain = int(q_rec.is_domain and s1_rec.is_domain)
    dom_base_jaro = (
        _jaro_winkler(q_rec.base_domain, s1_rec.base_domain)
        if both_domain else 0.0
    )
    tld_match = int(q_rec.tld == s1_rec.tld and q_rec.tld != "")
    max_name_jaro = max(
        _jaro_winkler(q_rec.name_legal, s1_rec.name_legal),
        _jaro_winkler(q_rec.name_trade, s1_rec.name_trade)
        if q_rec.name_trade and s1_rec.name_trade else 0.0,
    )
    name_dedup_q = _name_dedup_ratio(q_rec)
    name_dedup_s1 = _name_dedup_ratio(s1_rec)
    name_dedup = (name_dedup_q + name_dedup_s1) / 2.0

    # ---- Address features -----------------------------------------------
    q_addr = q_rec.addr_normalized
    s1_addr = s1_rec.addr_normalized
    addr_jaro = _jaro_winkler(q_addr, s1_addr)
    addr_tok_jaccard = _jaccard(q_rec.addr_tokens, s1_rec.addr_tokens)
    addr_tok_sort = _tok_sort_ratio(q_addr, s1_addr)
    addr_num_a = _numeric_tokens(q_addr)
    addr_num_b = _numeric_tokens(s1_addr)
    addr_numeric_overlap = _numeric_overlap(addr_num_a, addr_num_b)
    addr_tfidf_cos = (
        tfidf_store.addr_cosine(q_rec, s1_rec) if tfidf_store else 0.0
    )
    addr_lm_jaccard = _addr_landmark_penalised_jaccard(q_rec, s1_rec)
    addr_null_either = int(q_rec.addr_is_null or s1_rec.addr_is_null)
    addr_null_both = int(q_rec.addr_is_null and s1_rec.addr_is_null)
    addr_num_exact = int(
        len(addr_num_a) > 0 and addr_num_a == addr_num_b
    )
    addr_city_match = _addr_city_token_match(q_rec, s1_rec)

    # ---- Global features ------------------------------------------------
    bgem3_full = bgem3_full_cosine if bgem3_full_cosine is not None else (
        faiss_score if faiss_score is not None else 0.0
    )
    bgem3_name = bgem3_name_cosine if bgem3_name_cosine is not None else 0.0
    f_rank = faiss_rank if faiss_rank is not None else 99.0
    t_rank = tfidf_rank if tfidf_rank is not None else 99.0
    script_mm = _script_mismatch(q_rec, s1_rec)
    country_match = int(q_rec.country_canonical == s1_rec.country_canonical)
    # source_pair: 0 = S2 query, 1 = S3 query
    source_pair = 0 if q_rec.source == "S2" else 1
    tld_country_sig_match = int(
        q_rec.tld_country_signal != "unknown"
        and q_rec.tld_country_signal == s1_rec.tld_country_signal
    )

    return {
        # Name (15)
        "bgem3_name_cosine": bgem3_name,
        "bgem3_full_cosine": bgem3_full,
        "name_jaro_winkler": name_jaro,
        "name_levenshtein_ratio": name_lev,
        "name_token_jaccard": name_tok_jaccard,
        "name_token_sort_ratio": name_tok_sort,
        "name_char_trigram_jaccard": name_char_tg,
        "core_name_jaro_winkler": core_jaro,
        "core_name_token_jaccard": core_tok_jaccard,
        "name_tfidf_cosine": name_tfidf_cos,
        "name_is_domain_both": both_domain,
        "domain_base_jaro": dom_base_jaro,
        "tld_match": tld_match,
        "max_name_jaro": max_name_jaro,
        "name_dedup_ratio": name_dedup,
        # Address (10)
        "addr_jaro_winkler": addr_jaro,
        "addr_token_jaccard": addr_tok_jaccard,
        "addr_token_sort_ratio": addr_tok_sort,
        "addr_numeric_overlap": addr_numeric_overlap,
        "addr_tfidf_cosine": addr_tfidf_cos,
        "addr_landmark_penalised_jaccard": addr_lm_jaccard,
        "addr_is_null_either": addr_null_either,
        "addr_is_null_both": addr_null_both,
        "addr_numeric_exact_match": addr_num_exact,
        "addr_city_token_match": addr_city_match,
        # Global (7) — total = 32 including the two bge cosines above
        "faiss_rank": f_rank,
        "tfidf_rank": t_rank,
        "script_mismatch": script_mm,
        "country_canonical_match": country_match,
        "source_pair": source_pair,
        "tld_country_signal_match": tld_country_sig_match,
    }


# ===========================================================================
# 4. Record cache builder
# ===========================================================================


def _build_record_cache(
    source_files: List[Tuple[str, str]],
) -> Dict[str, NormalisedRecord]:
    """Normalise all records in the provided source TSVs and cache by entity_id.

    Parameters
    ----------
    source_files:
        List of (file_path, source_tag) tuples, e.g.
        [("train_source1.tsv", "S1"), ("train_source2.tsv", "S2"), ...]
    """
    cache: Dict[str, NormalisedRecord] = {}
    for path, src_tag in source_files:
        log.info("Loading and normalising %s (%s)...", Path(path).name, src_tag)
        df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
        df.columns = [c.strip().lower() for c in df.columns]
        # Detect actual column names
        id_col = df.columns[0]
        name_col = next((c for c in df.columns if "name" in c), None)
        addr_col = next((c for c in df.columns if "address" in c or "addr" in c), None)
        country_col = next((c for c in df.columns if "country" in c), None)

        for _, row in df.iterrows():
            eid = str(row[id_col])
            if eid in cache:
                continue
            rec = normalise_record(
                entity_id=eid,
                business_name=str(row[name_col]) if name_col and pd.notna(row.get(name_col)) else "",
                business_address=str(row[addr_col]) if addr_col and pd.notna(row.get(addr_col)) else "",
                country=str(row[country_col]) if country_col and pd.notna(row.get(country_col)) else "",
            )
            # Override source tag from file
            object.__setattr__(rec, "source", src_tag)
            cache[eid] = rec
        log.info("  cached %d records (total so far: %d)", len(df), len(cache))
    return cache


# ===========================================================================
# 5. Main run function
# ===========================================================================


def run(
    split: str,
    data_dir: str,
    candidates_path: str,
    output_path: str,
    ground_truth_path: Optional[str] = None,
    fit_tfidf: bool = True,
) -> pd.DataFrame:
    """Compute 30 features for all candidate pairs and save as parquet.

    Parameters
    ----------
    split:
        "train" or "test".
    data_dir:
        Root dataset directory.
    candidates_path:
        Merged candidate pairs TSV (from tfidf_blocking.run).
    output_path:
        Output parquet path.
    ground_truth_path:
        Path to train_ground_truth.tsv (train split only). When provided,
        a ``label`` column is added (1=match, 0=non-match).
    fit_tfidf:
        Whether to fit per-country TF-IDF for name/addr cosine features.
        Can be set to False to skip (saves memory; name_tfidf_cosine and
        addr_tfidf_cosine will be 0.0).
    """
    split_dir = Path(data_dir) / split
    os.makedirs(Path(output_path).parent, exist_ok=True)

    # ------------------------------------------------------------------
    # 5A. Load candidate pairs
    # ------------------------------------------------------------------
    log.info("Loading candidate pairs from %s ...", candidates_path)
    cands = pd.read_csv(candidates_path, sep="\t", dtype=str, low_memory=False)
    cands.columns = [c.strip().lower() for c in cands.columns]
    log.info("Loaded %d candidate pairs.", len(cands))

    # Ensure numeric columns are numeric
    for col in ("faiss_rank", "faiss_score", "tfidf_rank", "tfidf_score"):
        if col in cands.columns:
            cands[col] = pd.to_numeric(cands[col], errors="coerce")

    # ------------------------------------------------------------------
    # 5B. Build record cache from source TSVs
    # ------------------------------------------------------------------
    source_files = [
        (str(split_dir / f"{split}_source1.tsv"), "S1"),
        (str(split_dir / f"{split}_source2.tsv"), "S2"),
        (str(split_dir / f"{split}_source3.tsv"), "S3"),
    ]
    record_cache = _build_record_cache(source_files)

    # ------------------------------------------------------------------
    # 5C. Fit TF-IDF store on S1 records only
    # ------------------------------------------------------------------
    tfidf_store = None
    if fit_tfidf:
        log.info("Fitting TF-IDF store on S1 records...")
        s1_records = {
            eid: rec for eid, rec in record_cache.items() if rec.source == "S1"
        }
        tfidf_store = _TFIDFStore()
        tfidf_store.fit(s1_records)
        log.info("TF-IDF store fitted on %d S1 records.", len(s1_records))

    # ------------------------------------------------------------------
    # 5D. Compute features per pair (chunked)
    # ------------------------------------------------------------------
    log.info("Computing 30 features per pair (total: %d)...", len(cands))
    feature_rows: List[dict] = []
    n_chunks = (len(cands) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for chunk_i in range(n_chunks):
        chunk = cands.iloc[chunk_i * CHUNK_SIZE:(chunk_i + 1) * CHUNK_SIZE]
        for _, row in chunk.iterrows():
            q_id = str(row.get("query_entity_id", ""))
            s1_id = str(row.get("candidate_s1_entity_id", ""))

            q_rec = record_cache.get(q_id)
            s1_rec = record_cache.get(s1_id)

            if q_rec is None or s1_rec is None:
                continue  # entity missing from source files (shouldn't happen)

            feats = compute_pair_features(
                q_rec=q_rec,
                s1_rec=s1_rec,
                faiss_rank=row.get("faiss_rank"),
                faiss_score=row.get("faiss_score"),
                tfidf_rank=row.get("tfidf_rank"),
                tfidf_store=tfidf_store,
            )
            feats["query_entity_id"] = q_id
            feats["candidate_s1_entity_id"] = s1_id
            feature_rows.append(feats)

        if (chunk_i + 1) % 5 == 0 or (chunk_i + 1) == n_chunks:
            log.info("  chunk %d/%d done (%d pairs so far)", chunk_i + 1, n_chunks, len(feature_rows))

    features_df = pd.DataFrame(feature_rows)
    log.info("Feature matrix shape: %s", features_df.shape)

    # ------------------------------------------------------------------
    # 5E. Label pairs for training split
    # ------------------------------------------------------------------
    if ground_truth_path and Path(ground_truth_path).exists():
        log.info("Loading ground truth from %s ...", ground_truth_path)
        gt = pd.read_csv(ground_truth_path, sep="\t", dtype=str, low_memory=False)
        gt.columns = [c.strip().lower() for c in gt.columns]
        # Ground truth columns: source1_entity_id, matched_entity_ids (comma-sep)
        gt_pairs: Set[Tuple[str, str]] = set()
        s1_col = gt.columns[0]
        match_col = gt.columns[1]
        for _, row in gt.iterrows():
            s1_id = str(row[s1_col])
            for mid in str(row[match_col]).split(","):
                mid = mid.strip()
                if mid:
                    gt_pairs.add((mid, s1_id))  # (query_id, s1_id)

        features_df["label"] = features_df.apply(
            lambda r: int(
                (str(r["query_entity_id"]), str(r["candidate_s1_entity_id"])) in gt_pairs
            ),
            axis=1,
        )
        pos = features_df["label"].sum()
        log.info("Labels added: %d positives, %d negatives", pos, len(features_df) - pos)

    # ------------------------------------------------------------------
    # 5F. Save
    # ------------------------------------------------------------------
    features_df.to_parquet(output_path, index=False)
    log.info("Saved features to %s", output_path)
    return features_df


# ===========================================================================
# 6. CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature engineering — Phase 5")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--data-dir", default="../student_resource/dataset")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ground-truth", default=None)
    parser.add_argument("--no-tfidf", action="store_true")
    args = parser.parse_args()

    run(
        split=args.split,
        data_dir=args.data_dir,
        candidates_path=args.candidates,
        output_path=args.output,
        ground_truth_path=args.ground_truth,
        fit_tfidf=not args.no_tfidf,
    )
