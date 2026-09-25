"""
tfidf_blocking.py
=================
Phase 4 supplement — TF-IDF Blocking (runs on Notebook CPU, free tier)

Strategy reference: ER_Strategy.md §3 (Supplementary Blocking Keys)

What this module does
---------------------
1. Builds a per-country TF-IDF matrix over S1 name tokens.
   - Applies indic-transliteration before TF-IDF so that Indian-script S1
     records are also reachable via TF-IDF (though BGE-m3 remains primary).
2. For each S2/S3 record, queries the same-country TF-IDF index for top-20 S1
   candidates.
3. Merges results with the existing FAISS candidate pairs (candidate_pairs.tsv
   produced by build_faiss_and_block.py) via a union.
4. Adds/fills a ``tfidf_rank`` column (NaN for pairs only in FAISS list).
5. Saves the merged candidate pairs as candidate_pairs_{split}_with_tfidf.tsv.

Usage (called from main.ipynb)
-------------------------------
    import tfidf_blocking

    # Build TF-IDF indices and produce merged candidate pairs
    tfidf_blocking.run(
        split="train",                           # "train" or "test"
        data_dir="../student_resource/dataset",  # parent of train/ and test/ folders
        faiss_candidates_path="output/candidate_pairs_train.tsv",
        output_path="output/candidate_pairs_train_with_tfidf.tsv",
        top_k=20,                                # matches FAISS K
    )

Entry points
------------
    run(split, data_dir, faiss_candidates_path, output_path, top_k=20)
        Full pipeline: load, build, query, merge, save.

    build_tfidf_index(s1_df, country)
        Returns (vectorizer, matrix, index_entity_ids) for one country.

    query_tfidf(text, vectorizer, matrix, entity_ids, top_k)
        Returns list of (entity_id, score) tuples for one query.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Normaliser lives flat in the same directory
from normaliser import canonical_country, normalise_record

# ---------------------------------------------------------------------------
# Optional: indic-transliteration (Q29)
# ---------------------------------------------------------------------------
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _itrans

    _INDIC_AVAILABLE = True
except ImportError:
    _INDIC_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_K = 20
CHUNK_SIZE = 50_000  # rows processed per batch when querying TF-IDF

_INDIC_SCRIPTS = [
    sanscript.DEVANAGARI,
    sanscript.TAMIL,
    sanscript.TELUGU,
    sanscript.KANNADA,
    sanscript.MALAYALAM,
    sanscript.BENGALI,
    sanscript.GURMUKHI,
    sanscript.GUJARATI,
    sanscript.ORIYA,
] if _INDIC_AVAILABLE else []


# ===========================================================================
# 1. Script helpers
# ===========================================================================


def _dominant_script(text: str) -> str:
    """Return the dominant Unicode block of ``text`` (rough heuristic)."""
    counts: dict[str, int] = {}
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            block = unicodedata.name(ch, "").split(" ")[0]
            counts[block] = counts.get(block, 0) + 1
    if not counts:
        return "unknown"
    dominant = max(counts, key=counts.__getitem__)
    return dominant.lower()


def _transliterate_indic(text: str) -> str:
    """Transliterate common Indic scripts → ITRANS (Latin approximation).

    Falls back to the original text if indic-transliteration is unavailable
    or the script is not recognised.
    """
    if not _INDIC_AVAILABLE:
        return text
    for script in _INDIC_SCRIPTS:
        try:
            result = _itrans(text, script, sanscript.ITRANS)
            if result and result != text:
                return result
        except Exception:
            continue
    return text


def _tfidf_text(name_normalized: str, name_core: str) -> str:
    """Build the text fed into TF-IDF from normalised name fields.

    Concatenates name_normalized + name_core so that both the full name and
    the suffix-stripped core get representation.  Indic-transliteration is
    applied so that non-Latin tokens become searchable.
    """
    combined = f"{name_normalized} {name_core}".strip()
    return _transliterate_indic(combined)


# ===========================================================================
# 2. TF-IDF index construction
# ===========================================================================


def build_tfidf_index(
    s1_df: pd.DataFrame,
    country: str,
) -> Tuple[TfidfVectorizer, csr_matrix, np.ndarray]:
    """Build a TF-IDF index for all S1 records in ``country``.

    Parameters
    ----------
    s1_df:
        DataFrame with at least columns: entity_id, business_name,
        business_address, country.
    country:
        Canonical country string ("US", "India", "France", or similar).

    Returns
    -------
    vectorizer : TfidfVectorizer (fitted)
    matrix     : csr_matrix  [n_s1_records, n_vocab]
    ids        : np.ndarray  [n_s1_records]  — entity_id aligned to matrix rows
    """
    subset = s1_df[s1_df["country_canonical"] == country].copy()
    if subset.empty:
        log.warning("No S1 records for country=%s — skipping TF-IDF index.", country)
        return None, None, np.array([], dtype=str)

    log.info("Building TF-IDF index for %s (%d S1 records)...", country, len(subset))

    texts: List[str] = []
    ids: List[str] = []
    for _, row in subset.iterrows():
        rec = normalise_record(
            entity_id=str(row["entity_id"]),
            business_name=str(row.get("business_name", "")),
            business_address=str(row.get("business_address", "")),
            country=str(row.get("country", "")),
        )
        texts.append(_tfidf_text(rec.name_normalized, rec.name_core))
        ids.append(str(row["entity_id"]))

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=200_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(texts)
    log.info(
        "TF-IDF index built for %s: vocab=%d, shape=%s",
        country, len(vectorizer.vocabulary_), matrix.shape,
    )
    return vectorizer, matrix, np.array(ids, dtype=str)


# ===========================================================================
# 3. Per-record TF-IDF query
# ===========================================================================


def query_tfidf(
    text: str,
    vectorizer: TfidfVectorizer,
    matrix: csr_matrix,
    entity_ids: np.ndarray,
    top_k: int = TOP_K,
) -> List[Tuple[str, float]]:
    """Query TF-IDF index for one record.

    Returns
    -------
    List of (s1_entity_id, tfidf_score) sorted descending by score.
    Empty list if the vectorizer/matrix is None (no S1 records for country).
    """
    if vectorizer is None or matrix is None or len(entity_ids) == 0:
        return []
    vec = vectorizer.transform([text])  # [1, vocab]
    scores = (vec @ matrix.T).toarray()[0]  # [n_s1]
    top_idx = np.argpartition(scores, max(-len(scores), -top_k))[-top_k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return [(entity_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0.0]


# ===========================================================================
# 4. Batch query helper
# ===========================================================================


def _query_batch(
    rows: pd.DataFrame,
    indices: Dict[str, Tuple[TfidfVectorizer, csr_matrix, np.ndarray]],
    top_k: int,
) -> List[dict]:
    """Return tfidf candidate rows for a batch of S2/S3 records."""
    results = []
    for _, row in rows.iterrows():
        rec = normalise_record(
            entity_id=str(row["entity_id"]),
            business_name=str(row.get("business_name", "")),
            business_address=str(row.get("business_address", "")),
            country=str(row.get("country", "")),
        )
        country = rec.country_canonical
        if country not in indices:
            country = "Unknown"
        if country not in indices:
            continue
        vectorizer, matrix, entity_ids = indices[country]
        text = _tfidf_text(rec.name_normalized, rec.name_core)
        hits = query_tfidf(text, vectorizer, matrix, entity_ids, top_k)
        for rank, (s1_id, score) in enumerate(hits, start=1):
            results.append(
                {
                    "query_entity_id": str(row["entity_id"]),
                    "candidate_s1_entity_id": s1_id,
                    "tfidf_rank": rank,
                    "tfidf_score": score,
                }
            )
    return results


# ===========================================================================
# 5. Load helpers
# ===========================================================================


def _load_source(path: str, source_tag: str) -> pd.DataFrame:
    """Load a source TSV and add a source column."""
    log.info("Loading %s from %s ...", source_tag, path)
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    # Rename first column to entity_id regardless of original header
    first_col = df.columns[0]
    if first_col != "entity_id":
        df = df.rename(columns={first_col: "entity_id"})
    df["source_tag"] = source_tag
    return df


def _add_canonical_country(df: pd.DataFrame) -> pd.DataFrame:
    """Add country_canonical column using the normaliser."""
    df["country_canonical"] = df["country"].fillna("").apply(canonical_country)
    return df


# ===========================================================================
# 6. Main run function
# ===========================================================================


def run(
    split: str,
    data_dir: str,
    faiss_candidates_path: str,
    output_path: str,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Build TF-IDF supplement, merge with FAISS candidates, and save output.

    Parameters
    ----------
    split:
        "train" or "test".
    data_dir:
        Root dataset directory (parent of ``train/`` and ``test/`` subdirs).
    faiss_candidates_path:
        Path to the candidate_pairs_{split}.tsv produced by
        build_faiss_and_block.py.
    output_path:
        Where to write the merged candidate_pairs_{split}_with_tfidf.tsv.
    top_k:
        Number of TF-IDF candidates per query (default 20 — matches FAISS K).

    Returns
    -------
    Merged candidate pairs DataFrame.
    """
    split_dir = os.path.join(data_dir, split)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # ------------------------------------------------------------------
    # 6A. Load source files
    # ------------------------------------------------------------------
    s1_df = _load_source(
        os.path.join(split_dir, f"{split}_source1.tsv"), "S1"
    )
    s2_df = _load_source(
        os.path.join(split_dir, f"{split}_source2.tsv"), "S2"
    )
    s3_df = _load_source(
        os.path.join(split_dir, f"{split}_source3.tsv"), "S3"
    )
    query_df = pd.concat([s2_df, s3_df], ignore_index=True)

    s1_df = _add_canonical_country(s1_df)
    query_df = _add_canonical_country(query_df)

    countries = list(s1_df["country_canonical"].unique())
    if "Unknown" not in countries:
        countries.append("Unknown")

    # ------------------------------------------------------------------
    # 6B. Build one TF-IDF index per country
    # ------------------------------------------------------------------
    indices: Dict[str, Tuple] = {}
    for country in countries:
        if country == "Unknown":
            # Catch-all: S1 records not mapped to US/India/France
            unknown_s1 = s1_df[~s1_df["country_canonical"].isin(["US", "India", "France"])]
            if unknown_s1.empty:
                continue
            vec, mat, eids = build_tfidf_index(unknown_s1.assign(country_canonical="Unknown"), "Unknown")
        else:
            vec, mat, eids = build_tfidf_index(s1_df, country)
        if vec is not None:
            indices[country] = (vec, mat, eids)

    # ------------------------------------------------------------------
    # 6C. Query TF-IDF for all S2/S3 records in chunks
    # ------------------------------------------------------------------
    log.info("Querying TF-IDF for %d S2/S3 records (chunk=%d)...", len(query_df), CHUNK_SIZE)
    tfidf_rows: List[dict] = []
    n_chunks = (len(query_df) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(n_chunks):
        chunk = query_df.iloc[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
        tfidf_rows.extend(_query_batch(chunk, indices, top_k))
        if (i + 1) % 10 == 0:
            log.info("  chunk %d/%d done", i + 1, n_chunks)

    tfidf_df = pd.DataFrame(tfidf_rows)
    log.info("TF-IDF produced %d candidate pair rows.", len(tfidf_df))

    # ------------------------------------------------------------------
    # 6D. Load existing FAISS candidate pairs
    # ------------------------------------------------------------------
    if os.path.isfile(faiss_candidates_path):
        log.info("Loading FAISS candidates from %s ...", faiss_candidates_path)
        faiss_df = pd.read_csv(faiss_candidates_path, sep="\t", dtype=str, low_memory=False)
        faiss_df.columns = [c.strip().lower() for c in faiss_df.columns]
        # Ensure numeric columns are numeric
        for col in ("faiss_rank", "faiss_score"):
            if col in faiss_df.columns:
                faiss_df[col] = pd.to_numeric(faiss_df[col], errors="coerce")
    else:
        log.warning("FAISS candidates file not found at %s — TF-IDF only.", faiss_candidates_path)
        faiss_df = pd.DataFrame(columns=["query_entity_id", "candidate_s1_entity_id"])

    # ------------------------------------------------------------------
    # 6E. Union merge: FAISS left-join with TF-IDF; fill tfidf_rank for
    #     pairs already in FAISS; add new TF-IDF-only pairs.
    # ------------------------------------------------------------------
    merge_keys = ["query_entity_id", "candidate_s1_entity_id"]

    if not tfidf_df.empty:
        merged = faiss_df.merge(tfidf_df, on=merge_keys, how="outer")
    else:
        merged = faiss_df.copy()
        merged["tfidf_rank"] = pd.NA
        merged["tfidf_score"] = pd.NA

    # Deduplicate — keep the row with the lowest faiss_rank (most relevant)
    merged = merged.sort_values(
        ["query_entity_id", "faiss_rank", "tfidf_rank"],
        na_position="last",
    )
    merged = merged.drop_duplicates(subset=merge_keys, keep="first")
    merged = merged.reset_index(drop=True)

    log.info(
        "Merged candidate pairs: %d total (%d from FAISS, %d from TF-IDF only)",
        len(merged),
        len(faiss_df),
        len(merged) - len(faiss_df[faiss_df.set_index(merge_keys).index.isin(
            pd.MultiIndex.from_frame(faiss_df[merge_keys])
        )]),
    )

    # ------------------------------------------------------------------
    # 6F. Save
    # ------------------------------------------------------------------
    merged.to_csv(output_path, sep="\t", index=False)
    log.info("Saved merged candidates to %s", output_path)
    return merged


# ===========================================================================
# 7. CLI entrypoint (convenience — called from main.ipynb via !python or
#    directly from terminal)
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TF-IDF supplement blocking — Phase 4")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument(
        "--data-dir", default="../student_resource/dataset",
        help="Parent directory of train/ and test/ folders"
    )
    parser.add_argument(
        "--faiss-candidates", required=True,
        help="Path to candidate_pairs_{split}.tsv from build_faiss_and_block.py"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for merged candidate_pairs_{split}_with_tfidf.tsv"
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    run(
        split=args.split,
        data_dir=args.data_dir,
        faiss_candidates_path=args.faiss_candidates,
        output_path=args.output,
        top_k=args.top_k,
    )
