"""
train_embeddings.py
===================
SageMaker Training Job entry point — BGE-m3 Embedding Pass
Business Entity Resolution pipeline (Phase 3)

Strategy reference: ER_Strategy.md §2C, §2F, §3
Model             : BAAI/bge-m3  (MIT, 570M params, 100+ languages)
Instance          : ml.g4dn.4xlarge  (1×T4 16GB VRAM, 16 vCPU, 64GB RAM)

What this job does
------------------
1. Reads normalised TSV files from /opt/ml/input/data/ (SageMaker channel).
2. Normalises every record with normaliser.normalise_record() + build_embed_text().
3. Encodes in batches of 256 with BGE-m3, fp16, torch.no_grad().
4. Saves embeddings as .npy shards (500K records each) to /opt/ml/output/data/.
5. Saves a companion .meta.jsonl shard alongside each .npy with entity_ids +
   country_canonical so downstream FAISS sharding can work per-country.

SageMaker auto-uploads everything in /opt/ml/output/data/ to S3 after the job.

Usage (called by SageMaker, not directly):
    python train_embeddings.py [--job-mode train|test]
                               [--batch-size 256]
                               [--shard-size 500000]
                               [--fp16]
                               [--model-name BAAI/bge-m3]

Two jobs are expected (ER_Strategy.md §2F):
    Job 1 (train): processes train_source1.tsv, train_source2.tsv, train_source3.tsv
    Job 2 (test):  processes test_source1.tsv,  test_source2.tsv,  test_source3.tsv

Environment variables (set by SageMaker automatically):
    SM_CHANNEL_DATA   — directory containing the input TSV files
    SM_OUTPUT_DATA_DIR — directory to write output shards
    SM_MODEL_DIR      — directory for any saved model artefacts (unused here)
    SM_NUM_GPUS       — number of available GPUs

Shard naming convention:
    {source}_{split}_emb_shard{N:04d}.npy          (float16 ndarray, shape [K, 1024])
    {source}_{split}_emb_shard{N:04d}.meta.jsonl   (one JSON line per record)

    e.g.  s1_train_emb_shard0000.npy   /   s1_train_emb_shard0000.meta.jsonl
          s2_train_emb_shard0000.npy   ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SageMaker path constants (overridable via CLI for local testing)
# ---------------------------------------------------------------------------

_SM_INPUT_DIR   = os.environ.get("SM_CHANNEL_DATA",    "/opt/ml/input/data")
_SM_OUTPUT_DIR  = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
_SM_MODEL_DIR   = os.environ.get("SM_MODEL_DIR",       "/opt/ml/model")
_SM_NUM_GPUS    = int(os.environ.get("SM_NUM_GPUS", "1"))


# ---------------------------------------------------------------------------
# File layout expected in the data channel
# ---------------------------------------------------------------------------

_TRAIN_FILES = [
    ("train_source1.tsv", "s1", "train"),
    ("train_source2.tsv", "s2", "train"),
    ("train_source3.tsv", "s3", "train"),
]

_TEST_FILES = [
    ("test_source1.tsv",  "s1", "test"),
    ("test_source2.tsv",  "s2", "test"),
    ("test_source3.tsv",  "s3", "test"),
]


# ---------------------------------------------------------------------------
# 1. Normaliser import (normaliser.py is expected on PYTHONPATH)
# ---------------------------------------------------------------------------

def _import_normaliser():
    """Import normaliser; supports both installed package and local file."""
    try:
        from normaliser import normalise_record, build_embed_text  # type: ignore
        log.info("Imported normaliser from installed package.")
        return normalise_record, build_embed_text
    except ImportError:
        pass
    # Try loading from the same directory as this script
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir / "normaliser.py"
    if candidate.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("normaliser", candidate)
        mod  = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)                   # type: ignore[union-attr]
        log.info("Imported normaliser from %s", candidate)
        return mod.normalise_record, mod.build_embed_text
    raise ImportError(
        "Could not import normaliser. Place normaliser.py next to "
        "train_embeddings.py or install it as a package."
    )


# ---------------------------------------------------------------------------
# 2. TSV iterator — yields (entity_id, business_name, business_address, country)
# ---------------------------------------------------------------------------

def _iter_tsv(filepath: Path) -> Iterator[tuple[str, str, str, str]]:
    """
    Lazily read a source TSV.

    Expected header columns (order not fixed — detected from header row):
        entity_id | business_name | business_address | country

    Yields 4-tuples; missing/NaN values come through as empty strings.
    """
    with filepath.open(encoding="utf-8", errors="replace") as fh:
        header_line = fh.readline().rstrip("\n")
        cols = [c.strip().lower() for c in header_line.split("\t")]

        try:
            i_id   = cols.index("entity_id")
            i_name = cols.index("business_name")
            i_addr = cols.index("business_address")
            i_ctry = cols.index("country")
        except ValueError as exc:
            raise ValueError(
                f"Missing expected column in {filepath}: {exc}. "
                f"Found columns: {cols}"
            ) from exc

        for lineno, line in enumerate(fh, start=2):
            parts = line.rstrip("\n").split("\t")
            # Guard against short rows
            def _get(idx: int) -> str:
                if idx >= len(parts):
                    return ""
                v = parts[idx].strip()
                return "" if v.upper() in ("NAN", "NONE", "NULL", "") else v

            yield _get(i_id), _get(i_name), _get(i_addr), _get(i_ctry)


# ---------------------------------------------------------------------------
# 3. Model loader
# ---------------------------------------------------------------------------

def _load_model(model_name: str, use_fp16: bool, device: torch.device):
    """
    Load BGE-m3 via sentence-transformers SentenceTransformer.

    Falls back to FlagEmbedding (if installed) for better BGE-m3 support.
    """
    log.info("Loading model '%s' …", model_name)
    t0 = time.perf_counter()

    try:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore
        model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        model_type = "FlagEmbedding"
    except ImportError:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer(model_name, device=str(device))
        if use_fp16 and device.type == "cuda":
            model = model.half()
        model_type = "SentenceTransformer"

    log.info(
        "Model loaded via %s in %.1fs  |  device=%s  fp16=%s",
        model_type, time.perf_counter() - t0, device, use_fp16,
    )
    return model, model_type


# ---------------------------------------------------------------------------
# 4. Embedding function — handles both SentenceTransformer and FlagEmbedding
# ---------------------------------------------------------------------------

def _encode_batch(
    model,
    model_type: str,
    texts: list[str],
    use_fp16: bool,
) -> np.ndarray:
    """
    Encode a list of texts → float16 numpy array of shape [N, 1024].
    """
    with torch.no_grad():
        if model_type == "FlagEmbedding":
            # Returns dict; 'dense_vecs' is the primary BGE-m3 dense embedding
            out = model.encode(
                texts,
                batch_size=len(texts),
                max_length=512,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            vecs = out["dense_vecs"]  # np.ndarray float32
        else:
            # SentenceTransformer path
            vecs = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )

    if vecs.dtype != np.float16:
        vecs = vecs.astype(np.float16)
    return vecs


# ---------------------------------------------------------------------------
# 5. Shard writer
# ---------------------------------------------------------------------------

class ShardWriter:
    """
    Accumulates records and flushes to .npy + .meta.jsonl when the shard
    reaches `shard_size` records or when close() is called.
    """

    def __init__(self, output_dir: Path, prefix: str, shard_size: int):
        self.output_dir = output_dir
        self.prefix     = prefix            # e.g. "s2_train"
        self.shard_size = shard_size
        self._shard_idx = 0
        self._vecs:  list[np.ndarray] = []
        self._metas: list[dict]       = []
        self.total_written = 0

    def add(self, vec: np.ndarray, meta: dict):
        self._vecs.append(vec)
        self._metas.append(meta)
        if len(self._vecs) >= self.shard_size:
            self._flush()

    def _flush(self):
        if not self._vecs:
            return
        shard_name = f"{self.prefix}_emb_shard{self._shard_idx:04d}"
        npy_path  = self.output_dir / f"{shard_name}.npy"
        meta_path = self.output_dir / f"{shard_name}.meta.jsonl"

        arr = np.stack(self._vecs, axis=0)          # [K, 1024] float16
        np.save(npy_path, arr)

        with meta_path.open("w", encoding="utf-8") as fh:
            for m in self._metas:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")

        n = len(self._vecs)
        self.total_written += n
        log.info(
            "Flushed shard %04d → %s  (%d records, cumulative %d)",
            self._shard_idx, shard_name, n, self.total_written,
        )
        self._shard_idx += 1
        self._vecs  = []
        self._metas = []

    def close(self):
        self._flush()


# ---------------------------------------------------------------------------
# 6. Core embedding loop for a single source file
# ---------------------------------------------------------------------------

def embed_source_file(
    filepath: Path,
    source_tag: str,     # "s1" / "s2" / "s3"
    split_tag: str,      # "train" / "test"
    model,
    model_type: str,
    normalise_record,
    build_embed_text,
    output_dir: Path,
    batch_size: int,
    shard_size: int,
    use_fp16: bool,
) -> int:
    """
    Embed one source file end-to-end.
    Returns total number of records written.
    """
    if not filepath.exists():
        log.warning("File not found, skipping: %s", filepath)
        return 0

    prefix = f"{source_tag}_{split_tag}"
    log.info("=== Embedding %s → prefix='%s' ===", filepath.name, prefix)

    writer = ShardWriter(output_dir, prefix, shard_size)

    # Accumulate a mini-batch of (text, meta) before encoding
    batch_texts: list[str]  = []
    batch_metas: list[dict] = []

    t_file_start = time.perf_counter()
    row_count    = 0
    logged_at    = 0

    def _flush_mini_batch():
        if not batch_texts:
            return
        vecs = _encode_batch(model, model_type, batch_texts, use_fp16)
        for vec, meta in zip(vecs, batch_metas):
            writer.add(vec, meta)
        batch_texts.clear()
        batch_metas.clear()

    for entity_id, name, addr, country in _iter_tsv(filepath):
        try:
            rec  = normalise_record(entity_id, name, addr, country)
            text = build_embed_text(rec)
        except Exception as exc:
            log.debug("Normalisation failed for %s: %s", entity_id, exc)
            # Use raw fallback so we never drop a record
            text = f"Business: {name}  Address: {addr}"
            rec  = None

        meta = {
            "entity_id":         entity_id,
            "country_canonical": rec.country_canonical if rec else "",
            "is_domain":         int(rec.is_domain) if rec else 0,
            "addr_is_null":      int(rec.addr_is_null) if rec else 0,
            "source":            source_tag.upper(),
            "split":             split_tag,
        }

        batch_texts.append(text)
        batch_metas.append(meta)
        row_count += 1

        if len(batch_texts) >= batch_size:
            _flush_mini_batch()

        # Progress logging every 100K records
        if row_count - logged_at >= 100_000:
            elapsed   = time.perf_counter() - t_file_start
            rate      = row_count / elapsed if elapsed > 0 else 0
            remaining = (row_count / rate / 3600) if rate > 0 else 0
            log.info(
                "  %s | %7d rows | %.0f rec/s | ~%.1f hrs remaining",
                filepath.name, row_count, rate, remaining,
            )
            logged_at = row_count

    _flush_mini_batch()   # flush tail batch
    writer.close()

    elapsed = time.perf_counter() - t_file_start
    log.info(
        "Finished %s: %d records in %.1f min  (%.0f rec/s)",
        filepath.name, row_count, elapsed / 60, row_count / elapsed,
    )
    return writer.total_written


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BGE-m3 embedding job")
    parser.add_argument(
        "--job-mode",
        choices=["train", "test", "both"],
        default="train",
        help="Which split to embed (default: train). "
             "Run two separate jobs: one for 'train', one for 'test'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size per GPU forward pass (default: 256).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=500_000,
        help="Records per .npy shard (default: 500000).",
    )
    parser.add_argument(
        "--model-name",
        default="BAAI/bge-m3",
        help="HuggingFace model id (default: BAAI/bge-m3).",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        default=False,
        help="Disable fp16 (use fp32). Not recommended on T4.",
    )
    parser.add_argument(
        "--input-dir",
        default=_SM_INPUT_DIR,
        help="Directory containing source TSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=_SM_OUTPUT_DIR,
        help="Directory to write .npy shards.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    use_fp16   = not args.no_fp16
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Device setup -------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info("GPU: %s  |  VRAM: %.1f GB  |  fp16: %s", gpu_name, vram_gb, use_fp16)
    else:
        log.warning("No CUDA device found — running on CPU (slow).")
        use_fp16 = False  # fp16 on CPU is not meaningful

    # --- Normaliser import --------------------------------------------------
    normalise_record, build_embed_text = _import_normaliser()

    # --- Model load ---------------------------------------------------------
    model, model_type = _load_model(args.model_name, use_fp16, device)

    # --- Determine which files to process -----------------------------------
    if args.job_mode == "train":
        file_list = _TRAIN_FILES
    elif args.job_mode == "test":
        file_list = _TEST_FILES
    else:  # "both"
        file_list = _TRAIN_FILES + _TEST_FILES

    log.info(
        "Job mode: %s  |  Files to process: %d  |  batch_size=%d  shard_size=%d",
        args.job_mode, len(file_list), args.batch_size, args.shard_size,
    )

    # --- Embed each file ----------------------------------------------------
    t_job_start   = time.perf_counter()
    total_records = 0

    for filename, source_tag, split_tag in file_list:
        filepath = input_dir / filename
        n = embed_source_file(
            filepath         = filepath,
            source_tag       = source_tag,
            split_tag        = split_tag,
            model            = model,
            model_type       = model_type,
            normalise_record = normalise_record,
            build_embed_text = build_embed_text,
            output_dir       = output_dir,
            batch_size       = args.batch_size,
            shard_size       = args.shard_size,
            use_fp16         = use_fp16,
        )
        total_records += n

    # --- Job summary --------------------------------------------------------
    elapsed_hr = (time.perf_counter() - t_job_start) / 3600
    log.info(
        "=== Job complete: %d records total in %.2f hrs ===",
        total_records, elapsed_hr,
    )

    # Write a manifest so downstream jobs know what was produced
    manifest_path = output_dir / f"manifest_{args.job_mode}.json"
    manifest = {
        "job_mode":       args.job_mode,
        "model_name":     args.model_name,
        "total_records":  total_records,
        "elapsed_hrs":    round(elapsed_hr, 3),
        "embedding_dim":  1024,
        "dtype":          "float16" if use_fp16 else "float32",
        "shard_size":     args.shard_size,
        "files_processed": [f for f, _, _ in file_list],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written: %s", manifest_path)


if __name__ == "__main__":
    main()
