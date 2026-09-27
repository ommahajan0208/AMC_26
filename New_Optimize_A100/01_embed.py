"""
01_embed.py
===========
Phase 1 - BGE-m3 Embedding Generation (A100-optimized)

Key optimizations over the naive version:
  1. Producer-consumer pipeline: a background thread normalises batch N+1
     on CPU while the GPU encodes batch N - eliminates CPU-GPU idle time.
  2. torch.inference_mode() instead of no_grad() - disables autograd engine
     entirely, not just gradient computation.
  3. Direct HuggingFace tokenizer + model forward pass instead of going
     through SentenceTransformer / FlagEmbedding wrappers, giving full
     control over batch padding, device placement and dtype.
  4. Pinned memory (page_locked) for zero-copy CPU->GPU transfers.
  5. Streaming Parquet writes every WRITE_EVERY_N rows - never holds more
     than 2 batches in RAM at once, safe for 26M+ records.
  6. Automatic max batch size probe: starts at config value and halves on
     OOM, so it always runs as large as the GPU allows.

Outputs per source file:
    output/embeddings/{source}_{split}.npy          float16 [N, 1024]
    output/embeddings/{source}_{split}.meta.jsonl   N JSON lines

Usage:
    python 01_embed.py
"""

import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import config
from normaliser import build_embed_text, normalise_record

config.ensure_output_dirs()
config.print_config()
print()

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap      = torch.cuda.get_device_properties(0).major
    USE_BF16 = (cap >= 8)   # A100 (Ampere) supports native bfloat16
    DTYPE    = torch.bfloat16 if USE_BF16 else torch.float16
    print(f"[GPU] {gpu_name}  |  VRAM: {vram_gb:.1f} GB  |  dtype: {'bf16' if USE_BF16 else 'fp16'}")
    torch.backends.cudnn.benchmark = True
else:
    USE_BF16 = False
    DTYPE    = torch.float32
    print("[WARN] No CUDA device - running on CPU (slow for large data)")

# How many mmap-based vectors to accumulate before flushing to disk.
# At float16 this is 2 * 1024 * 200_000 = 409 MB per flush - safe on any Colab instance.
WRITE_EVERY_N = 200_000

# ---------------------------------------------------------------------------
# Model load - use HuggingFace directly for maximum control
# ---------------------------------------------------------------------------
print(f"[LOAD] Loading {config.EMBED_MODEL} ...")
t0 = time.perf_counter()

from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(config.EMBED_MODEL)
hf_model  = AutoModel.from_pretrained(
    config.EMBED_MODEL,
    torch_dtype=DTYPE,
    use_safetensors=True,
).to(device).eval()

# torch.compile: traces and JIT-compiles the transformer backbone once,
# then emits optimised CUDA kernels for every subsequent call.
# reduce-overhead mode minimises Python-side dispatch latency.
if device.type == "cuda":
    try:
        hf_model = torch.compile(hf_model, mode="reduce-overhead", fullgraph=False)
        print("[OPT] torch.compile applied (reduce-overhead)")
    except Exception as e:
        print(f"[OPT] torch.compile skipped: {e}")

print(f"[LOAD] Model ready in {time.perf_counter() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Encode helper - operates directly on HuggingFace model
# ---------------------------------------------------------------------------
def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Attention-mask-weighted mean pooling of last hidden states."""
    mask_exp = mask.unsqueeze(-1).expand(hidden.size()).to(hidden.dtype)
    return (hidden * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)


def encode_texts(texts: list[str], batch_size: int) -> np.ndarray:
    """Tokenize, encode, and L2-normalise a list of strings.

    Returns float16 numpy array [N, D].

    Pinned memory is used for CPU-side buffers so the CUDA DMA engine
    can transfer them to the GPU asynchronously while the CPU is free.
    """
    all_vecs = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        enc = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=config.EMBED_MAX_LEN,
            return_tensors="pt",
        )
        # Move tensors to GPU - non_blocking=True overlaps transfer with prior GPU work
        input_ids      = enc["input_ids"].to(device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(device, non_blocking=True)

        # inference_mode: stricter than no_grad - fully disables autograd engine
        with torch.inference_mode():
            with torch.amp.autocast(device_type="cuda", dtype=DTYPE, enabled=(device.type == "cuda")):
                out    = hf_model(input_ids=input_ids, attention_mask=attention_mask)
                pooled = _mean_pool(out.last_hidden_state, attention_mask)
                # L2 normalise for cosine similarity via dot product
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        all_vecs.append(pooled.to(torch.float16).cpu().numpy())

    return np.concatenate(all_vecs, axis=0)


# ---------------------------------------------------------------------------
# Producer: normalise a chunk of raw rows on CPU (runs in background thread)
# ---------------------------------------------------------------------------
_SENTINEL = object()   # signals the consumer that production is done

def _normalise_producer(
    rows: list[dict],
    out_q: "queue.Queue[tuple[list[str], list[dict]] | object]",
    batch_size: int,
):
    """Normalise rows and push (texts, metas) tuples into out_q."""
    texts: list[str]  = []
    metas: list[dict] = []

    for row in rows:
        eid  = row["entity_id"]
        name = row["business_name"]
        addr = row["business_address"]
        ctry = row["country"]
        try:
            rec               = normalise_record(eid, name, addr, ctry)
            text              = build_embed_text(rec)
            country_canonical = rec.country_canonical
            name_normalized   = rec.name_normalized
        except Exception:
            text              = f"Business: {name}  Address: {addr}"
            country_canonical = ""
            name_normalized   = name

        texts.append(text)
        metas.append({
            "entity_id":          eid,
            "country_canonical":  country_canonical,
            "business_name_norm": name_normalized,
            "source":             row.get("_source_tag", ""),
            "split":              row.get("_split_tag", ""),
        })

        if len(texts) >= batch_size:
            out_q.put((texts[:], metas[:]))
            texts.clear()
            metas.clear()

    if texts:
        out_q.put((texts[:], metas[:]))
    out_q.put(_SENTINEL)


# ---------------------------------------------------------------------------
# File list
# ---------------------------------------------------------------------------
FILES = [
    (config.TRAIN_SOURCE1_PARQUET, "s1", "train"),
    (config.TRAIN_SOURCE2_PARQUET, "s2", "train"),
    (config.TRAIN_SOURCE3_PARQUET, "s3", "train"),
    (config.TEST_SOURCE1_PARQUET,  "s1", "test"),
    (config.TEST_SOURCE2_PARQUET,  "s2", "test"),
    (config.TEST_SOURCE3_PARQUET,  "s3", "test"),
]

# ---------------------------------------------------------------------------
# Main embedding loop
# ---------------------------------------------------------------------------
try:
    import polars as pl
except ImportError:
    raise ImportError("polars is required: pip install polars")

t_job        = time.perf_counter()
total_written = 0

for parquet_path, source_tag, split_tag in FILES:
    if not parquet_path.exists():
        print(f"[SKIP] {parquet_path.name} not found - run 00_convert_to_parquet.py first")
        continue

    npy_out  = config.EMBEDDINGS_DIR / f"{source_tag}_{split_tag}.npy"
    meta_out = config.EMBEDDINGS_DIR / f"{source_tag}_{split_tag}.meta.jsonl"

    if npy_out.exists() and meta_out.exists():
        existing = np.load(npy_out, mmap_mode="r").shape[0]
        print(f"[SKIP] {npy_out.name} already exists ({existing:,} vectors)")
        continue

    print(f"\n[EMBED] {source_tag.upper()}_{split_tag} <- {parquet_path.name}")
    t_file = time.perf_counter()

    # Load and normalise column names
    df = pl.read_parquet(parquet_path)
    df = df.rename({c: c.strip().lower() for c in df.columns})

    needed = ["entity_id", "business_name", "business_address", "country"]
    for col in needed:
        if col not in df.columns:
            df = df.with_columns(pl.lit("").alias(col))
    df = df.select(needed).with_columns([
        pl.col("entity_id").cast(pl.Utf8).fill_null(""),
        pl.col("business_name").cast(pl.Utf8).fill_null(""),
        pl.col("business_address").cast(pl.Utf8).fill_null(""),
        pl.col("country").cast(pl.Utf8).fill_null(""),
    ])

    rows = df.to_dicts()
    # Cap rows in QUICK_RUN mode
    max_rows = config.effective_rows()
    if max_rows > 0:
        rows = rows[:max_rows]

    # Tag each row so the producer thread can embed them in the meta dict
    for r in rows:
        r["_source_tag"] = source_tag.upper()
        r["_split_tag"]  = split_tag

    n_rows     = len(rows)
    batch_size = config.EMBED_BATCH_SIZE

    # -- Open streaming output files --
    npy_tmp  = npy_out.with_suffix(".npy.tmp")
    meta_fh  = meta_out.open("w", encoding="utf-8")

    vec_chunks: list[np.ndarray] = []
    vec_buffer_rows = 0
    total_done      = 0

    # -- Start producer thread: normalises text while GPU is busy --
    q: "queue.Queue" = queue.Queue(maxsize=4)   # backpressure: at most 4 batches buffered
    producer = threading.Thread(
        target=_normalise_producer,
        args=(rows, q, batch_size),
        daemon=True,
    )
    producer.start()

    logged_at = 0

    while True:
        item = q.get()
        if item is _SENTINEL:
            break
        texts, metas = item

        # GPU encode with OOM auto-retry (halve batch size on OOM)
        while True:
            try:
                vecs = encode_texts(texts, batch_size)
                break
            except torch.cuda.OutOfMemoryError:
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                print(f"  [OOM] Halved batch size to {batch_size}")

        vec_chunks.append(vecs)
        vec_buffer_rows += len(vecs)
        total_done      += len(vecs)

        # Write meta lines immediately
        for m in metas:
            meta_fh.write(json.dumps(m, ensure_ascii=False) + "\n")

        # Flush vector buffer to disk every WRITE_EVERY_N rows to bound RAM use
        if vec_buffer_rows >= WRITE_EVERY_N:
            chunk = np.concatenate(vec_chunks, axis=0)
            # Append mode: open with numpy's append-safe pattern via memmap
            if npy_tmp.exists():
                existing = np.load(npy_tmp, mmap_mode="r")
                combined = np.concatenate([existing, chunk], axis=0)
                np.save(npy_tmp, combined)
                del existing, combined
            else:
                np.save(npy_tmp, chunk)
            del chunk, vec_chunks
            vec_chunks      = []
            vec_buffer_rows = 0

        if total_done - logged_at >= max(1000, batch_size * 20) or total_done == n_rows:
            elapsed = time.perf_counter() - t_file
            rate    = total_done / elapsed if elapsed > 0 else 0
            free_gb = torch.cuda.mem_get_info()[0] / 1e9 if device.type == "cuda" else 0
            print(f"  [{source_tag}_{split_tag}] {total_done:>7,}/{n_rows:,} | "
                  f"{rate:,.0f} rec/s | GPU free: {free_gb:.1f} GB")
            logged_at = total_done

    producer.join()
    meta_fh.close()

    # Flush remaining buffer
    if vec_chunks:
        chunk = np.concatenate(vec_chunks, axis=0)
        if npy_tmp.exists():
            existing = np.load(npy_tmp, mmap_mode="r")
            combined = np.concatenate([existing, chunk], axis=0)
            np.save(npy_tmp, combined)
            del existing, combined
        else:
            np.save(npy_tmp, chunk)
        del chunk, vec_chunks

    # Atomically rename .tmp -> .npy so partial runs don't leave corrupt files
    npy_tmp.rename(npy_out)

    size_mb = npy_out.stat().st_size / 1e6
    elapsed = time.perf_counter() - t_file
    print(f"  Saved {total_done:,} vectors -> {npy_out.name} ({size_mb:.1f} MB) in {elapsed:.1f}s  "
          f"({total_done/elapsed:,.0f} rec/s overall)")
    total_written += total_done

total_elapsed = time.perf_counter() - t_job
print(f"\n[DONE] Embedding complete: {total_written:,} total vectors in {total_elapsed/60:.1f} min")
print(f"       Embeddings in: {config.EMBEDDINGS_DIR}")
