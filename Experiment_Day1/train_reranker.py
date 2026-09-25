"""
train_reranker.py
==================
SageMaker Training Job entry point — bge-reranker-v2-m3 Fine-tuning (Phase 7)

Strategy reference: ER_Strategy.md §7 (features), §8 (two-stage model),
                     §9 (validation), §10 Q8/Q13/Q14/Q27/Q32
Model             : BAAI/bge-reranker-v2-m3  (MIT, ~568M params, multilingual
                     cross-encoder)
Instance          : ml.g4dn.xlarge  (1×T4 16GB VRAM, 4 vCPU, 16GB RAM)

What this job does
-------------------
1. Reads train_ground_truth.tsv, candidate_pairs_train.tsv (FAISS blocking
   output — Phase 4) and the three raw train_source{1,2,3}.tsv files from
   /opt/ml/input/data/.
2. Normalises records with normaliser.normalise_record() and builds the
   cross-encoder side-text with normaliser.build_embed_text(), i.e. exactly
   "Business: {name}  Address: {addr}" (or name-only when addr_is_null) —
   the same text convention used for the BGE-m3 bi-encoder in
   train_embeddings.py, per ER_Strategy.md §8.
3. Assembles training pairs from a *sample* of ground truth (see
   --max-positives — fine-tuning on all ~7M+ GT rows is outside a 1-3 hr /
   g4dn.xlarge budget; §2C estimates ~3 hrs / ~$4.52 for this job):
     positive        — every sampled (query_id, s1_id) row from GT, label=1
     hard negative    — same-FAISS-bucket S1 candidates for that query with
                         faiss_score > --hard-neg-cosine (0.7) that are NOT
                         the ground-truth match                        (Q27)
     easy negative     — a random S1 entity from the same canonical country
                          that is not a GT match for that query         (Q27)
     ratio              — 1 positive : --neg-per-pos negatives (default 3),
                           split --hard-neg-frac (0.8) hard / 0.2 easy  (Q27/Q32)
4. Fine-tunes BAAI/bge-reranker-v2-m3 as a binary cross-encoder
   (AutoModelForSequenceClassification, num_labels=1, BCEWithLogitsLoss),
   fp16 autocast, batch_size=16, on AWS g4dn.xlarge.
   Tokenizer sentence-pair encoding (tokenizer(text_a, text_b, ...)) is the
   exact equivalent of the strategy's
       "Business: {name1} Address: {addr1} [SEP] Business: {name2} Address: {addr2}"
   input format — the tokenizer inserts the separator automatically.
5. Holds out a stratified slice of GT pairs (--val-frac) for an in-job
   sanity check (accuracy / AUC / approx F0.5 @ 0.5). This is NOT a
   substitute for the full Eval 1/2/3 harness in ER_Strategy.md §9, which
   runs after this job as part of the full two-stage pipeline.
6. Saves the fine-tuned model + tokenizer to /opt/ml/model/ (SageMaker
   auto-uploads to S3) and a training manifest to /opt/ml/output/data/.

Usage (called by SageMaker, not directly):
    python train_reranker.py [--max-positives 300000]
                              [--neg-per-pos 3] [--hard-neg-frac 0.8]
                              [--hard-neg-cosine 0.7]
                              [--batch-size 16] [--epochs 2] [--lr 2e-5]
                              [--fp16]

Environment variables (set by SageMaker automatically):
    SM_CHANNEL_DATA     — directory containing ground truth / candidate pairs
                           / source TSVs
    SM_OUTPUT_DATA_DIR  — directory to write the training manifest
    SM_MODEL_DIR        — directory for the fine-tuned model artefacts
    SM_NUM_GPUS         — number of available GPUs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

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
_SM_MODEL_DIR  = os.environ.get("SM_MODEL_DIR",       "/opt/ml/model")
_SM_NUM_GPUS   = int(os.environ.get("SM_NUM_GPUS", "1"))

_SOURCE_FILES = {
    "s1": "train_source1.tsv",
    "s2": "train_source2.tsv",
    "s3": "train_source3.tsv",
}


# ===========================================================================
# 1. Normaliser import (normaliser.py is expected on PYTHONPATH)
# ===========================================================================

def _import_normaliser():
    """Import normaliser; supports both installed package and local file."""
    try:
        from normaliser import normalise_record, build_embed_text  # type: ignore
        log.info("Imported normaliser from installed package.")
        return normalise_record, build_embed_text
    except ImportError:
        pass
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
        "train_reranker.py or install it as a package."
    )


# ===========================================================================
# 2. TSV readers
# ===========================================================================

def _detect_cols(header_line: str) -> dict[str, int]:
    cols = [c.strip().lower() for c in header_line.split("\t")]
    try:
        return {
            "id":   cols.index("entity_id"),
            "name": cols.index("business_name"),
            "addr": cols.index("business_address"),
            "ctry": cols.index("country"),
        }
    except ValueError as exc:
        raise ValueError(f"Missing expected column: {exc}. Found: {cols}") from exc


def _clean(v: str) -> str:
    v = v.strip()
    return "" if v.upper() in ("NAN", "NONE", "NULL", "") else v


def iter_source_rows(
    filepath: Path,
    wanted_ids: Optional[set[str]] = None,
) -> Iterator[tuple[str, str, str, str]]:
    """
    Stream (entity_id, business_name, business_address, country) from a
    source TSV. If `wanted_ids` is given, only rows whose entity_id is in
    that set are yielded — this bounds memory when we only need text for a
    small subset of a multi-million-row source file.
    """
    with filepath.open(encoding="utf-8", errors="replace") as fh:
        idx = _detect_cols(fh.readline().rstrip("\n"))
        for line in fh:
            parts = line.rstrip("\n").split("\t")

            def _get(i: int) -> str:
                return _clean(parts[i]) if i < len(parts) else ""

            entity_id = _get(idx["id"])
            if wanted_ids is not None and entity_id not in wanted_ids:
                continue
            yield entity_id, _get(idx["name"]), _get(idx["addr"]), _get(idx["ctry"])


def load_source_country_map(filepath: Path) -> dict[str, str]:
    """
    Lightweight pass over a source TSV: entity_id -> raw country only.
    Used to build country pools for easy-negative sampling without holding
    full name/address text for every record.
    """
    out: dict[str, str] = {}
    for entity_id, _name, _addr, country in iter_source_rows(filepath):
        out[entity_id] = country
    return out


def build_text_store(
    filepath: Path,
    wanted_ids: set[str],
    normalise_record,
    build_embed_text,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Targeted scan of one source TSV: for every entity_id in `wanted_ids`,
    normalise the record and build its cross-encoder side-text.

    Returns (id -> text, id -> country_canonical).
    """
    texts:     dict[str, str] = {}
    countries: dict[str, str] = {}
    if not wanted_ids:
        return texts, countries

    remaining = set(wanted_ids)
    for entity_id, name, addr, country in iter_source_rows(filepath, wanted_ids):
        try:
            rec = normalise_record(entity_id, name, addr, country)
            texts[entity_id]     = build_embed_text(rec)
            countries[entity_id] = rec.country_canonical
        except Exception as exc:
            log.debug("Normalisation failed for %s: %s", entity_id, exc)
            texts[entity_id]     = f"Business: {name}  Address: {addr}"
            countries[entity_id] = country or "Unknown"
        remaining.discard(entity_id)
        if not remaining:
            break

    if remaining:
        log.warning(
            "%d/%d wanted ids not found in %s (dropped from training set)",
            len(remaining), len(wanted_ids), filepath.name,
        )
    return texts, countries


# ===========================================================================
# 3. Ground truth
# ===========================================================================

def load_ground_truth(gt_path: Path) -> tuple[list[tuple[str, str]], dict[str, set[str]]]:
    """
    Read train_ground_truth.tsv → (positive_pairs, gt_lookup).

    positive_pairs : list of (query_entity_id, s1_entity_id)   — every GT row
    gt_lookup      : query_entity_id -> {all correct s1_entity_id}
                     (used to make sure negative sampling never picks a
                     record that is secretly a correct match for that query)

    Column names are detected flexibly, matching the logic used in
    build_faiss_and_block.py's evaluate_blocking_recall().
    """
    with gt_path.open(encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n")
        cols = [c.strip().lower() for c in header.split("\t")]

        def _find(pred) -> Optional[int]:
            for i, c in enumerate(cols):
                if pred(c):
                    return i
            return None

        i_s1  = _find(lambda c: "s1" in c and "entity" in c) or _find(lambda c: "s1" in c)
        i_s23 = _find(lambda c: ("s2" in c or "s3" in c) and "entity" in c) or \
                _find(lambda c: "s2" in c or "s3" in c or "match" in c)
        if i_s1 is None or i_s23 is None:
            i_s1, i_s23 = 0, 1
            log.warning("GT column detection ambiguous — assuming col0=s1, col1=s23")

        log.info("GT columns: s1=idx%d  s23=idx%d", i_s1, i_s23)

        positive_pairs: list[tuple[str, str]] = []
        gt_lookup: dict[str, set[str]] = defaultdict(set)

        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_s1, i_s23):
                continue
            s1_id, s23_id = parts[i_s1].strip(), parts[i_s23].strip()
            if not s1_id or not s23_id:
                continue
            positive_pairs.append((s23_id, s1_id))
            gt_lookup[s23_id].add(s1_id)

    log.info(
        "Ground truth loaded: %d pairs  |  %d unique query ids  |  %d unique S1 ids",
        len(positive_pairs), len(gt_lookup),
        len({s1 for pairs in gt_lookup.values() for s1 in pairs}),
    )
    return positive_pairs, gt_lookup


# ===========================================================================
# 4. Hard-negative mining from candidate_pairs_train.tsv
# ===========================================================================

def mine_hard_negatives(
    candidate_pairs_path: Path,
    train_query_ids: set[str],
    gt_lookup: dict[str, set[str]],
    hard_neg_cosine: float,
    max_candidates_per_query: int,
) -> dict[str, list[tuple[str, float]]]:
    """
    Stream candidate_pairs_train.tsv once. For each row whose query is in
    `train_query_ids`, keep it as a hard-negative candidate iff:
        faiss_score > hard_neg_cosine   AND   candidate_s1_id not in GT match set

    Returns query_id -> [(s1_id, faiss_score), ...] sorted by score desc,
    capped at `max_candidates_per_query` per query (keeps memory bounded
    and gives the sampler its best-scoring near-misses first — Q27).
    """
    pool: dict[str, list[tuple[str, float]]] = defaultdict(list)
    if not train_query_ids:
        return pool

    log.info(
        "Mining hard negatives from %s for %d query ids (cosine > %.2f) …",
        candidate_pairs_path.name, len(train_query_ids), hard_neg_cosine,
    )
    t0 = time.perf_counter()
    rows_scanned = 0
    rows_kept    = 0

    with candidate_pairs_path.open(encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        i_q   = header.index("query_entity_id")
        i_s1  = header.index("candidate_s1_entity_id")
        i_sc  = header.index("faiss_score")

        for line in fh:
            rows_scanned += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_q, i_s1, i_sc):
                continue
            qid = parts[i_q]
            if qid not in train_query_ids:
                continue
            try:
                score = float(parts[i_sc])
            except ValueError:
                continue
            if score <= hard_neg_cosine:
                continue
            s1_id = parts[i_s1]
            if s1_id in gt_lookup.get(qid, ()):
                continue  # this is actually a correct match — not a negative

            bucket = pool[qid]
            if len(bucket) < max_candidates_per_query:
                bucket.append((s1_id, score))
                rows_kept += 1
            elif score > bucket[-1][1]:
                bucket[-1] = (s1_id, score)
                bucket.sort(key=lambda t: -t[1])
                rows_kept += 1

            if rows_scanned % 20_000_000 == 0:
                log.info(
                    "  scanned %d rows | kept %d hard-neg candidates | %.1f min",
                    rows_scanned, rows_kept, (time.perf_counter() - t0) / 60,
                )

    for qid in pool:
        pool[qid].sort(key=lambda t: -t[1])

    log.info(
        "Hard-negative mining done: %d rows scanned, %d queries with ≥1 hard "
        "candidate, %.1f min",
        rows_scanned, len(pool), (time.perf_counter() - t0) / 60,
    )
    return pool


# ===========================================================================
# 5. Country pools for easy-negative sampling
# ===========================================================================

def build_country_pools(
    s1_country_raw: dict[str, str],
    canonical_country_fn,
) -> dict[str, list[str]]:
    """entity_id -> canonical_country, grouped into per-country id lists."""
    pools: dict[str, list[str]] = defaultdict(list)
    for entity_id, raw_country in s1_country_raw.items():
        pools[canonical_country_fn(raw_country)].append(entity_id)
    log.info(
        "Country pools for easy negatives: %s",
        {c: len(v) for c, v in sorted(pools.items())},
    )
    return pools


def sample_easy_negative(
    country: str,
    exclude: set[str],
    pools: dict[str, list[str]],
    rng: random.Random,
    max_tries: int = 20,
) -> Optional[str]:
    """Pick a random S1 id from `country`'s pool that isn't in `exclude`."""
    pool = pools.get(country) or pools.get("Unknown")
    if not pool:
        return None
    for _ in range(max_tries):
        cand = pool[rng.randrange(len(pool))]
        if cand not in exclude:
            return cand
    return None


# ===========================================================================
# 6. Training-pair assembly
# ===========================================================================

class TrainingPair:
    __slots__ = ("query_id", "s1_id", "label")

    def __init__(self, query_id: str, s1_id: str, label: int) -> None:
        self.query_id = query_id
        self.s1_id    = s1_id
        self.label    = label


def assemble_pairs(
    sampled_positives: list[tuple[str, str]],
    gt_lookup: dict[str, set[str]],
    hard_neg_pool: dict[str, list[tuple[str, float]]],
    country_pools: dict[str, list[str]],
    query_country: dict[str, str],
    neg_per_pos: int,
    hard_neg_frac: float,
    rng: random.Random,
) -> list[TrainingPair]:
    """
    For every sampled (query_id, s1_id) positive, emit:
        1 positive pair
        neg_per_pos negative pairs, split hard_neg_frac hard / rest easy
    per ER_Strategy.md §7/§27/§32.
    """
    n_hard_target = round(neg_per_pos * hard_neg_frac)
    n_easy_target = neg_per_pos - n_hard_target

    pairs: list[TrainingPair] = []
    stats = {"pos": 0, "hard": 0, "easy": 0, "hard_shortfall": 0}

    for query_id, s1_id in sampled_positives:
        pairs.append(TrainingPair(query_id, s1_id, 1))
        stats["pos"] += 1

        gt_matches = gt_lookup.get(query_id, {s1_id})
        used_negs: set[str] = set()

        # --- hard negatives -------------------------------------------------
        candidates = hard_neg_pool.get(query_id, [])
        n_hard_taken = 0
        for cand_id, _score in candidates:
            if n_hard_taken >= n_hard_target:
                break
            if cand_id in gt_matches or cand_id in used_negs:
                continue
            pairs.append(TrainingPair(query_id, cand_id, 0))
            used_negs.add(cand_id)
            n_hard_taken += 1
            stats["hard"] += 1

        shortfall = n_hard_target - n_hard_taken
        if shortfall:
            stats["hard_shortfall"] += shortfall

        # --- easy negatives (also backfill any hard-negative shortfall) -----
        n_easy_needed = n_easy_target + shortfall
        country = query_country.get(query_id, "Unknown")
        exclude = gt_matches | used_negs | {s1_id}
        n_easy_taken = 0
        for _ in range(n_easy_needed):
            cand_id = sample_easy_negative(country, exclude, country_pools, rng)
            if cand_id is None:
                break
            pairs.append(TrainingPair(query_id, cand_id, 0))
            exclude.add(cand_id)
            n_easy_taken += 1
            stats["easy"] += 1

    log.info(
        "Assembled %d training pairs  |  positives=%d  hard_neg=%d  "
        "easy_neg=%d  (hard shortfall backfilled with easy: %d)",
        len(pairs), stats["pos"], stats["hard"], stats["easy"],
        stats["hard_shortfall"],
    )
    return pairs


# ===========================================================================
# 7. PyTorch dataset / collation
# ===========================================================================

def build_dataset(pairs: list[TrainingPair], texts: dict[str, str]):
    import torch

    class PairDataset(torch.utils.data.Dataset):
        def __init__(self, items: list[TrainingPair]):
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, i: int):
            p = self.items[i]
            text_a = texts.get(p.query_id, f"Business: ")
            text_b = texts.get(p.s1_id, f"Business: ")
            return text_a, text_b, float(p.label)

    return PairDataset(pairs)


def make_collate_fn(tokenizer, max_length: int):
    import torch

    def collate(batch):
        texts_a = [b[0] for b in batch]
        texts_b = [b[1] for b in batch]
        labels  = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        enc = tokenizer(
            texts_a,
            texts_b,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc

    return collate


# ===========================================================================
# 8. Training loop
# ===========================================================================

def train_reranker(
    train_pairs: list[TrainingPair],
    val_pairs: list[TrainingPair],
    texts: dict[str, str],
    model_name: str,
    batch_size: int,
    epochs: int,
    lr: float,
    warmup_ratio: float,
    max_length: int,
    use_fp16: bool,
    output_model_dir: Path,
) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log.warning("No CUDA device found — training on CPU (very slow).")
        use_fp16 = False

    log.info("Loading tokenizer + model '%s' …", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1
    ).to(device)

    collate_fn = make_collate_fn(tokenizer, max_length)
    train_ds = build_dataset(train_pairs, texts)
    val_ds   = build_dataset(val_pairs, texts)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps  = max(1, len(train_loader) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    log.info(
        "Training config: %d train pairs | %d val pairs | batch=%d | "
        "epochs=%d | lr=%s | fp16=%s | steps/epoch=%d",
        len(train_ds), len(val_ds), batch_size, epochs, lr, use_fp16,
        len(train_loader),
    )

    best_val_f05 = -1.0
    history: list[dict] = []
    t_train = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        t_epoch = time.perf_counter()

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_fp16):
                out = model(**batch)
                logits = out.logits.squeeze(-1)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()

            if step % 500 == 0:
                elapsed = time.perf_counter() - t_epoch
                rate = step * batch_size / elapsed if elapsed > 0 else 0
                log.info(
                    "  epoch %d/%d | step %d/%d | loss=%.4f | %.0f pairs/s",
                    epoch, epochs, step, len(train_loader),
                    running_loss / step, rate,
                )

        val_metrics = evaluate(model, val_loader, device, use_fp16)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = running_loss / max(1, len(train_loader))
        val_metrics["epoch_minutes"] = round((time.perf_counter() - t_epoch) / 60, 2)
        history.append(val_metrics)

        log.info(
            "=== Epoch %d done: train_loss=%.4f | val_acc=%.4f | val_auc=%.4f | "
            "val_f05@0.5=%.4f | %.1f min ===",
            epoch, val_metrics["train_loss"], val_metrics["accuracy"],
            val_metrics["auc"], val_metrics["f0.5"], val_metrics["epoch_minutes"],
        )

        if val_metrics["f0.5"] >= best_val_f05:
            best_val_f05 = val_metrics["f0.5"]
            log.info("  New best val F0.5=%.4f — saving checkpoint.", best_val_f05)
            output_model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_model_dir)
            tokenizer.save_pretrained(output_model_dir)

    elapsed_hr = (time.perf_counter() - t_train) / 3600
    log.info("Training complete in %.2f hrs. Best val F0.5=%.4f", elapsed_hr, best_val_f05)

    return {
        "history": history,
        "best_val_f0.5": round(best_val_f05, 4),
        "train_hours": round(elapsed_hr, 3),
    }


def evaluate(model, val_loader, device, use_fp16: bool) -> dict:
    """
    In-job sanity check only (accuracy / AUC / approx F0.5 @ 0.5 threshold).
    The authoritative reranker evaluation is the standalone + full-pipeline
    F0.5 runs in ER_Strategy.md §9 (Phase 7/9 checklist items), which need
    the actual downstream 0.60–0.90 uncertain-band population from Stage 1.
    """
    import torch

    model.eval()
    all_probs:  list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.cuda.amp.autocast(enabled=use_fp16):
                out = model(**batch)
                probs = torch.sigmoid(out.logits.squeeze(-1))
            all_probs.extend(probs.float().cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    preds  = (probs >= 0.5).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    accuracy  = (tp + tn) / max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    beta2 = 0.5 ** 2
    f0_5 = (
        (1 + beta2) * precision * recall / (beta2 * precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    auc = _approx_auc(probs, labels)

    return {
        "accuracy":  round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall":    round(float(recall), 4),
        "f0.5":      round(float(f0_5), 4),
        "auc":       round(float(auc), 4),
        "n_val":     int(len(labels)),
    }


def _approx_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (no sklearn dependency)."""
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
        if len(set(labels.tolist())) < 2:
            return 0.5
        return float(roc_auc_score(labels, probs))
    except ImportError:
        pos = probs[labels == 1]
        neg = probs[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        order = np.argsort(probs)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(probs) + 1)
        rank_sum_pos = ranks[labels == 1].sum()
        n_pos, n_neg = len(pos), len(neg)
        auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return float(auc)


# ===========================================================================
# 9. CLI + main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ER Pipeline Phase 7 — bge-reranker-v2-m3 fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-name", default="BAAI/bge-reranker-v2-m3",
                   help="HuggingFace model id (Q13 — locked).")
    p.add_argument("--max-length", type=int, default=512,
                   help="Tokenizer max sequence length for the pair.")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Per-step batch size (Q8/§2B: fits ~4GB VRAM on T4).")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--neg-per-pos", type=int, default=3,
                   help="Positive:negative ratio — locked 1:3 (Q27/Q32).")
    p.add_argument("--hard-neg-frac", type=float, default=0.8,
                   help="Fraction of negatives that are hard (Q27 — locked 80%%).")
    p.add_argument("--hard-neg-cosine", type=float, default=0.7,
                   help="FAISS cosine threshold for a hard negative (Q27 — locked).")
    p.add_argument("--max-candidates-per-query", type=int, default=15,
                   help="Cap on hard-negative candidates kept per query "
                        "before sampling (bounds memory; K=20 FAISS top-K).")
    p.add_argument("--max-positives", type=int, default=300_000,
                   help="Subsample of GT positives to train on. Full GT is "
                        "several million rows — outside the ~1-3hr / "
                        "g4dn.xlarge budget in ER_Strategy.md §2C. Raise "
                        "this if more GPU-hours are available. 0 = use all.")
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="Fraction of sampled positives held out for the "
                        "in-job validation check.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-fp16", action="store_true", default=False)
    p.add_argument("--gt-file", default="train_ground_truth.tsv")
    p.add_argument("--candidate-pairs-file", default="candidate_pairs_train.tsv")
    p.add_argument("--input-dir", default=_SM_INPUT_DIR)
    p.add_argument("--output-dir", default=_SM_OUTPUT_DIR)
    p.add_argument("--model-dir", default=_SM_MODEL_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_dir  = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("ER Pipeline — Phase 7: bge-reranker-v2-m3 fine-tuning")
    log.info(
        "  neg_per_pos=%d  hard_neg_frac=%.2f  hard_neg_cosine=%.2f  "
        "max_positives=%s",
        args.neg_per_pos, args.hard_neg_frac, args.hard_neg_cosine,
        args.max_positives or "ALL",
    )
    log.info("  input_dir=%s  output_dir=%s  model_dir=%s",
              input_dir, output_dir, model_dir)
    log.info("=" * 70)

    t_job = time.perf_counter()

    normalise_record, build_embed_text = _import_normaliser()
    from normaliser import canonical_country  # same module, direct import

    # -----------------------------------------------------------------------
    # Phase A — Ground truth + positive sampling
    # -----------------------------------------------------------------------
    log.info("[Phase A] Loading ground truth …")
    positive_pairs, gt_lookup = load_ground_truth(input_dir / args.gt_file)

    if args.max_positives and args.max_positives < len(positive_pairs):
        sampled_positives = rng.sample(positive_pairs, args.max_positives)
        log.info(
            "Subsampled %d / %d GT positives for training.",
            len(sampled_positives), len(positive_pairs),
        )
    else:
        sampled_positives = positive_pairs

    n_val = max(1, int(len(sampled_positives) * args.val_frac))
    rng.shuffle(sampled_positives)
    val_positives   = sampled_positives[:n_val]
    train_positives = sampled_positives[n_val:]
    log.info(
        "Split: %d train positives / %d val positives",
        len(train_positives), len(val_positives),
    )

    all_query_ids = {q for q, _ in sampled_positives}

    # -----------------------------------------------------------------------
    # Phase B — Hard-negative mining from FAISS blocking output
    # -----------------------------------------------------------------------
    log.info("[Phase B] Mining hard negatives …")
    hard_neg_pool = mine_hard_negatives(
        candidate_pairs_path      = input_dir / args.candidate_pairs_file,
        train_query_ids           = all_query_ids,
        gt_lookup                 = gt_lookup,
        hard_neg_cosine           = args.hard_neg_cosine,
        max_candidates_per_query  = args.max_candidates_per_query,
    )
    n_no_hard = len(all_query_ids) - len(hard_neg_pool)
    if n_no_hard:
        log.warning(
            "%d / %d queries have no hard-negative candidate above cosine>%.2f "
            "— will backfill with easy negatives.",
            n_no_hard, len(all_query_ids), args.hard_neg_cosine,
        )

    # -----------------------------------------------------------------------
    # Phase C — Country pools (S1-side, for easy negatives)
    # -----------------------------------------------------------------------
    log.info("[Phase C] Building S1 country pools …")
    s1_country_raw = load_source_country_map(input_dir / _SOURCE_FILES["s1"])
    country_pools = build_country_pools(s1_country_raw, canonical_country)
    del s1_country_raw

    # -----------------------------------------------------------------------
    # Phase D — Targeted text scans (only ids we actually need)
    # -----------------------------------------------------------------------
    log.info("[Phase D] Building query-side (S2/S3) text store …")
    query_ids_by_file: dict[str, set[str]] = defaultdict(set)
    for qid in all_query_ids:
        prefix = qid.split("-")[0].lower() if "-" in qid else ""
        if prefix in ("s2", "s3"):
            query_ids_by_file[prefix].add(qid)
        else:
            log.debug("Unrecognised query id prefix, skipping: %s", qid)

    texts: dict[str, str] = {}
    query_country: dict[str, str] = {}
    for src in ("s2", "s3"):
        wanted = query_ids_by_file.get(src, set())
        if not wanted:
            continue
        t, c = build_text_store(
            input_dir / _SOURCE_FILES[src], wanted, normalise_record, build_embed_text
        )
        texts.update(t)
        query_country.update(c)

    log.info(
        "[Phase D] Query-side text built for %d / %d ids.",
        len(texts), len(all_query_ids),
    )

    # Needed S1 ids: every positive s1 + every hard-neg candidate we may draw
    needed_s1_ids: set[str] = {s1 for _, s1 in sampled_positives}
    for cands in hard_neg_pool.values():
        needed_s1_ids.update(s1 for s1, _ in cands)
    # Also reserve some headroom for easy-negative sampling: include a random
    # slice of each country pool so the text store already covers likely
    # draws (avoids a second S1 pass mid-assembly).
    for country, ids in country_pools.items():
        sample_n = min(len(ids), max(2_000, len(query_ids_by_file) * 2))
        needed_s1_ids.update(rng.sample(ids, sample_n) if len(ids) > sample_n else ids)

    log.info(
        "[Phase D] Building S1-side text store for %d ids …", len(needed_s1_ids)
    )
    s1_texts, _ = build_text_store(
        input_dir / _SOURCE_FILES["s1"], needed_s1_ids, normalise_record, build_embed_text
    )
    texts.update(s1_texts)
    log.info("[Phase D] Total text store size: %d ids.", len(texts))

    # -----------------------------------------------------------------------
    # Phase E — Assemble train / val pair sets
    # -----------------------------------------------------------------------
    log.info("[Phase E] Assembling training pairs …")
    train_pairs = assemble_pairs(
        train_positives, gt_lookup, hard_neg_pool, country_pools,
        query_country, args.neg_per_pos, args.hard_neg_frac, rng,
    )
    val_pairs = assemble_pairs(
        val_positives, gt_lookup, hard_neg_pool, country_pools,
        query_country, args.neg_per_pos, args.hard_neg_frac, rng,
    )

    # Drop any pair whose text is missing (e.g. easy-negative id not covered
    # by the headroom sample above) rather than crash mid-training.
    def _has_text(p: TrainingPair) -> bool:
        return p.query_id in texts and p.s1_id in texts

    before = len(train_pairs) + len(val_pairs)
    train_pairs = [p for p in train_pairs if _has_text(p)]
    val_pairs   = [p for p in val_pairs if _has_text(p)]
    after = len(train_pairs) + len(val_pairs)
    if after < before:
        log.warning(
            "Dropped %d / %d pairs with missing text (untextified easy "
            "negatives) before training.", before - after, before,
        )

    # -----------------------------------------------------------------------
    # Phase F — Fine-tune
    # -----------------------------------------------------------------------
    log.info("[Phase F] Fine-tuning %s …", args.model_name)
    train_results = train_reranker(
        train_pairs       = train_pairs,
        val_pairs         = val_pairs,
        texts             = texts,
        model_name        = args.model_name,
        batch_size        = args.batch_size,
        epochs            = args.epochs,
        lr                = args.lr,
        warmup_ratio      = args.warmup_ratio,
        max_length        = args.max_length,
        use_fp16          = not args.no_fp16,
        output_model_dir  = model_dir,
    )

    # -----------------------------------------------------------------------
    # Phase G — Manifest
    # -----------------------------------------------------------------------
    elapsed_hr = (time.perf_counter() - t_job) / 3600
    manifest = {
        "phase":               "7-reranker-finetune",
        "model_name":          args.model_name,
        "gt_pairs_total":      len(positive_pairs),
        "positives_sampled":   len(sampled_positives),
        "train_positives":     len(train_positives),
        "val_positives":       len(val_positives),
        "train_pairs":         len(train_pairs),
        "val_pairs":           len(val_pairs),
        "neg_per_pos":         args.neg_per_pos,
        "hard_neg_frac":       args.hard_neg_frac,
        "hard_neg_cosine":     args.hard_neg_cosine,
        "queries_with_no_hard_candidate": n_no_hard,
        "batch_size":          args.batch_size,
        "epochs":              args.epochs,
        "lr":                  args.lr,
        "fp16":                not args.no_fp16,
        "seed":                args.seed,
        "elapsed_hrs":         round(elapsed_hr, 3),
        **train_results,
    }
    manifest_path = output_dir / "manifest_reranker_train.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written: %s", manifest_path)

    log.info(
        "=== Phase 7 complete: %.2f hrs total | best val F0.5=%.4f ===",
        elapsed_hr, train_results.get("best_val_f0.5", -1.0),
    )


if __name__ == "__main__":
    main()
