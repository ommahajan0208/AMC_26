"""
00_convert_to_parquet.py
========================
One-time conversion of all source TSV files to compressed Parquet.

Run this ONCE before anything else. It converts the raw TSVs to
zstd-compressed Parquet files, which are 3-4x smaller and load
50-100x faster than TSV for all downstream steps.

When WHOLE_DATA = False, it creates tiny dev Parquets with DEV_ROWS
rows per file so the rest of the pipeline runs in seconds.

Usage:
    python 00_convert_to_parquet.py
"""

import sys
import time
from pathlib import Path

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import duckdb
import config

config.ensure_output_dirs()
config.print_config()

TSV_TO_PARQUET = [
    (config.TRAIN_SOURCE1_TSV, config.TRAIN_SOURCE1_PARQUET, "train_source1"),
    (config.TRAIN_SOURCE2_TSV, config.TRAIN_SOURCE2_PARQUET, "train_source2"),
    (config.TRAIN_SOURCE3_TSV, config.TRAIN_SOURCE3_PARQUET, "train_source3"),
    (config.TRAIN_GT_TSV,      config.TRAIN_GT_PARQUET,      "train_ground_truth"),
    (config.TEST_SOURCE1_TSV,  config.TEST_SOURCE1_PARQUET,  "test_source1"),
    (config.TEST_SOURCE2_TSV,  config.TEST_SOURCE2_PARQUET,  "test_source2"),
    (config.TEST_SOURCE3_TSV,  config.TEST_SOURCE3_PARQUET,  "test_source3"),
]

t_total = time.perf_counter()
con = duckdb.connect()

for tsv_path, parquet_path, label in TSV_TO_PARQUET:
    if not tsv_path.exists():
        print(f"[SKIP] {label}: TSV not found at {tsv_path}")
        continue

    if parquet_path.exists():
        print(f"[SKIP] {label}: Parquet already exists at {parquet_path}")
        continue

    t0 = time.perf_counter()
    print(f"[CONV] {label}: {tsv_path.name} -> {parquet_path.name} ...", end=" ", flush=True)

    limit_rows   = config.effective_rows()
    limit_clause = f"LIMIT {limit_rows}" if limit_rows > 0 else ""

    # Use DuckDB to read TSV and write Parquet with zstd compression
    # read_csv auto-detects delimiter, headers, and types
    con.execute(f"""
        COPY (
            SELECT *
            FROM read_csv(
                '{tsv_path.as_posix()}',
                delim = '\t',
                header = true,
                ignore_errors = true,
                null_padding = true
            )
            {limit_clause}
        )
        TO '{parquet_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """)

    elapsed = time.perf_counter() - t0
    # Get row count
    row_count = con.execute(f"SELECT COUNT(*) FROM '{parquet_path.as_posix()}'").fetchone()[0]
    size_mb   = parquet_path.stat().st_size / 1e6
    print(f"{row_count:,} rows | {size_mb:.1f} MB | {elapsed:.1f}s")

con.close()
total_elapsed = time.perf_counter() - t_total
print(f"\n[DONE] All conversions complete in {total_elapsed:.1f}s")
print(f"       Parquet files in: {config.PARQUET_DIR}")
