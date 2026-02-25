"""Write final assembled dataset to Parquet files with metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from bin2vec.assemble.grouping import group_functions
from bin2vec.assemble.splitting import split_by_identity
from bin2vec.utils.logging import get_logger
from bin2vec.utils.paths import dataset_dir

log = get_logger("writer")


def _collect_extracted_tables(extracted_root: Path) -> pa.Table:
    """Read and concatenate all extracted Parquet files."""
    parquet_files = sorted(extracted_root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {extracted_root}")

    log.info("Reading %d extracted Parquet files", len(parquet_files))
    tables = [pq.read_table(f) for f in tqdm(parquet_files, desc="Reading", unit="file")]
    return pa.concat_tables(tables)


def assemble_dataset(
    data_dir: str,
    min_variants: int = 10,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Assemble the final dataset from extracted Parquet files.

    Returns {"train": N, "val": N, "test": N} row counts.
    """
    from bin2vec.utils.paths import data_root

    extracted_root = data_root(data_dir) / "extracted"
    table = _collect_extracted_tables(extracted_root)

    log.info("Total extracted rows: %d", len(table))

    # Group by function identity
    groups = group_functions(table, min_variants=min_variants)

    if not groups:
        log.warning("No function groups with >= %d variants found", min_variants)
        return {"train": 0, "val": 0, "test": 0}

    # Add identity key column
    from bin2vec.assemble.grouping import compute_identity_key

    identity_keys = []
    for i in range(len(table)):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}
        identity_keys.append(compute_identity_key(row))
    table = table.append_column("identity_key", pa.array(identity_keys))

    # Split by identity
    splits = split_by_identity(list(groups.keys()), train_ratio, val_ratio, seed)

    # Write output
    out_dir = dataset_dir(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split_name, keys in splits.items():
        key_set = set(keys)
        # Collect all row indices belonging to this split
        indices = []
        for key in keys:
            if key in groups:
                indices.extend(groups[key])

        if not indices:
            counts[split_name] = 0
            continue

        split_table = table.take(indices)
        out_path = out_dir / f"{split_name}.parquet"
        pq.write_table(split_table, out_path)
        counts[split_name] = len(split_table)
        log.info("Wrote %s: %d rows (%d functions)", split_name, len(split_table), len(keys))

    # Write metadata
    metadata = {
        "min_variants": min_variants,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "seed": seed,
        "total_identities": len(groups),
        "split_counts": counts,
        "split_identity_counts": {k: len(v) for k, v in splits.items()},
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info("Metadata written to %s", meta_path)

    return counts
