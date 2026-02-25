"""Group extracted functions into equivalence classes by identity."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from bin2vec.utils.logging import get_logger

log = get_logger("grouping")


def compute_identity_key(row: dict) -> str:
    """Compute a function identity key.

    For static functions (source_file is meaningful), include source_file
    to distinguish same-named statics in different files.
    """
    name = row["function_name"]
    pkg = row["package_name"]
    source = row["source_file"]

    # Use source file in key to handle static function name collisions
    return f"{pkg}::{source}::{name}"


def group_functions(
    table: pa.Table,
    min_variants: int = 10,
) -> dict[str, list[int]]:
    """Group table rows by function identity.

    Returns {identity_key: [row_indices]} for groups with >= min_variants.
    """
    groups: dict[str, list[int]] = {}

    for i in range(len(table)):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}
        key = compute_identity_key(row)
        groups.setdefault(key, []).append(i)

    # Filter by minimum variant count
    filtered = {k: v for k, v in groups.items() if len(v) >= min_variants}

    log.info(
        "Grouped %d functions into %d identities (%d with >= %d variants)",
        len(table),
        len(groups),
        len(filtered),
        min_variants,
    )

    return filtered
