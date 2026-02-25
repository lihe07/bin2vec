"""Train/val/test split by function identity (no data leakage)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def split_by_identity(
    identity_keys: Sequence[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split function identity keys into train/val/test sets.

    Uses deterministic hashing so the split is reproducible.
    All variants of a function go to the same split (no leakage).

    Returns {"train": [...], "val": [...], "test": [...]}.
    """
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    val_threshold = train_ratio
    test_threshold = train_ratio + val_ratio

    for key in identity_keys:
        # Deterministic hash → float in [0, 1)
        h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
        bucket = int(h[:8], 16) / 0xFFFFFFFF

        if bucket < val_threshold:
            splits["train"].append(key)
        elif bucket < test_threshold:
            splits["val"].append(key)
        else:
            splits["test"].append(key)

    return splits
