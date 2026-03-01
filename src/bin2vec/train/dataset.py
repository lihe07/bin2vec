"""Bin2Vec contrastive dataset and PyG data-graph construction.

Dataset design
--------------
The assembled dataset (``data/dataset/{train,val,test}.parquet``) has one row
per **compiled variant** of a function.  Multiple rows with the same
``identity_key`` are different compilations of the same source function.

For contrastive training we use **in-batch negatives** (NT-Xent / SimCLR
style):

1. Sample a batch of *identity keys* (i.e. source functions).
2. For each identity, randomly pick 2 different compiled variants → (view_A, view_B).
3. The model should produce similar embeddings for (view_A, view_B) of the
   same identity and dissimilar embeddings for different identities.

Each view is represented as a ``torch_geometric.data.Data`` object:
    * ``x``          : LongTensor [num_blocks, max_seq_len] — tokenised blocks
    * ``edge_index`` : LongTensor [2, num_edges]
    * ``identity``   : str — the identity_key string (for loss computation)

The :class:`FunctionDataset` caches the ``(raw_bytes, isa, identity_key)``
tuples in memory.  CFG extraction + tokenization happen on the fly in the
``__getitem__`` worker process to spread I/O cost.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

try:
    from torch_geometric.data import Data, Batch
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "torch-geometric is required. Install with: uv add torch-geometric"
    ) from _err

from bin2vec.preprocess.cfg import extract_cfg, CFGGraph
from bin2vec.preprocess.tokenizer import Tokenizer, PAD_ID


# ---------------------------------------------------------------------------
# Helper: CFGGraph → PyG Data
# ---------------------------------------------------------------------------


def cfg_to_pyg(
    cfg: CFGGraph,
    tokenizer: Tokenizer,
    max_seq_len: int = 128,
) -> Data:
    """Convert a :class:`~bin2vec.preprocess.cfg.CFGGraph` to a PyG ``Data`` object.

    Args:
        cfg:         Extracted CFG with normalised instruction strings.
        tokenizer:   Vocabulary to encode instructions to integer IDs.
        max_seq_len: Truncate / pad each basic block to this many tokens.

    Returns:
        ``Data`` with:
            * ``x``          LongTensor [num_blocks, max_seq_len]
            * ``edge_index`` LongTensor [2, num_edges]
    """
    blocks = cfg.blocks
    if not blocks:
        # Degenerate: empty function — return a single all-PAD block, no edges
        x = torch.full((1, max_seq_len), PAD_ID, dtype=torch.long)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    # Encode each basic block's instruction list
    encoded: list[list[int]] = tokenizer.encode_batch(
        [bb.instructions for bb in blocks],
        add_cls=True,
        add_sep=False,
        max_length=max_seq_len,
        pad=True,
    )
    x = torch.tensor(encoded, dtype=torch.long)  # [num_blocks, max_seq_len]

    # Build edge_index
    if cfg.edges:
        edge_index = (
            torch.tensor(cfg.edges, dtype=torch.long).t().contiguous()
        )  # [2, E]
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


# ---------------------------------------------------------------------------
# Record type
# ---------------------------------------------------------------------------


class FunctionRecord:
    """Lightweight record for one compiled variant of a function."""

    __slots__ = ("raw_bytes", "isa", "identity_key")

    def __init__(self, raw_bytes: bytes, isa: str, identity_key: str) -> None:
        self.raw_bytes = raw_bytes
        self.isa = isa
        self.identity_key = identity_key


# ---------------------------------------------------------------------------
# Identity-keyed dataset
# ---------------------------------------------------------------------------


class FunctionDataset(Dataset):
    """Dataset of function identities, each holding ≥2 compiled variants.

    This dataset returns **pairs** (view_A, view_B) of different compiled
    variants of the same source function.  Used with NT-Xent contrastive loss
    where the batch contains pairs from many different identities.

    Parameters
    ----------
    parquet_path:
        Path to ``train.parquet`` / ``val.parquet`` / ``test.parquet``.
    tokenizer:
        Pre-built :class:`~bin2vec.preprocess.tokenizer.Tokenizer`.
    max_seq_len:
        Maximum tokens per basic block fed to the model.
    min_variants:
        Identities with fewer compiled variants than this are dropped.
    seed:
        RNG seed for reproducible pair sampling.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer: Tokenizer,
        max_seq_len: int = 128,
        min_variants: int = 2,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.rng = random.Random(seed)

        # Load the parquet file into memory (row-by-row)
        table = pq.read_table(
            str(parquet_path), columns=["raw_bytes", "isa", "identity_key"]
        )

        # Group by identity_key
        identity_map: dict[str, list[FunctionRecord]] = {}
        for batch in table.to_batches():
            for raw_bytes, isa, ikey in zip(
                batch.column("raw_bytes").to_pylist(),
                batch.column("isa").to_pylist(),
                batch.column("identity_key").to_pylist(),
            ):
                if raw_bytes is None or len(raw_bytes) < 4:
                    continue
                rec = FunctionRecord(bytes(raw_bytes), isa, ikey)
                identity_map.setdefault(ikey, []).append(rec)

        # Keep only identities with enough variants
        self.identities: list[str] = [
            k for k, v in identity_map.items() if len(v) >= min_variants
        ]
        self.identity_map = identity_map

    def __len__(self) -> int:
        return len(self.identities)

    def __getitem__(self, idx: int) -> Tuple[Data, Data, str]:
        """Return a (view_A, view_B, identity_key) tuple.

        Both views are different compiled variants of the same function.
        CFG extraction and tokenization happen here (in the worker process).
        """
        ikey = self.identities[idx]
        variants = self.identity_map[ikey]

        # Pick two distinct variants randomly
        v_a, v_b = self.rng.sample(variants, 2)

        data_a = self._to_pyg(v_a)
        data_b = self._to_pyg(v_b)
        return data_a, data_b, ikey

    def _to_pyg(self, record: FunctionRecord) -> Data:
        """Extract CFG and convert to PyG Data."""
        try:
            cfg = extract_cfg(record.raw_bytes, record.isa)
        except Exception:
            # Fallback: single empty block (rare — malformed bytes)
            from bin2vec.preprocess.cfg import CFGGraph, BasicBlock

            cfg = CFGGraph(blocks=[BasicBlock(start_offset=0)], edges=[])
        return cfg_to_pyg(cfg, self.tokenizer, self.max_seq_len)


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------


def collate_fn(
    items: list[Tuple[Data, Data, str]],
) -> Tuple[Batch, Batch, list[str]]:
    """Collate a list of (data_a, data_b, identity) tuples into batches.

    Returns:
        ``(batch_a, batch_b, identity_keys)`` where ``batch_a`` and
        ``batch_b`` are :class:`torch_geometric.data.Batch` objects
        (auto-handles variable graph sizes).
    """
    data_as, data_bs, keys = zip(*items)
    batch_a = Batch.from_data_list(list(data_as))
    batch_b = Batch.from_data_list(list(data_bs))
    return batch_a, batch_b, list(keys)


def make_dataloader(
    parquet_path: str | Path,
    tokenizer: Tokenizer,
    batch_size: int = 64,
    max_seq_len: int = 128,
    min_variants: int = 2,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42,
) -> DataLoader:
    """Convenience factory for a :class:`FunctionDataset` + :class:`DataLoader`.

    Args:
        parquet_path: Path to a split parquet file.
        tokenizer:    Vocabulary for instruction encoding.
        batch_size:   Number of function **identities** per batch (each yields
                      2 views → effective batch size is 2 × batch_size).
        max_seq_len:  Token-ID sequence length per basic block.
        min_variants: Minimum compiled variants to include an identity.
        shuffle:      Whether to shuffle at each epoch.
        num_workers:  DataLoader worker processes (CFG extraction is CPU-bound).
        seed:         Seed for pair sampling.

    Returns:
        A :class:`~torch.utils.data.DataLoader` yielding
        ``(batch_a, batch_b, identity_keys)`` tuples.
    """
    dataset = FunctionDataset(
        parquet_path=parquet_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        min_variants=min_variants,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
