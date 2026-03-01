"""Bin2Vec top-level model.

End-to-end pipeline:

    raw_bytes (per function)
        │
        ▼  CFG extraction (Capstone, offline or on-the-fly)
    CFGGraph (BasicBlock list + edge list)
        │
        ▼  Token encoding per basic block (BasicBlockEncoder)
    node embeddings  [num_blocks, hidden_dim]
        │
        ▼  GNN message passing over CFG topology (CFGEncoder)
    function embedding  [embed_dim]

The model is designed to be used with a **contrastive loss** (NT-Xent / InfoNCE
or triplet loss) so that compiled versions of the same source function are pulled
together in embedding space while different functions are pushed apart.

Usage
-----
>>> cfg = Bin2VecConfig()
>>> model = Bin2VecModel(cfg)
>>> # Prepare a PyG batch (see bin2vec.train.dataset for helpers)
>>> emb = model(token_ids, edge_index, batch)  # [B, embed_dim]
>>> # L2-normalise before computing cosine similarities
>>> emb = torch.nn.functional.normalize(emb, dim=-1)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from bin2vec.model.basic_block_encoder import BasicBlockEncoder
from bin2vec.model.cfg_encoder import CFGEncoder
from bin2vec.preprocess.tokenizer import PAD_ID


@dataclass
class Bin2VecConfig:
    """Hyperparameter bundle for :class:`Bin2VecModel`.

    Attributes
    ----------
    vocab_size:
        Tokenizer vocabulary size (including special tokens).
    bb_hidden_dim:
        Hidden dimension of the BasicBlockEncoder Transformer.
    bb_num_heads:
        Attention heads in the BasicBlockEncoder.
    bb_num_layers:
        Number of Transformer layers in the BasicBlockEncoder.
    bb_ffn_dim:
        FFN inner dimension of the BasicBlockEncoder.
    bb_max_seq_len:
        Maximum tokens per basic block (longer blocks are truncated).
    bb_dropout:
        Dropout inside BasicBlockEncoder.
    gnn_hidden_dim:
        Hidden dimension of the CFGEncoder GATv2 layers.
    gnn_out_dim:
        Final function embedding dimension output by CFGEncoder.
    gnn_num_layers:
        Number of GATv2 message-passing rounds.
    gnn_num_heads:
        Attention heads per GATv2 layer.
    gnn_dropout:
        Dropout inside CFGEncoder.
    gnn_residual:
        Whether to use residual connections in the CFGEncoder.
    embed_dim:
        Final L2-normalised embedding dimension (after optional projection).
        If None, defaults to gnn_out_dim.
    """

    vocab_size: int = 32_768
    # BasicBlockEncoder
    bb_hidden_dim: int = 128
    bb_num_heads: int = 4
    bb_num_layers: int = 2
    bb_ffn_dim: int = 512
    bb_max_seq_len: int = 128
    bb_dropout: float = 0.1
    # CFGEncoder
    gnn_hidden_dim: int = 128
    gnn_out_dim: int = 128
    gnn_num_layers: int = 3
    gnn_num_heads: int = 4
    gnn_dropout: float = 0.1
    gnn_residual: bool = True
    # Final embedding
    embed_dim: int | None = None  # defaults to gnn_out_dim

    def __post_init__(self) -> None:
        if self.embed_dim is None:
            self.embed_dim = self.gnn_out_dim


class Bin2VecModel(nn.Module):
    """Full Bin2Vec model: raw_bytes → CFG → function embedding.

    Parameters
    ----------
    config:
        :class:`Bin2VecConfig` with all hyperparameters.
    pad_id:
        Token ID used for padding in BasicBlockEncoder input.
    """

    def __init__(self, config: Bin2VecConfig, pad_id: int = PAD_ID) -> None:
        super().__init__()
        self.config = config

        # Basic-block encoder (shared across all blocks)
        self.bb_encoder = BasicBlockEncoder(
            vocab_size=config.vocab_size,
            hidden_dim=config.bb_hidden_dim,
            num_heads=config.bb_num_heads,
            num_layers=config.bb_num_layers,
            ffn_dim=config.bb_ffn_dim,
            max_seq_len=config.bb_max_seq_len,
            dropout=config.bb_dropout,
            pad_id=pad_id,
        )

        # CFG encoder
        self.cfg_encoder = CFGEncoder(
            node_feat_dim=config.bb_hidden_dim,
            hidden_dim=config.gnn_hidden_dim,
            out_dim=config.gnn_out_dim,
            num_layers=config.gnn_num_layers,
            num_heads=config.gnn_num_heads,
            dropout=config.gnn_dropout,
            residual=config.gnn_residual,
        )

        # Optional final projection to a different embed_dim
        assert config.embed_dim is not None  # set by __post_init__
        if config.embed_dim != config.gnn_out_dim:
            self.proj: nn.Module = nn.Linear(config.gnn_out_dim, config.embed_dim)
        else:
            self.proj = nn.Identity()

    def encode_blocks(self, token_ids: Tensor) -> Tensor:
        """Encode a batch of basic blocks → node feature matrix.

        Args:
            token_ids: LongTensor [num_blocks, max_seq_len]

        Returns:
            FloatTensor [num_blocks, bb_hidden_dim]
        """
        return self.bb_encoder(token_ids)

    def forward(
        self,
        token_ids: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """Compute function embeddings.

        Args:
            token_ids:   LongTensor [num_blocks_total, max_seq_len]
                         Padded token-ID sequences for every basic block
                         across all graphs in the batch.
            edge_index:  LongTensor [2, num_edges_total]
                         COO edge list for all graphs (node indices are
                         global within the batch, as produced by
                         ``torch_geometric.data.Batch``).
            batch:       LongTensor [num_blocks_total]
                         Graph membership for each node (0..B-1).

        Returns:
            FloatTensor [B, embed_dim] — **not** L2-normalised.
            Normalise with ``F.normalize(emb, dim=-1)`` before computing
            cosine similarities or applying contrastive losses.
        """
        # Step 1: encode every basic block independently
        node_emb = self.bb_encoder(token_ids)  # [N, bb_hidden_dim]

        # Step 2: message-pass over the CFG topology
        graph_emb = self.cfg_encoder(node_emb, edge_index, batch)  # [B, gnn_out_dim]

        # Step 3: optional linear projection
        emb = self.proj(graph_emb)  # [B, embed_dim]
        return emb

    # ------------------------------------------------------------------
    # Convenience: count parameters
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the number of (trainable) parameters."""
        params = (
            self.parameters()
            if not trainable_only
            else (p for p in self.parameters() if p.requires_grad)
        )
        return sum(p.numel() for p in params)
