"""BasicBlockEncoder: Transformer over instruction tokens within a basic block.

Architecture
------------
Each basic block is represented as a sequence of normalised instruction token
IDs.  A small Transformer encoder (typically 2–4 layers, 128–256 hidden dims)
processes the sequence and produces a fixed-size embedding by pooling the CLS
token output.

The encoder is **shared** across all basic blocks in all functions — it learns
a universal instruction-level representation.

Input
-----
token_ids : LongTensor  [batch_size, max_seq_len]
    Padded token-ID sequences.  PAD_ID (0) positions are masked out.

Output
------
block_emb : FloatTensor  [batch_size, hidden_dim]
    One embedding vector per basic block.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from bin2vec.preprocess.tokenizer import PAD_ID, CLS_ID


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(
        self, hidden_dim: int, max_len: int = 512, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it moves with the model (cuda/cpu)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, hidden_dim]

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to *x* [B, L, D]."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BasicBlockEncoder(nn.Module):
    """Encode a padded batch of basic blocks into fixed-size vectors.

    Parameters
    ----------
    vocab_size:
        Total vocabulary size (including special tokens).
    hidden_dim:
        Transformer hidden / embedding dimension.
    num_heads:
        Number of attention heads (must divide *hidden_dim*).
    num_layers:
        Number of Transformer encoder layers.
    ffn_dim:
        Feedforward layer inner dimension inside each Transformer block.
    max_seq_len:
        Maximum number of tokens per basic block (for positional encoding).
    dropout:
        Dropout probability applied in attention, FFN, and embeddings.
    pad_id:
        Token ID used for padding (masked in attention).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 512,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        pad_id: int = PAD_ID,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pad_id = pad_id

        # Token embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=pad_id,
        )

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(
            hidden_dim=hidden_dim,
            max_len=max_seq_len,
            dropout=dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,  # [B, L, D] convention
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,  # avoids a PyTorch warning with padding masks
        )

        # Layer norm on output
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, token_ids: Tensor) -> Tensor:
        """Encode a batch of token-ID sequences.

        Args:
            token_ids: LongTensor of shape ``[B, L]`` where *B* is batch size
                       and *L* is (padded) sequence length.

        Returns:
            FloatTensor of shape ``[B, hidden_dim]`` — the CLS-token output
            for each basic block.
        """
        # Build padding mask: True where position is padding (to be ignored)
        pad_mask = token_ids.eq(self.pad_id)  # [B, L]

        # Embed tokens and add positional encodings
        x = self.embedding(token_ids)  # [B, L, D]
        x = self.pos_enc(x)  # [B, L, D]

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=pad_mask)  # [B, L, D]
        x = self.out_norm(x)

        # Pool: use CLS token (position 0) as the block embedding
        block_emb = x[:, 0, :]  # [B, D]
        return block_emb
