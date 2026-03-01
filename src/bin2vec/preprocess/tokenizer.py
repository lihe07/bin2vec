"""Instruction tokenizer for Bin2Vec.

Maps normalised instruction strings (produced by :mod:`bin2vec.preprocess.cfg`)
to integer token IDs suitable for embedding layers.

Vocabulary design
-----------------
* Special tokens: PAD (0), UNK (1), CLS (2), SEP (3)
* Every unique instruction token string seen during vocabulary construction
  gets its own ID starting at SPECIAL_TOKEN_COUNT.

The vocabulary is ISA-specific to avoid collisions between x86_64 and
aarch64 mnemonics. A single :class:`Tokenizer` instance can hold
vocabularies for multiple ISAs.

Persistence
-----------
The vocabulary is serialised as a plain JSON file so it can be reproduced
without any framework dependency::

    tokenizer.save("vocab.json")
    tokenizer = Tokenizer.load("vocab.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Special token IDs
# ---------------------------------------------------------------------------

PAD_ID = 0
UNK_ID = 1
CLS_ID = 2  # prepended to every basic-block sequence
SEP_ID = 3  # appended to every basic-block sequence (optional)
SPECIAL_TOKEN_COUNT = 4

# String representations (useful for display / debugging)
SPECIAL_TOKENS = {
    "<PAD>": PAD_ID,
    "<UNK>": UNK_ID,
    "<CLS>": CLS_ID,
    "<SEP>": SEP_ID,
}


class Tokenizer:
    """Bi-directional vocabulary mapping instruction tokens ↔ integer IDs.

    Parameters
    ----------
    vocab:
        Optional pre-built ``{token_string: token_id}`` mapping.
        If *None*, only the special tokens are present initially.
    max_vocab_size:
        Maximum number of token types (including special tokens).
        New tokens beyond this cap are mapped to UNK.
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        max_vocab_size: int = 65_536,
    ) -> None:
        self.max_vocab_size = max_vocab_size

        if vocab is not None:
            self._token2id: dict[str, int] = dict(vocab)
        else:
            self._token2id = dict(SPECIAL_TOKENS)

        self._id2token: dict[int, str] = {v: k for k, v in self._token2id.items()}

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------

    def add_token(self, token: str) -> int:
        """Add *token* to the vocabulary and return its ID.

        If the token already exists, its existing ID is returned.
        If the vocabulary is full, returns :data:`UNK_ID`.
        """
        if token in self._token2id:
            return self._token2id[token]
        if len(self._token2id) >= self.max_vocab_size:
            return UNK_ID
        new_id = len(self._token2id)
        self._token2id[token] = new_id
        self._id2token[new_id] = token
        return new_id

    def build_from_token_lists(self, token_lists: list[list[str]]) -> None:
        """Populate vocabulary from a corpus of token lists.

        Each inner list represents the instruction tokens for one basic block
        or one function.  Call this before tokenizing the dataset.
        """
        for tokens in token_lists:
            for token in tokens:
                self.add_token(token)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(
        self,
        tokens: list[str],
        add_cls: bool = True,
        add_sep: bool = False,
        max_length: Optional[int] = None,
    ) -> list[int]:
        """Encode a list of token strings to a list of integer IDs.

        Args:
            tokens:     Instruction token strings (one per instruction).
            add_cls:    Prepend ``CLS_ID`` (recommended for Transformer input).
            add_sep:    Append ``SEP_ID``.
            max_length: If set, truncate/pad to exactly this length.
                        Truncation happens *before* special tokens are added.

        Returns:
            List of integer token IDs.
        """
        ids = [self._token2id.get(t, UNK_ID) for t in tokens]

        if add_cls:
            ids = [CLS_ID] + ids
        if add_sep:
            ids = ids + [SEP_ID]

        if max_length is not None:
            if len(ids) >= max_length:
                ids = ids[:max_length]
            else:
                ids = ids + [PAD_ID] * (max_length - len(ids))

        return ids

    def decode(self, ids: list[int]) -> list[str]:
        """Decode a list of integer IDs back to token strings."""
        return [self._id2token.get(i, "<UNK>") for i in ids]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total number of tokens (including special tokens)."""
        return len(self._token2id)

    def __len__(self) -> int:
        return self.vocab_size

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save vocabulary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "max_vocab_size": self.max_vocab_size,
                    "vocab": self._token2id,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        """Load vocabulary from a JSON file saved by :meth:`save`."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            vocab=data["vocab"],
            max_vocab_size=data["max_vocab_size"],
        )

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def encode_batch(
        self,
        token_lists: List[List[str]],
        add_cls: bool = True,
        add_sep: bool = False,
        max_length: Optional[int] = None,
        pad: bool = True,
    ) -> list[list[int]]:
        """Encode multiple token lists.

        If *pad* is True and *max_length* is None, all sequences in the batch
        are padded to the length of the longest sequence.

        Returns:
            List of integer-ID lists, all of equal length if *pad* is True.
        """
        encoded = [
            self.encode(tokens, add_cls=add_cls, add_sep=add_sep)
            for tokens in token_lists
        ]
        if max_length is not None:
            encoded = [(seq + [PAD_ID] * max_length)[:max_length] for seq in encoded]
        elif pad and encoded:
            max_len = max(len(seq) for seq in encoded)
            encoded = [seq + [PAD_ID] * (max_len - len(seq)) for seq in encoded]
        return encoded
