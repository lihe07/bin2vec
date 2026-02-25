"""Abstract interface for future preprocessing steps."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Preprocessor(ABC):
    """Base class for function preprocessing (disassembly, CFG, tokenization)."""

    @abstractmethod
    def process(self, raw_bytes: bytes, isa: str) -> dict:
        """Process raw function bytes into a structured representation.

        Args:
            raw_bytes: Raw bytes of the function.
            isa: Target ISA (e.g. "x86_64", "aarch64", "mipsel").

        Returns:
            Dictionary with preprocessed data (format depends on implementation).
        """
        ...


class IdentityPreprocessor(Preprocessor):
    """Passthrough preprocessor for testing — returns raw bytes unchanged."""

    def process(self, raw_bytes: bytes, isa: str) -> dict:
        return {
            "raw_bytes": raw_bytes,
            "isa": isa,
        }
