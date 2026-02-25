"""DWARF-based ELF function extraction via pyelftools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from elftools.elf.elffile import ELFFile
from elftools.dwarf.die import DIE

from bin2vec.utils.logging import get_logger

log = get_logger("elf_parser")


@dataclass
class RawFunction:
    """A single extracted function with raw bytes and metadata."""
    name: str
    source_file: str
    address: int
    size: int
    raw_bytes: bytes


class ElfParser:
    """Parse ELF files using DWARF debug info to extract functions."""

    def __init__(self, elf_path: Path) -> None:
        self.elf_path = elf_path

    def extract_functions(self) -> list[RawFunction]:
        """Extract all functions with DWARF info from the ELF file."""
        functions = []

        try:
            with open(self.elf_path, "rb") as f:
                elf = ELFFile(f)

                if not elf.has_dwarf_info():
                    log.debug("No DWARF info in %s", self.elf_path)
                    return []

                dwarf = elf.get_dwarf_info()

                for cu in dwarf.iter_CUs():
                    cu_die = cu.get_top_DIE()
                    comp_dir = self._get_attr_str(cu_die, "DW_AT_comp_dir", "")
                    cu_name = self._get_attr_str(cu_die, "DW_AT_name", "")

                    for die in cu.iter_DIEs():
                        if die.tag != "DW_TAG_subprogram":
                            continue

                        func = self._parse_subprogram(die, elf, cu_name, comp_dir)
                        if func is not None:
                            functions.append(func)

        except Exception as e:
            log.warning("Failed to parse %s: %s", self.elf_path, e)

        return functions

    def _parse_subprogram(
        self,
        die: DIE,
        elf: ELFFile,
        cu_name: str,
        comp_dir: str,
    ) -> RawFunction | None:
        """Parse a DW_TAG_subprogram DIE into a RawFunction."""
        # Skip declarations (no body)
        if "DW_AT_declaration" in die.attributes:
            return None

        # Skip inlined functions (no standalone bytes)
        if "DW_AT_inline" in die.attributes:
            inline_val = die.attributes["DW_AT_inline"].value
            if inline_val in (1, 3):  # DW_INL_inlined or DW_INL_declared_inlined
                return None

        # Get function name
        name = self._get_attr_str(die, "DW_AT_name")
        if not name:
            # Try linkage name
            name = self._get_attr_str(die, "DW_AT_linkage_name")
        if not name:
            return None

        # Get address
        if "DW_AT_low_pc" not in die.attributes:
            return None
        low_pc = die.attributes["DW_AT_low_pc"].value

        # Get size: DW_AT_high_pc can be absolute address or offset (DWARF 4 vs 5)
        if "DW_AT_high_pc" not in die.attributes:
            return None

        high_pc_attr = die.attributes["DW_AT_high_pc"]
        if high_pc_attr.form in ("DW_FORM_addr",):
            # Absolute address
            size = high_pc_attr.value - low_pc
        else:
            # Offset from low_pc (DWARF 4 constant forms)
            size = high_pc_attr.value

        if size <= 0:
            return None

        # Get source file
        source = self._get_attr_str(die, "DW_AT_decl_file")
        if not source:
            source = cu_name

        # Read raw bytes from the ELF section
        raw_bytes = self._read_bytes(elf, low_pc, size)
        if raw_bytes is None:
            return None

        return RawFunction(
            name=name,
            source_file=source,
            address=low_pc,
            size=size,
            raw_bytes=raw_bytes,
        )

    def _read_bytes(self, elf: ELFFile, address: int, size: int) -> bytes | None:
        """Read raw bytes from the ELF file at the given virtual address."""
        for section in elf.iter_sections():
            sec_addr = section["sh_addr"]
            sec_size = section["sh_size"]
            if sec_addr <= address < sec_addr + sec_size:
                offset = address - sec_addr
                data = section.data()
                if offset + size <= len(data):
                    return data[offset : offset + size]
        return None

    @staticmethod
    def _get_attr_str(die: DIE, attr_name: str, default: str = "") -> str:
        """Get a string attribute from a DIE, handling bytes."""
        if attr_name not in die.attributes:
            return default
        val = die.attributes[attr_name].value
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="replace")
        return str(val)
