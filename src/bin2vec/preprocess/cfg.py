"""CFG (Control Flow Graph) extraction from raw function bytes using Capstone.

Each basic block is a node; edges represent control flow between blocks.

Data structures
---------------
BasicBlock:
    start_offset  int        – byte offset of first instruction (relative to function start)
    instructions  list[str]  – normalised instruction tokens, one string per instruction
                               format: "<mnemonic> <op0>,<op1>,..."
                               immediate values are replaced with symbolic tokens (IMM)
                               address-like operands are replaced with BB_REF or SYM_REF

CFGGraph:
    blocks   list[BasicBlock]   – ordered by start_offset
    edges    list[(int,int)]    – (src_block_index, dst_block_index) in block list
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

import capstone  # type: ignore

# ---------------------------------------------------------------------------
# ISA → Capstone arch/mode mapping
# ---------------------------------------------------------------------------

_ISA_MAP: dict[str, tuple[int, int]] = {
    "x86_64": (capstone.CS_ARCH_X86, capstone.CS_MODE_64),
    "aarch64": (capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM),
    # MIPS little-endian 32-bit
    "mipsel": (
        capstone.CS_ARCH_MIPS,
        capstone.CS_MODE_MIPS32 | capstone.CS_MODE_LITTLE_ENDIAN,
    ),
}

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

# Hex immediate pattern (e.g. 0x1a4, -0x8, 1234)
_HEX_RE = re.compile(r"-?0x[0-9a-fA-F]+")
_DEC_RE = re.compile(r"\b-?\d+\b")


def _normalise_operand(op: str, branch_targets: set[int]) -> str:
    """Replace concrete values with abstract tokens to improve invariance.

    * Memory references with numeric immediates → keep structure, replace number
    * Branch targets that are inside the function → BB_REF
    * Branch targets outside → SYM_REF
    * Other immediates → IMM
    """
    # First check if this looks like a pure address operand (branch target)
    stripped = op.strip()
    hex_match = _HEX_RE.fullmatch(stripped)
    dec_match = _DEC_RE.fullmatch(stripped)
    if hex_match or dec_match:
        try:
            addr = int(stripped, 16) if "x" in stripped else int(stripped)
            if addr in branch_targets:
                return "BB_REF"
            return "SYM_REF"
        except ValueError:
            pass

    # Replace hex immediates inside compound operands (e.g. "[rsp + 0x18]")
    result = _HEX_RE.sub("IMM", op)
    result = _DEC_RE.sub("IMM", result)
    return result


def _normalise_insn(mnemonic: str, op_str: str, branch_targets: set[int]) -> str:
    """Return a single normalised token string for one instruction."""
    if not op_str:
        return mnemonic
    parts = [p.strip() for p in op_str.split(",")]
    norm_parts = [_normalise_operand(p, branch_targets) for p in parts]
    return f"{mnemonic} {','.join(norm_parts)}"


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class BasicBlock:
    start_offset: int
    instructions: List[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        n = len(self.instructions)
        return f"<BasicBlock offset={self.start_offset} insns={n}>"


@dataclass
class CFGGraph:
    blocks: List[BasicBlock] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<CFGGraph blocks={len(self.blocks)} edges={len(self.edges)}>"


# ---------------------------------------------------------------------------
# CFG builder
# ---------------------------------------------------------------------------


def extract_cfg(raw_bytes: bytes, isa: str, base_addr: int = 0) -> CFGGraph:
    """Disassemble *raw_bytes* and build a basic-block CFG.

    Args:
        raw_bytes: Raw machine code bytes of a single function.
        isa:       ISA string matching those used in the dataset
                   (``"x86_64"``, ``"aarch64"``, ``"mipsel"``).
        base_addr: Virtual address at which the function starts (used only
                   for resolving intra-function branch targets; the graph
                   itself uses offsets from 0).

    Returns:
        A :class:`CFGGraph` with normalised instruction strings.

    Raises:
        ValueError: If *isa* is not supported.
    """
    if isa not in _ISA_MAP:
        raise ValueError(f"Unsupported ISA '{isa}'. Supported: {list(_ISA_MAP)}")

    arch, mode = _ISA_MAP[isa]
    cs = capstone.Cs(arch, mode)
    cs.detail = True  # needed to inspect groups / operands

    # ------------------------------------------------------------------ #
    # Pass 1: linear disassembly — collect all instructions and their      #
    #         addresses so we know which branch targets are intra-function. #
    # ------------------------------------------------------------------ #
    insns: list[tuple[int, str, str]] = []  # (addr, mnemonic, op_str)
    addr_set: set[int] = set()

    for insn in cs.disasm(raw_bytes, base_addr):
        insns.append((insn.address, insn.mnemonic, insn.op_str))
        addr_set.add(insn.address)

    if not insns:
        # Nothing disassembled — return a single empty block
        return CFGGraph(blocks=[BasicBlock(start_offset=0)], edges=[])

    # ------------------------------------------------------------------ #
    # Pass 2: identify basic-block leaders                                 #
    #   • first instruction is always a leader                             #
    #   • target of any branch is a leader                                 #
    #   • instruction after any branch/call is a leader                    #
    # ------------------------------------------------------------------ #
    leaders: set[int] = {insns[0][0]}

    # Branch/call group IDs per architecture
    _BRANCH_GROUPS = {
        capstone.CS_ARCH_X86: {
            capstone.x86.X86_GRP_JUMP,
            capstone.x86.X86_GRP_CALL,
            capstone.x86.X86_GRP_RET,
        },
        capstone.CS_ARCH_ARM64: {
            capstone.arm64.ARM64_GRP_JUMP,
            capstone.arm64.ARM64_GRP_CALL,
            capstone.arm64.ARM64_GRP_RET,
        },
        capstone.CS_ARCH_MIPS: {
            capstone.mips.MIPS_GRP_JUMP,
            capstone.mips.MIPS_GRP_CALL,
            capstone.mips.MIPS_GRP_RET,
        },
    }
    branch_groups = _BRANCH_GROUPS.get(arch, set())

    # We need a second Cs pass with detail to check groups
    cs2 = capstone.Cs(arch, mode)
    cs2.detail = True

    branch_targets: set[int] = set()  # concrete addresses referenced by branches

    for insn in cs2.disasm(raw_bytes, base_addr):
        groups = set(insn.groups)
        is_branch_or_call = bool(groups & branch_groups)

        if is_branch_or_call:
            # Try to extract the branch target (first operand if it's an imm)
            try:
                if arch == capstone.CS_ARCH_X86 and insn.operands:
                    op = insn.operands[0]
                    if op.type == capstone.x86.X86_OP_IMM:
                        branch_targets.add(op.imm)
                        if op.imm in addr_set:
                            leaders.add(op.imm)
                elif arch == capstone.CS_ARCH_ARM64 and insn.operands:
                    op = insn.operands[0]
                    if op.type == capstone.arm64.ARM64_OP_IMM:
                        branch_targets.add(op.imm)
                        if op.imm in addr_set:
                            leaders.add(op.imm)
                elif arch == capstone.CS_ARCH_MIPS and insn.operands:
                    for op in insn.operands:
                        if op.type == capstone.mips.MIPS_OP_IMM:
                            branch_targets.add(op.imm)
                            if op.imm in addr_set:
                                leaders.add(op.imm)
            except Exception:
                pass

            # Instruction immediately following a branch/call is also a leader
            next_addr = insn.address + insn.size
            if next_addr in addr_set:
                leaders.add(next_addr)

    # ------------------------------------------------------------------ #
    # Pass 3: partition instructions into basic blocks                     #
    # ------------------------------------------------------------------ #
    sorted_leaders = sorted(leaders)
    leader_to_block_idx: dict[int, int] = {
        addr: i for i, addr in enumerate(sorted_leaders)
    }

    blocks: list[BasicBlock] = [
        BasicBlock(start_offset=addr - base_addr) for addr in sorted_leaders
    ]

    # Assign each instruction to its block
    current_block_idx = 0
    for addr, mnemonic, op_str in insns:
        if addr in leader_to_block_idx:
            current_block_idx = leader_to_block_idx[addr]
        token = _normalise_insn(mnemonic, op_str, branch_targets)
        blocks[current_block_idx].instructions.append(token)

    # ------------------------------------------------------------------ #
    # Pass 4: build edges                                                  #
    # ------------------------------------------------------------------ #
    edges: list[tuple[int, int]] = []

    cs3 = capstone.Cs(arch, mode)
    cs3.detail = True

    for insn in cs3.disasm(raw_bytes, base_addr):
        groups = set(insn.groups)
        is_branch_or_call = bool(groups & branch_groups)
        if not is_branch_or_call:
            continue

        src_block_idx = leader_to_block_idx.get(insn.address)
        # Walk back: find which block this insn belongs to
        if src_block_idx is None:
            for ldr in reversed(sorted_leaders):
                if ldr <= insn.address:
                    src_block_idx = leader_to_block_idx[ldr]
                    break

        if src_block_idx is None:
            continue

        # Explicit branch target
        try:
            target_addr: int | None = None
            if arch == capstone.CS_ARCH_X86 and insn.operands:
                op = insn.operands[0]
                if op.type == capstone.x86.X86_OP_IMM:
                    target_addr = op.imm
            elif arch == capstone.CS_ARCH_ARM64 and insn.operands:
                op = insn.operands[0]
                if op.type == capstone.arm64.ARM64_OP_IMM:
                    target_addr = op.imm
            elif arch == capstone.CS_ARCH_MIPS and insn.operands:
                for op in insn.operands:
                    if op.type == capstone.mips.MIPS_OP_IMM:
                        target_addr = op.imm
                        break

            if target_addr is not None and target_addr in leader_to_block_idx:
                dst_block_idx = leader_to_block_idx[target_addr]
                edge = (src_block_idx, dst_block_idx)
                if edge not in edges:
                    edges.append(edge)
        except Exception:
            pass

        # Fall-through edge (for conditional branches)
        fall_addr = insn.address + insn.size
        if fall_addr in leader_to_block_idx:
            dst_block_idx = leader_to_block_idx[fall_addr]
            edge = (src_block_idx, dst_block_idx)
            if edge not in edges:
                edges.append(edge)

    return CFGGraph(blocks=blocks, edges=edges)
