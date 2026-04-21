"""Register liveness analysis for goal-directed slicing.

A register R is *live* for reachability on a function iff its value
can affect whether the `bad` expression (`pc == target_pc`) ever
holds. Because rotor's `build_reach` emits a PC-dispatch ITE over
*every* instruction in the function regardless of reachability, the
only way a register value affects `pc` is by feeding a branch or
jalr comparison — directly (as rs1/rs2) or transitively (via a
write whose result later flows into a branch read).

This analysis is flow-insensitive: it considers all instructions in
the function together, without a CFG. That is a safe over-
approximation — the actual live set is no larger than this analysis
reports — and it is cheap enough to run on every build_reach call.

The resulting `dead_registers` set is the complement of live (minus
x0, which is a hard-wired constant, not a register that can be
havoc'd). `SsaEmitter` passes this set as `havoc_regs` to
`build_reach`, so dead registers become per-cycle BTOR2 inputs
rather than stateful reg-file entries — collapsing the PDR state
space the unbounded engines have to reason over.
"""

from __future__ import annotations

from typing import Iterable

from rotor.binary import Function, RISCVBinary
from rotor.btor2.riscv.decoder import Decoded, decode

NREGS = 32

_BRANCHES = frozenset({"beq", "bne", "blt", "bge", "bltu", "bgeu"})

# Mnemonic classification — mirrors rotor/cegar.py. Kept independent
# so liveness doesn't couple to the CEGAR loop; any decoder mnemonic
# must be classified in both places.
_READS_RS1_AND_RS2 = _BRANCHES | frozenset({
    "add", "sub", "and", "or", "xor", "slt", "sltu", "sll", "srl", "sra",
    "addw", "subw", "sllw", "srlw", "sraw",
    "mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu",
    "mulw", "divw", "divuw", "remw", "remuw",
    "sb", "sh", "sw", "sd",
})
_READS_RS1_ONLY = frozenset({
    "addi", "slti", "sltiu", "xori", "ori", "andi",
    "slli", "srli", "srai",
    "addiw", "slliw", "srliw", "sraiw",
    "jalr",
    "lb", "lh", "lw", "ld", "lbu", "lhu", "lwu",
})
_WRITES_RD = _READS_RS1_AND_RS2 - _BRANCHES - frozenset({"sb", "sh", "sw", "sd"}) | \
             _READS_RS1_ONLY | frozenset({"lui", "auipc", "jal"})


def _reads_of(d: Decoded) -> frozenset[int]:
    if d.mnem in _READS_RS1_AND_RS2:
        return frozenset({d.rs1, d.rs2}) - {0}
    if d.mnem in _READS_RS1_ONLY:
        return frozenset({d.rs1}) - {0}
    return frozenset()


def _write_of(d: Decoded) -> int:
    if d.mnem in _WRITES_RD and d.rd != 0:
        return d.rd
    return 0


def live_registers(binary: RISCVBinary, function: Function) -> frozenset[int]:
    """Return the set of register indices whose values can affect
    reachability of any PC within the function.

    Always contains at least the registers read by the function's
    branches and jalr instructions, plus transitively the registers
    that can flow into those reads via writes in the function.
    """
    initial: set[int] = set()
    # write_reads[r] = list of reads performed by instructions that write r.
    write_reads: dict[int, list[frozenset[int]]] = {}

    for inst in binary.instructions(function):
        d = decode(inst.word)
        if d is None:
            raise ValueError(f"unable to decode 0x{inst.word:08x} at pc 0x{inst.pc:x}")
        reads = _reads_of(d)
        if d.mnem in _BRANCHES or d.mnem == "jalr":
            initial |= reads
        w = _write_of(d)
        if w != 0:
            write_reads.setdefault(w, []).append(reads)

    live = set(initial)
    changed = True
    while changed:
        changed = False
        for r, read_sets in write_reads.items():
            if r in live:
                for reads in read_sets:
                    if not reads.issubset(live):
                        live |= reads
                        changed = True
    return frozenset(live)


def dead_registers(binary: RISCVBinary, function: Function) -> frozenset[int]:
    """Return the register indices that can be havoc'd without
    affecting reachability verdicts on the function.

    This is the set `SsaEmitter` passes as `havoc_regs` to
    `build_reach`. `x0` is excluded (hard-wired constant).
    """
    live = live_registers(binary, function)
    return frozenset(range(1, NREGS)) - live
