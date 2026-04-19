"""Instruction disassembler: Decoded -> assembler string.

Recognizes the common pseudo-instructions GCC emits so traces read
naturally (ret, li, mv, neg, snez, bgtz, bltz). Used by trace rendering
and by the `rotor disasm` CLI subcommand.
"""

from __future__ import annotations

from rotor.btor2.riscv.decoder import Decoded

ABI = (
    "zero", "ra", "sp", "gp", "tp",
    "t0", "t1", "t2",
    "s0", "s1",
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
    "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
    "t3", "t4", "t5", "t6",
)


def disasm(d: Decoded) -> str:
    r = ABI
    if d.mnem == "jalr" and d.rd == 0 and d.rs1 == 1 and d.imm == 0:
        return "ret"
    if d.mnem == "addi" and d.rs1 == 0:
        return f"li {r[d.rd]}, {d.imm}"
    if d.mnem == "addi" and d.imm == 0:
        return f"mv {r[d.rd]}, {r[d.rs1]}"
    if d.mnem == "sub" and d.rs1 == 0:
        return f"neg {r[d.rd]}, {r[d.rs2]}"
    if d.mnem == "sltu" and d.rs1 == 0:
        return f"snez {r[d.rd]}, {r[d.rs2]}"
    if d.mnem == "blt" and d.rs1 == 0:
        return f"bgtz {r[d.rs2]}, {d.imm:+d}"
    if d.mnem == "blt" and d.rs2 == 0:
        return f"bltz {r[d.rs1]}, {d.imm:+d}"
    if d.mnem == "addi":
        return f"{d.mnem} {r[d.rd]}, {r[d.rs1]}, {d.imm}"
    if d.mnem in ("addw", "sub", "sltu"):
        return f"{d.mnem} {r[d.rd]}, {r[d.rs1]}, {r[d.rs2]}"
    if d.mnem == "blt":
        return f"blt {r[d.rs1]}, {r[d.rs2]}, {d.imm:+d}"
    if d.mnem == "jalr":
        return f"jalr {r[d.rd]}, {d.imm}({r[d.rs1]})"
    return d.mnem                          # pragma: no cover
