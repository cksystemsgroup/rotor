"""Instruction disassembler: Decoded -> assembler string.

Recognizes the common pseudo-instructions GCC emits so traces read
naturally (ret, li, mv, neg, snez, bgtz, bltz, jr, j, nop, not, ...).
Used by trace rendering and by the `rotor disasm` CLI subcommand.
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

# Instruction-class groups for compact rendering.
_RR_64 = {"add", "sub", "and", "or", "xor", "slt", "sltu", "sll", "srl", "sra"}
_RR_32 = {"addw", "subw", "sllw", "srlw", "sraw"}
_RI_64 = {"addi", "andi", "ori", "xori", "slti", "sltiu", "slli", "srli", "srai"}
_RI_32 = {"addiw", "slliw", "srliw", "sraiw"}
_BR    = {"beq", "bne", "blt", "bge", "bltu", "bgeu"}
_LOAD  = {"lb", "lh", "lw", "ld", "lbu", "lhu", "lwu"}
_STORE = {"sb", "sh", "sw", "sd"}


def disasm(d: Decoded) -> str:
    r = ABI

    # ---- pseudo-instructions ------------------------------------------------
    if d.mnem == "addi" and d.rd == 0 and d.rs1 == 0 and d.imm == 0:
        return "nop"
    if d.mnem == "jalr" and d.rd == 0 and d.rs1 == 1 and d.imm == 0:
        return "ret"
    if d.mnem == "jalr" and d.rd == 0 and d.imm == 0:
        return f"jr {r[d.rs1]}"
    if d.mnem == "jal" and d.rd == 0:
        return f"j {d.imm:+d}"
    if d.mnem == "addi" and d.rs1 == 0:
        return f"li {r[d.rd]}, {d.imm}"
    if d.mnem == "addi" and d.imm == 0:
        return f"mv {r[d.rd]}, {r[d.rs1]}"
    if d.mnem == "addiw" and d.imm == 0:
        return f"sext.w {r[d.rd]}, {r[d.rs1]}"
    if d.mnem == "xori" and d.imm == -1:
        return f"not {r[d.rd]}, {r[d.rs1]}"
    if d.mnem == "sub" and d.rs1 == 0:
        return f"neg {r[d.rd]}, {r[d.rs2]}"
    if d.mnem == "subw" and d.rs1 == 0:
        return f"negw {r[d.rd]}, {r[d.rs2]}"
    if d.mnem == "sltu" and d.rs1 == 0:
        return f"snez {r[d.rd]}, {r[d.rs2]}"
    if d.mnem == "slt" and d.rs2 == 0:
        return f"sltz {r[d.rd]}, {r[d.rs1]}"
    if d.mnem == "slt" and d.rs1 == 0:
        return f"sgtz {r[d.rd]}, {r[d.rs2]}"
    if d.mnem == "blt" and d.rs1 == 0:
        return f"bgtz {r[d.rs2]}, {d.imm:+d}"
    if d.mnem == "blt" and d.rs2 == 0:
        return f"bltz {r[d.rs1]}, {d.imm:+d}"
    if d.mnem == "bge" and d.rs1 == 0:
        return f"blez {r[d.rs2]}, {d.imm:+d}"
    if d.mnem == "bge" and d.rs2 == 0:
        return f"bgez {r[d.rs1]}, {d.imm:+d}"
    if d.mnem == "beq" and d.rs2 == 0:
        return f"beqz {r[d.rs1]}, {d.imm:+d}"
    if d.mnem == "bne" and d.rs2 == 0:
        return f"bnez {r[d.rs1]}, {d.imm:+d}"
    if d.mnem == "fence":
        return "fence"

    # ---- canonical forms ----------------------------------------------------
    if d.mnem in _RR_64 or d.mnem in _RR_32:
        return f"{d.mnem} {r[d.rd]}, {r[d.rs1]}, {r[d.rs2]}"
    if d.mnem in _RI_64 or d.mnem in _RI_32:
        return f"{d.mnem} {r[d.rd]}, {r[d.rs1]}, {d.imm}"
    if d.mnem in _BR:
        return f"{d.mnem} {r[d.rs1]}, {r[d.rs2]}, {d.imm:+d}"
    if d.mnem == "lui":
        return f"lui {r[d.rd]}, {d.imm >> 12:#x}"
    if d.mnem == "auipc":
        return f"auipc {r[d.rd]}, {d.imm >> 12:#x}"
    if d.mnem == "jal":
        return f"jal {r[d.rd]}, {d.imm:+d}"
    if d.mnem == "jalr":
        return f"jalr {r[d.rd]}, {d.imm}({r[d.rs1]})"
    if d.mnem in _LOAD:
        return f"{d.mnem} {r[d.rd]}, {d.imm}({r[d.rs1]})"
    if d.mnem in _STORE:
        return f"{d.mnem} {r[d.rs2]}, {d.imm}({r[d.rs1]})"

    return d.mnem                                  # pragma: no cover
