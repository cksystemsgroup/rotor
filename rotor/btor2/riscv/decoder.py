"""Pure-Python bit extractors that build BTOR2 slice nodes.

Given a 32-bit instruction node, each helper returns a Node for the
corresponding RISC-V field (opcode, funct3, rs1, etc.) or a sign/zero
extended immediate. The extractors reuse the builder's structural sharing,
so repeated calls with identical arguments return the same node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from rotor.btor2 import Node
    from rotor.btor2.builder import RISCVMachineBuilder


def field(b: "RISCVMachineBuilder", instr: "Node", hi: int, lo: int, name: str) -> "Node":
    """Slice bits [hi:lo] of ``instr`` into a fresh BTOR2 node."""
    width = hi - lo + 1
    sort = b.bitvec(width)
    return b.slice(sort, instr, hi, lo, name)


def opcode(b: "RISCVMachineBuilder", instr: "Node") -> "Node":
    return field(b, instr, 6, 0, "opcode")


def rd(b: "RISCVMachineBuilder", instr: "Node") -> "Node":
    return field(b, instr, 11, 7, "rd")


def funct3(b: "RISCVMachineBuilder", instr: "Node") -> "Node":
    return field(b, instr, 14, 12, "funct3")


def rs1(b: "RISCVMachineBuilder", instr: "Node") -> "Node":
    return field(b, instr, 19, 15, "rs1")


def rs2(b: "RISCVMachineBuilder", instr: "Node") -> "Node":
    return field(b, instr, 24, 20, "rs2")


def funct7(b: "RISCVMachineBuilder", instr: "Node") -> "Node":
    return field(b, instr, 31, 25, "funct7")


# ──────────────────────────────────────────────────────────────────────────
# Immediates (all sign-extended to machine word width)
# ──────────────────────────────────────────────────────────────────────────


def imm_i(b: "RISCVMachineBuilder", instr: "Node") -> "Node":
    """I-type immediate, bits[31:20], sign-extended."""
    raw = field(b, instr, 31, 20, "imm_i_raw")
    return _sext_to_word(b, raw, "imm_i")


def imm_s(b: "RISCVMachineBuilder", instr: "Node") -> "Node":
    """S-type immediate, bits[31:25]|bits[11:7], sign-extended."""
    hi = field(b, instr, 31, 25, "imm_s_hi")
    lo = field(b, instr, 11, 7, "imm_s_lo")
    raw = b.concat(hi, lo, "imm_s_raw")
    return _sext_to_word(b, raw, "imm_s")


def imm_b(b: "RISCVMachineBuilder", instr: "Node") -> "Node":
    """B-type immediate: {imm[12], imm[10:5], imm[4:1], 0} sign-extended."""
    sign = field(b, instr, 31, 31, "imm_b_31")
    mid_hi = field(b, instr, 30, 25, "imm_b_30_25")
    mid_lo = field(b, instr, 11, 8, "imm_b_11_8")
    bit11 = field(b, instr, 7, 7, "imm_b_7")
    zero = b.zero(b.bitvec(1), "b_lsb0")
    # {sign(1), bit11(1), mid_hi(6), mid_lo(4), 0(1)} = 13 bits
    concat1 = b.concat(sign, bit11, "b_hi2")
    concat2 = b.concat(concat1, mid_hi, "b_hi8")
    concat3 = b.concat(concat2, mid_lo, "b_hi12")
    raw = b.concat(concat3, zero, "imm_b_raw")
    return _sext_to_word(b, raw, "imm_b")


def imm_u(b: "RISCVMachineBuilder", instr: "Node") -> "Node":
    """U-type immediate: bits[31:12]<<12, zero-extended/sign-extended."""
    hi = field(b, instr, 31, 12, "imm_u_hi")
    z12 = b.zero(b.bitvec(12), "u_lsb12")
    raw = b.concat(hi, z12, "imm_u_raw")  # 32 bits
    return _sext_to_word(b, raw, "imm_u")


def imm_j(b: "RISCVMachineBuilder", instr: "Node") -> "Node":
    """J-type immediate: {imm[20], imm[10:1], imm[11], imm[19:12], 0}."""
    sign = field(b, instr, 31, 31, "imm_j_31")
    mid_hi = field(b, instr, 19, 12, "imm_j_19_12")
    bit11 = field(b, instr, 20, 20, "imm_j_20")
    low = field(b, instr, 30, 21, "imm_j_30_21")
    zero = b.zero(b.bitvec(1), "j_lsb0")
    concat1 = b.concat(sign, mid_hi, "j_hi9")
    concat2 = b.concat(concat1, bit11, "j_hi10")
    concat3 = b.concat(concat2, low, "j_hi20")
    raw = b.concat(concat3, zero, "imm_j_raw")  # 21 bits
    return _sext_to_word(b, raw, "imm_j")


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _sext_to_word(b: "RISCVMachineBuilder", raw: "Node", name: str) -> "Node":
    assert b.SID_MACHINE_WORD is not None
    ext_bits = (b.SID_MACHINE_WORD.width or 0) - (raw.sort.width or 0)
    if ext_bits <= 0:
        return raw
    return b.sext(b.SID_MACHINE_WORD, raw, ext_bits, name)


def is_opcode(b: "RISCVMachineBuilder", instr: "Node", value: int, name: str) -> "Node":
    """Predicate: opcode(instr) == value."""
    op = opcode(b, instr)
    k = b.constd(b.bitvec(7), value, f"opcode_{value:02x}")
    return b.eq(op, k, name)


def is_funct3(b: "RISCVMachineBuilder", instr: "Node", value: int, name: str) -> "Node":
    f3 = funct3(b, instr)
    k = b.constd(b.bitvec(3), value, f"f3_{value}")
    return b.eq(f3, k, name)


def is_funct7(b: "RISCVMachineBuilder", instr: "Node", value: int, name: str) -> "Node":
    f7 = funct7(b, instr)
    k = b.constd(b.bitvec(7), value, f"f7_{value:02x}")
    return b.eq(f7, k, name)
