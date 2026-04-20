"""Witness simulator coverage for the RV64I instruction expansion.

For each new mnemonic class, we hand-construct one or more
instructions and step them concretely with the witness simulator,
verifying the resulting register / pc state matches RISC-V semantics.

The decoder test asserts (word -> Decoded) is correct; this test
asserts (Decoded -> register/pc effect) matches the spec.
"""

from __future__ import annotations

from rotor.btor2.riscv.decoder import Decoded
from rotor.witness import _step

XLEN = 64
MASK = (1 << XLEN) - 1


def _regs(**named: int) -> list[int]:
    r = [0] * 32
    for name, value in named.items():
        idx = int(name[1:])
        r[idx] = value & MASK
    return r


# --------------------------------- I-type --------------------------------- #

def test_addi_sext_imm() -> None:
    r = _regs(x10=0)
    pc = _step(Decoded("addi", 11, 10, 0, -3), 0x100, r)
    assert pc == 0x104
    assert r[11] == ((-3) & MASK)


def test_andi_ori_xori() -> None:
    r = _regs(x10=0xF0)
    _step(Decoded("andi", 11, 10, 0, 0x0F), 0, r); assert r[11] == 0
    _step(Decoded("ori",  12, 10, 0, 0x0F), 0, r); assert r[12] == 0xFF
    _step(Decoded("xori", 13, 10, 0, 0x0F), 0, r); assert r[13] == 0xFF


def test_slti_sltiu() -> None:
    r = _regs(x10=(-5) & MASK)
    _step(Decoded("slti",  11, 10, 0, 0), 0, r); assert r[11] == 1
    _step(Decoded("sltiu", 12, 10, 0, 0), 0, r); assert r[12] == 0  # -5 unsigned >> 0


def test_slli_srli_srai() -> None:
    r = _regs(x10=0xDEADBEEF)
    _step(Decoded("slli", 11, 10, 0, 4), 0, r); assert r[11] == (0xDEADBEEF << 4) & MASK
    _step(Decoded("srli", 12, 10, 0, 4), 0, r); assert r[12] == 0xDEADBEEF >> 4
    r2 = _regs(x10=(-1) & MASK)
    _step(Decoded("srai", 13, 10, 0, 8), 0, r2); assert r2[13] == MASK   # arithmetic of all-ones is all-ones


# --------------------------- I-type 32-bit (W) ---------------------------- #

def test_addiw_sign_extends_to_64() -> None:
    r = _regs(x10=0x7FFFFFFF)
    _step(Decoded("addiw", 11, 10, 0, 1), 0, r)
    # Result is 0x80000000 in 32-bit, sign-extended to 64.
    assert r[11] == 0xFFFFFFFF80000000


def test_sllw_clears_high_bits_then_sext() -> None:
    r = _regs(x10=1)
    _step(Decoded("slliw", 11, 10, 0, 31), 0, r)
    assert r[11] == 0xFFFFFFFF80000000


# ------------------------------- R-type ----------------------------------- #

def test_add_sub_64() -> None:
    r = _regs(x10=10, x11=3)
    _step(Decoded("add", 12, 10, 11, 0), 0, r); assert r[12] == 13
    _step(Decoded("sub", 13, 10, 11, 0), 0, r); assert r[13] == 7


def test_logical_64() -> None:
    r = _regs(x10=0xF0F0, x11=0x0FF0)
    _step(Decoded("and", 12, 10, 11, 0), 0, r); assert r[12] == 0x00F0
    _step(Decoded("or",  13, 10, 11, 0), 0, r); assert r[13] == 0xFFF0
    _step(Decoded("xor", 14, 10, 11, 0), 0, r); assert r[14] == 0xFF00


def test_slt_sltu() -> None:
    r = _regs(x10=(-1) & MASK, x11=1)
    _step(Decoded("slt",  12, 10, 11, 0), 0, r); assert r[12] == 1   # -1 < 1 signed
    _step(Decoded("sltu", 13, 10, 11, 0), 0, r); assert r[13] == 0   # max-uint > 1


def test_sll_srl_sra_use_low_six_bits_of_rs2() -> None:
    r = _regs(x10=1, x11=64 + 3)                                     # shift by 3 effective
    _step(Decoded("sll", 12, 10, 11, 0), 0, r); assert r[12] == 8


# ------------------------------- W R-type --------------------------------- #

def test_addw_subw_sext() -> None:
    r = _regs(x10=0xFFFFFFFF, x11=1)                                 # 32-bit -1 + 1 = 0
    _step(Decoded("addw", 12, 10, 11, 0), 0, r); assert r[12] == 0
    _step(Decoded("subw", 13, 11, 10, 0), 0, r); assert r[13] == 2   # 1 - (-1) = 2 in 32-bit


# ------------------------------- branches --------------------------------- #

def test_branches_taken_and_not() -> None:
    r = _regs(x10=5, x11=5)
    assert _step(Decoded("beq", 0, 10, 11, 16), 0x100, r) == 0x110
    assert _step(Decoded("bne", 0, 10, 11, 16), 0x100, r) == 0x104

    r2 = _regs(x10=(-3) & MASK, x11=2)
    assert _step(Decoded("blt",  0, 10, 11, 16), 0x100, r2) == 0x110  # -3 < 2 signed
    assert _step(Decoded("bge",  0, 10, 11, 16), 0x100, r2) == 0x104  # -3 < 2 signed
    assert _step(Decoded("bltu", 0, 10, 11, 16), 0x100, r2) == 0x104  # -3 unsigned > 2
    assert _step(Decoded("bgeu", 0, 10, 11, 16), 0x100, r2) == 0x110


# ----------------------------- U / J / JALR -------------------------------- #

def test_lui_auipc_jal_jalr() -> None:
    # lui rd, 0x1000 -> rd = 0x1000_0000 (sign-extended to 64)
    r = _regs()
    _step(Decoded("lui", 11, 0, 0, 0x10000000), 0, r)
    assert r[11] == 0x10000000

    # auipc rd, 0 at pc=0x100 -> rd = pc
    r2 = _regs()
    _step(Decoded("auipc", 11, 0, 0, 0), 0x100, r2)
    assert r2[11] == 0x100

    # jal: rd = pc+4, pc = pc+imm
    r3 = _regs()
    nxt = _step(Decoded("jal", 1, 0, 0, 0x20), 0x100, r3)
    assert r3[1] == 0x104 and nxt == 0x120

    # jalr: rd = pc+4, pc = (rs1 + imm) & ~1
    r4 = _regs(x5=0x301)
    nxt = _step(Decoded("jalr", 1, 5, 0, 0), 0x100, r4)
    assert r4[1] == 0x104 and nxt == 0x300                           # low bit cleared


def test_fence_is_noop() -> None:
    r = _regs(x10=0xABCD)
    nxt = _step(Decoded("fence", 0, 0, 0, 0), 0x100, r)
    assert nxt == 0x104 and r[10] == 0xABCD


def test_x0_writes_are_dropped() -> None:
    r = _regs()
    _step(Decoded("addi", 0, 0, 0, 99), 0, r)
    assert r[0] == 0                                                 # x0 stays zero
