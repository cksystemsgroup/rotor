"""RVC expander coverage.

Each test expands a known 16-bit RVC encoding and cross-checks the
result by running it through rotor's 32-bit decoder. This verifies
two things at once: the RVC bit extraction and the re-encoding into
a 32-bit word that the RV64I decoder accepts and classifies
correctly.

Encodings are hand-written with explicit bit positions so a failure
is easy to trace back to the spec table.
"""

from __future__ import annotations

from rotor.btor2.riscv.decoder import decode
from rotor.btor2.riscv.rvc import expand_rvc


def _expanded(word16: int):
    expanded = expand_rvc(word16)
    assert expanded is not None, f"rvc 0x{word16:04x} should expand"
    d = decode(expanded)
    assert d is not None, f"expanded 0x{expanded:08x} must decode"
    return d


def _pack(funct3: int, bits12_2: int, q: int) -> int:
    """Assemble a 16-bit RVC word from funct3, middle bits, quadrant."""
    return (funct3 << 13) | ((bits12_2 & 0x7FF) << 2) | q


# ---- Quadrant 1 ----

def test_c_nop_expands_to_addi_x0_x0_0() -> None:
    # c.nop: funct3=000, all middle bits zero, q=01.
    d = _expanded(_pack(0b000, 0, 0b01))
    assert d.mnem == "addi" and d.rd == 0 and d.rs1 == 0 and d.imm == 0


def test_c_li_rd_nonzero() -> None:
    # c.li x5, 0 — funct3=010, rd=5, imm=0
    # bits[12:2] = imm[5] | rd | imm[4:0]  = 0 | 00101 | 00000
    word = _pack(0b010, (0 << 10) | (5 << 5) | 0, 0b01)
    d = _expanded(word)
    assert d.mnem == "addi" and d.rd == 5 and d.rs1 == 0 and d.imm == 0


def test_c_addi_immediate_sign_extended() -> None:
    # c.addi x5, -1  — funct3=000, rd=rs1=5, imm=-1 (6-bit: 111111)
    # bits[12:2] = 1 | 00101 | 11111
    word = _pack(0b000, (1 << 10) | (5 << 5) | 0b11111, 0b01)
    d = _expanded(word)
    assert d.mnem == "addi" and d.rd == 5 and d.rs1 == 5 and d.imm == -1


def test_c_addiw_is_64bit_variant() -> None:
    # c.addiw x5, 1 — funct3=001, rd=rs1=5, imm=1
    word = _pack(0b001, (0 << 10) | (5 << 5) | 1, 0b01)
    d = _expanded(word)
    assert d.mnem == "addiw" and d.rd == 5 and d.rs1 == 5 and d.imm == 1


def test_c_j_is_jal_x0() -> None:
    # c.j +8 — funct3=101, q=01, imm encoding: bits[5:3]=100 → imm[3:1]=100
    # → +8. Encoding 0xA021 per hand-decode of the CJ format table.
    d = _expanded(0xA021)
    assert d.mnem == "jal" and d.rd == 0 and d.imm == 8


def test_c_beqz_is_beq_rs1p_x0() -> None:
    # c.beqz x8, +4 — funct3=110, rs1'=0 (→x8). Encoding 0xC011 per
    # CB-format: bits[6:2]=00100 → imm[2:1]=10 → +4.
    d = _expanded(0xC011)
    assert d.mnem == "beq" and d.rs1 == 8 and d.rs2 == 0 and d.imm == 4


# ---- Quadrant 2 ----

def test_c_jr_is_jalr_x0() -> None:
    # c.jr x1 — funct4=1000, rs1=1, rs2=0, q=10
    # bits[15:12] = 1000; bits[11:7] = 00001; bits[6:2] = 00000
    word = (0b1000 << 12) | (1 << 7) | 0b10
    d = _expanded(word)
    assert d.mnem == "jalr" and d.rd == 0 and d.rs1 == 1 and d.imm == 0


def test_c_jalr_writes_x1_return_address() -> None:
    # c.jalr x5 — funct4=1001, rs1=5, rs2=0
    word = (0b1001 << 12) | (5 << 7) | 0b10
    d = _expanded(word)
    assert d.mnem == "jalr" and d.rd == 1 and d.rs1 == 5 and d.imm == 0


def test_c_mv_is_add_from_x0() -> None:
    # c.mv x5, x10 — funct4=1000, rd=5, rs2=10
    word = (0b1000 << 12) | (5 << 7) | (10 << 2) | 0b10
    d = _expanded(word)
    assert d.mnem == "add" and d.rd == 5 and d.rs1 == 0 and d.rs2 == 10


def test_c_add_is_add_rd_rs2() -> None:
    # c.add x5, x10 — funct4=1001, rd=5, rs2=10
    word = (0b1001 << 12) | (5 << 7) | (10 << 2) | 0b10
    d = _expanded(word)
    assert d.mnem == "add" and d.rd == 5 and d.rs1 == 5 and d.rs2 == 10


def test_c_ebreak() -> None:
    # funct4=1001, rd=rs1=0, rs2=0
    word = (0b1001 << 12) | 0b10
    d = _expanded(word)
    assert d.mnem == "ebreak"


def test_c_slli_is_slli() -> None:
    # c.slli x5, 4 — funct3=000, rd=rs1=5, shamt=4, q=10
    # bits[12] = shamt[5] = 0; bits[11:7] = rd = 5; bits[6:2] = shamt[4:0] = 4
    word = (0b000 << 13) | (0 << 12) | (5 << 7) | (4 << 2) | 0b10
    d = _expanded(word)
    assert d.mnem == "slli" and d.rd == 5 and d.rs1 == 5 and d.imm == 4


# ---- illegal / unsupported ----

def test_all_zero_first_halfword_is_illegal() -> None:
    assert expand_rvc(0x0000) is None


def test_uncompressed_low_bits_returns_none() -> None:
    assert expand_rvc(0x0013) is None                # q == 0b11
