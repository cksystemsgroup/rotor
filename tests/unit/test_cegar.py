"""Unit tests for CEGAR's internal register-read classification.

The classifier decides which registers a decoded instruction reads,
driving the refinement strategy (unhavoc every register the concrete
replay path actually read). Correctness matters because
under-approximation here would cause CEGAR to loop forever on a
spurious CEX; over-approximation only costs extra solver time.
"""

from __future__ import annotations

from rotor.btor2.riscv.decoder import decode
from rotor.cegar import _regs_read_by


def _decode(word: int):
    d = decode(word)
    assert d is not None, f"unable to decode 0x{word:08x}"
    return d


def test_r_type_add_reads_rs1_and_rs2() -> None:
    # add x5, x10, x11 — 0x00b50[2b3]=0b0110011 OPC, funct3=0, funct7=0
    # manually: rd=5, rs1=10, rs2=11. Build: funct7|rs2|rs1|funct3|rd|opcode
    word = (0 << 25) | (11 << 20) | (10 << 15) | (0 << 12) | (5 << 7) | 0b0110011
    d = _decode(word)
    assert d.mnem == "add"
    assert _regs_read_by(d) == {10, 11}


def test_i_type_addi_reads_only_rs1() -> None:
    # addi x5, x10, 42 — rs2 field holds the imm bits, not a register.
    word = (42 << 20) | (10 << 15) | (0 << 12) | (5 << 7) | 0b0010011
    d = _decode(word)
    assert d.mnem == "addi"
    assert _regs_read_by(d) == {10}


def test_branch_bne_reads_rs1_and_rs2() -> None:
    # bne x10, x11, +4 — B-type. Imm=+4 split across the word.
    # imm[12|10:5|4:1|11] = 0|000000|0010|0
    # Layout: imm12,imm10:5,rs2,rs1,funct3,imm4:1,imm11,opcode
    word = (0 << 31) | (0 << 25) | (11 << 20) | (10 << 15) | (1 << 12) | (2 << 8) | (0 << 7) | 0b1100011
    d = _decode(word)
    assert d.mnem == "bne"
    assert _regs_read_by(d) == {10, 11}


def test_store_sw_reads_rs1_and_rs2() -> None:
    # sw x11, 0(x10) — S-type, rs1=10 (base), rs2=11 (value to store).
    word = (0 << 25) | (11 << 20) | (10 << 15) | (2 << 12) | (0 << 7) | 0b0100011
    d = _decode(word)
    assert d.mnem == "sw"
    assert _regs_read_by(d) == {10, 11}


def test_load_lw_reads_only_rs1() -> None:
    # lw x5, 0(x10) — I-type.
    word = (0 << 20) | (10 << 15) | (2 << 12) | (5 << 7) | 0b0000011
    d = _decode(word)
    assert d.mnem == "lw"
    assert _regs_read_by(d) == {10}


def test_lui_reads_nothing() -> None:
    # lui x5, 0x12345 — U-type, no register reads.
    word = (0x12345 << 12) | (5 << 7) | 0b0110111
    d = _decode(word)
    assert d.mnem == "lui"
    assert _regs_read_by(d) == set()


def test_jal_reads_nothing() -> None:
    # jal x1, +4 — J-type, rd=1, no register reads.
    # Encoded imm=+4: imm[20|10:1|11|19:12] = 0|0000000010|0|00000000
    word = (0 << 31) | (2 << 21) | (0 << 20) | (0 << 12) | (1 << 7) | 0b1101111
    d = _decode(word)
    assert d.mnem == "jal"
    assert _regs_read_by(d) == set()


def test_jalr_reads_only_rs1() -> None:
    # jalr x0, 0(x1) — aka `ret`. rs1=1 is the target register; rs2 unused.
    word = (0 << 20) | (1 << 15) | (0 << 12) | (0 << 7) | 0b1100111
    d = _decode(word)
    assert d.mnem == "jalr"
    assert _regs_read_by(d) == {1}


def test_x0_reads_are_filtered() -> None:
    # add x5, x0, x10 — reads x0 and x10. x0 is constant zero; never
    # included in the refinement set since unhavoc'ing it is a no-op.
    word = (0 << 25) | (10 << 20) | (0 << 15) | (0 << 12) | (5 << 7) | 0b0110011
    d = _decode(word)
    assert d.mnem == "add"
    assert _regs_read_by(d) == {10}
