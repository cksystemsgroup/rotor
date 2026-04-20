"""Decoder coverage for RV64I loads and stores (M6)."""

import pytest

from rotor.btor2.riscv.decoder import decode

# Hand-assembled instruction words validated against llvm-objdump.
# Each row: (word, mnem, rd, rs1, rs2, imm)
LOADS = [
    (0x00052503, "lw",   10, 10,  0, 0),      # lw a0, 0(a0)
    (0x00452503, "lw",   10, 10,  0, 4),      # lw a0, 4(a0)
    (0xFFC52503, "lw",   10, 10,  0, -4),     # lw a0, -4(a0)   sign-extended
    (0x00050503, "lb",   10, 10,  0, 0),
    (0x00051503, "lh",   10, 10,  0, 0),
    (0x00053503, "ld",   10, 10,  0, 0),
    (0x00054503, "lbu",  10, 10,  0, 0),
    (0x00055503, "lhu",  10, 10,  0, 0),
    (0x00056503, "lwu",  10, 10,  0, 0),
]

STORES = [
    (0x00B52023, "sw",    0, 10, 11, 0),      # sw a1, 0(a0)
    (0x00B52223, "sw",    0, 10, 11, 4),      # sw a1, 4(a0)
    (0xFEB52E23, "sw",    0, 10, 11, -4),     # sw a1, -4(a0)   sign-extended
    (0x00B50023, "sb",    0, 10, 11, 0),
    (0x00B51023, "sh",    0, 10, 11, 0),
    (0x00B53023, "sd",    0, 10, 11, 0),
]


@pytest.mark.parametrize("word,mnem,rd,rs1,rs2,imm", LOADS)
def test_decode_load(word, mnem, rd, rs1, rs2, imm) -> None:
    d = decode(word)
    assert d is not None, f"decoder rejected load 0x{word:08x}"
    assert (d.mnem, d.rd, d.rs1, d.rs2, d.imm) == (mnem, rd, rs1, rs2, imm)


@pytest.mark.parametrize("word,mnem,rd,rs1,rs2,imm", STORES)
def test_decode_store(word, mnem, rd, rs1, rs2, imm) -> None:
    d = decode(word)
    assert d is not None, f"decoder rejected store 0x{word:08x}"
    assert (d.mnem, d.rd, d.rs1, d.rs2, d.imm) == (mnem, rd, rs1, rs2, imm)


def test_load_invalid_funct3() -> None:
    # LOAD with funct3 = 111 is reserved / not a supported width.
    word = (0b111 << 12) | 0b0000011
    assert decode(word) is None


def test_store_invalid_funct3() -> None:
    # STORE with funct3 = 100..111 is reserved.
    for f3 in (0b100, 0b101, 0b110, 0b111):
        word = (f3 << 12) | 0b0100011
        assert decode(word) is None, f"expected None for store funct3={f3:03b}"
