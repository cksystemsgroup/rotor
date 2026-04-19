"""Decode a 32-bit RISC-V instruction word into a tagged record.

M1 supports a minimal subset of RV64I sufficient for the add2.elf fixture:

    addi    I-type   OP-IMM   funct3=000
    addw    R-type   OP-32    funct3=000  funct7=0000000
    sub     R-type   OP       funct3=000  funct7=0100000
    sltu    R-type   OP       funct3=011  funct7=0000000
    blt     B-type   BRANCH   funct3=100
    jalr    I-type   JALR     funct3=000

decode() returns None for any opcode outside the subset; callers treat
that as an unsupported-instruction error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Decoded:
    mnem: str          # "addi", "addw", "sub", "sltu", "blt", "jalr"
    rd: int            # 0..31
    rs1: int
    rs2: int           # unused for I-type; kept for shape uniformity
    imm: int           # sign-extended


def decode(word: int) -> Optional[Decoded]:
    opcode = word & 0x7F
    rd = (word >> 7) & 0x1F
    funct3 = (word >> 12) & 0x7
    rs1 = (word >> 15) & 0x1F
    rs2 = (word >> 20) & 0x1F
    funct7 = (word >> 25) & 0x7F

    if opcode == 0b0010011 and funct3 == 0b000:
        # addi rd, rs1, imm
        return Decoded("addi", rd, rs1, 0, _sext(_bits(word, 31, 20), 12))

    if opcode == 0b0111011 and funct3 == 0b000 and funct7 == 0b0000000:
        # addw rd, rs1, rs2
        return Decoded("addw", rd, rs1, rs2, 0)

    if opcode == 0b0110011 and funct3 == 0b000 and funct7 == 0b0100000:
        # sub rd, rs1, rs2
        return Decoded("sub", rd, rs1, rs2, 0)

    if opcode == 0b0110011 and funct3 == 0b011 and funct7 == 0b0000000:
        # sltu rd, rs1, rs2
        return Decoded("sltu", rd, rs1, rs2, 0)

    if opcode == 0b1100011 and funct3 == 0b100:
        # blt rs1, rs2, offset
        imm = (
            (_bit(word, 31) << 12)
            | (_bit(word, 7) << 11)
            | (_bits(word, 30, 25) << 5)
            | (_bits(word, 11, 8) << 1)
        )
        return Decoded("blt", 0, rs1, rs2, _sext(imm, 13))

    if opcode == 0b1100111 and funct3 == 0b000:
        # jalr rd, offset(rs1)
        return Decoded("jalr", rd, rs1, 0, _sext(_bits(word, 31, 20), 12))

    return None


def _bits(word: int, hi: int, lo: int) -> int:
    return (word >> lo) & ((1 << (hi - lo + 1)) - 1)


def _bit(word: int, n: int) -> int:
    return (word >> n) & 1


def _sext(value: int, width: int) -> int:
    sign_bit = 1 << (width - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)
