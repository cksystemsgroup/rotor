"""Decode a 32-bit RISC-V instruction word into a tagged record.

M6 covers the full RV64I base instruction set plus byte-addressed
loads and stores:

    OP-IMM      addi  slti  sltiu  xori  ori  andi  slli  srli  srai
    OP-IMM-32   addiw  slliw  srliw  sraiw
    OP          add   sub   sll   slt   sltu  xor   srl   sra   or   and
    OP-32       addw  subw  sllw  srlw  sraw
    BRANCH      beq  bne  blt  bge  bltu  bgeu
    LUI                                                                 (U)
    AUIPC                                                                (U)
    JAL                                                                  (J)
    JALR                                                                 (I)
    LOAD        lb  lh  lw  ld  lbu  lhu  lwu                           (I)
    STORE       sb  sh  sw  sd                                          (S)
    MISC-MEM    fence                                          (no-op)

SYSTEM instructions (ecall, ebreak) remain out of scope; they belong
to Phase F (syscalls).

Compressed (RVC) instructions are not handled — fixtures must compile
with -march=rv64i (no `c`) so the stream is pure 32-bit.

decode() returns None for any opcode outside the supported subset;
callers treat that as an unsupported-instruction error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Opcodes (low 7 bits of the instruction word).
_OP_IMM       = 0b0010011
_OP_IMM_32    = 0b0011011
_OP           = 0b0110011
_OP_32        = 0b0111011
_BRANCH       = 0b1100011
_LUI          = 0b0110111
_AUIPC        = 0b0010111
_JAL          = 0b1101111
_JALR         = 0b1100111
_LOAD         = 0b0000011
_STORE        = 0b0100011
_MISC_MEM     = 0b0001111
_SYSTEM       = 0b1110011


@dataclass(frozen=True)
class Decoded:
    mnem: str
    rd: int                # 0..31
    rs1: int
    rs2: int               # unused for I/U/J types — kept for shape uniformity
    imm: int               # sign-extended where applicable; shamt for shifts
    size: int = 4          # 2 for RVC (compressed), 4 for RV64I/M


# ---------------------------------------------------------------------------
# Top-level dispatch.
# ---------------------------------------------------------------------------

def decode(word: int) -> Optional[Decoded]:
    opcode = word & 0x7F
    decoder = _OPCODE_TABLE.get(opcode)
    if decoder is None:
        return None
    return decoder(word)


# ---------------------------------------------------------------------------
# Per-format decoders.
# ---------------------------------------------------------------------------

def _decode_op_imm(word: int) -> Optional[Decoded]:
    rd, funct3, rs1 = _rd(word), _funct3(word), _rs1(word)
    imm = _sext(_bits(word, 31, 20), 12)

    if funct3 == 0b000:
        return Decoded("addi", rd, rs1, 0, imm)
    if funct3 == 0b010:
        return Decoded("slti", rd, rs1, 0, imm)
    if funct3 == 0b011:
        return Decoded("sltiu", rd, rs1, 0, imm)
    if funct3 == 0b100:
        return Decoded("xori", rd, rs1, 0, imm)
    if funct3 == 0b110:
        return Decoded("ori", rd, rs1, 0, imm)
    if funct3 == 0b111:
        return Decoded("andi", rd, rs1, 0, imm)

    # Shifts: shamt is bits 25:20 (RV64), funct6 is bits 31:26.
    if funct3 == 0b001:
        if _bits(word, 31, 26) == 0:
            return Decoded("slli", rd, rs1, 0, _bits(word, 25, 20))
        return None
    if funct3 == 0b101:
        funct6 = _bits(word, 31, 26)
        shamt = _bits(word, 25, 20)
        if funct6 == 0b000000:
            return Decoded("srli", rd, rs1, 0, shamt)
        if funct6 == 0b010000:
            return Decoded("srai", rd, rs1, 0, shamt)
        return None
    return None


def _decode_op_imm_32(word: int) -> Optional[Decoded]:
    rd, funct3, rs1 = _rd(word), _funct3(word), _rs1(word)
    imm = _sext(_bits(word, 31, 20), 12)

    if funct3 == 0b000:
        return Decoded("addiw", rd, rs1, 0, imm)
    # 32-bit shifts: shamt is bits 24:20 (5 bits); bits 31:25 are funct7.
    if funct3 == 0b001:
        if _bits(word, 31, 25) == 0:
            return Decoded("slliw", rd, rs1, 0, _bits(word, 24, 20))
        return None
    if funct3 == 0b101:
        funct7 = _bits(word, 31, 25)
        shamt = _bits(word, 24, 20)
        if funct7 == 0b0000000:
            return Decoded("srliw", rd, rs1, 0, shamt)
        if funct7 == 0b0100000:
            return Decoded("sraiw", rd, rs1, 0, shamt)
        return None
    return None


def _decode_op(word: int) -> Optional[Decoded]:
    rd, funct3, rs1, rs2 = _rd(word), _funct3(word), _rs1(word), _rs2(word)
    funct7 = _funct7(word)

    if funct7 == 0b0000000:
        m = {
            0b000: "add",
            0b001: "sll",
            0b010: "slt",
            0b011: "sltu",
            0b100: "xor",
            0b101: "srl",
            0b110: "or",
            0b111: "and",
        }.get(funct3)
        return Decoded(m, rd, rs1, rs2, 0) if m else None
    if funct7 == 0b0100000:
        if funct3 == 0b000:
            return Decoded("sub", rd, rs1, rs2, 0)
        if funct3 == 0b101:
            return Decoded("sra", rd, rs1, rs2, 0)
        return None
    if funct7 == 0b0000001:                        # M extension
        m = {
            0b000: "mul",
            0b001: "mulh",
            0b010: "mulhsu",
            0b011: "mulhu",
            0b100: "div",
            0b101: "divu",
            0b110: "rem",
            0b111: "remu",
        }.get(funct3)
        return Decoded(m, rd, rs1, rs2, 0) if m else None
    return None


def _decode_op_32(word: int) -> Optional[Decoded]:
    rd, funct3, rs1, rs2 = _rd(word), _funct3(word), _rs1(word), _rs2(word)
    funct7 = _funct7(word)

    if funct7 == 0b0000000:
        m = {0b000: "addw", 0b001: "sllw", 0b101: "srlw"}.get(funct3)
        return Decoded(m, rd, rs1, rs2, 0) if m else None
    if funct7 == 0b0100000:
        if funct3 == 0b000:
            return Decoded("subw", rd, rs1, rs2, 0)
        if funct3 == 0b101:
            return Decoded("sraw", rd, rs1, rs2, 0)
    if funct7 == 0b0000001:                        # M extension (32-bit variants)
        m = {
            0b000: "mulw",
            0b100: "divw",
            0b101: "divuw",
            0b110: "remw",
            0b111: "remuw",
        }.get(funct3)
        return Decoded(m, rd, rs1, rs2, 0) if m else None
    return None


def _decode_branch(word: int) -> Optional[Decoded]:
    funct3, rs1, rs2 = _funct3(word), _rs1(word), _rs2(word)
    imm = (
        (_bit(word, 31) << 12)
        | (_bit(word, 7) << 11)
        | (_bits(word, 30, 25) << 5)
        | (_bits(word, 11, 8) << 1)
    )
    imm = _sext(imm, 13)
    m = {
        0b000: "beq",
        0b001: "bne",
        0b100: "blt",
        0b101: "bge",
        0b110: "bltu",
        0b111: "bgeu",
    }.get(funct3)
    if m is None:
        return None
    return Decoded(m, 0, rs1, rs2, imm)


def _decode_lui(word: int) -> Optional[Decoded]:
    rd = _rd(word)
    imm = _sext(_bits(word, 31, 12) << 12, 32)         # 32-bit then sext to 64 in lower
    return Decoded("lui", rd, 0, 0, imm)


def _decode_auipc(word: int) -> Optional[Decoded]:
    rd = _rd(word)
    imm = _sext(_bits(word, 31, 12) << 12, 32)
    return Decoded("auipc", rd, 0, 0, imm)


def _decode_jal(word: int) -> Optional[Decoded]:
    rd = _rd(word)
    imm = (
        (_bit(word, 31) << 20)
        | (_bits(word, 19, 12) << 12)
        | (_bit(word, 20) << 11)
        | (_bits(word, 30, 21) << 1)
    )
    imm = _sext(imm, 21)
    return Decoded("jal", rd, 0, 0, imm)


def _decode_jalr(word: int) -> Optional[Decoded]:
    if _funct3(word) != 0b000:
        return None
    rd, rs1 = _rd(word), _rs1(word)
    imm = _sext(_bits(word, 31, 20), 12)
    return Decoded("jalr", rd, rs1, 0, imm)


def _decode_misc_mem(word: int) -> Optional[Decoded]:
    # FENCE / FENCE.I — modeled as a no-op.
    if _funct3(word) in (0b000, 0b001):
        return Decoded("fence", 0, 0, 0, 0)
    return None


def _decode_system(word: int) -> Optional[Decoded]:
    # ECALL / EBREAK only. The I-type immediate field distinguishes
    # them: 0 → ecall, 1 → ebreak. CSR instructions are out of scope
    # (no privileged-mode modeling) and return None to keep the
    # decoder conservative.
    if _funct3(word) != 0b000:
        return None
    if _rd(word) != 0 or _rs1(word) != 0:
        return None
    imm = _bits(word, 31, 20)
    if imm == 0:
        return Decoded("ecall", 0, 0, 0, 0)
    if imm == 1:
        return Decoded("ebreak", 0, 0, 0, 0)
    return None


def _decode_load(word: int) -> Optional[Decoded]:
    rd, funct3, rs1 = _rd(word), _funct3(word), _rs1(word)
    imm = _sext(_bits(word, 31, 20), 12)
    m = {
        0b000: "lb",
        0b001: "lh",
        0b010: "lw",
        0b011: "ld",
        0b100: "lbu",
        0b101: "lhu",
        0b110: "lwu",
    }.get(funct3)
    if m is None:
        return None
    return Decoded(m, rd, rs1, 0, imm)


def _decode_store(word: int) -> Optional[Decoded]:
    funct3, rs1, rs2 = _funct3(word), _rs1(word), _rs2(word)
    imm = (_funct7(word) << 5) | _bits(word, 11, 7)
    imm = _sext(imm, 12)
    m = {
        0b000: "sb",
        0b001: "sh",
        0b010: "sw",
        0b011: "sd",
    }.get(funct3)
    if m is None:
        return None
    return Decoded(m, 0, rs1, rs2, imm)


_OPCODE_TABLE = {
    _OP_IMM:    _decode_op_imm,
    _OP_IMM_32: _decode_op_imm_32,
    _OP:        _decode_op,
    _OP_32:     _decode_op_32,
    _BRANCH:    _decode_branch,
    _LUI:       _decode_lui,
    _AUIPC:     _decode_auipc,
    _JAL:       _decode_jal,
    _JALR:      _decode_jalr,
    _LOAD:      _decode_load,
    _STORE:     _decode_store,
    _MISC_MEM:  _decode_misc_mem,
    _SYSTEM:    _decode_system,
}


# ---------------------------------------------------------------------------
# Bit helpers.
# ---------------------------------------------------------------------------

def _bits(word: int, hi: int, lo: int) -> int:
    return (word >> lo) & ((1 << (hi - lo + 1)) - 1)


def _bit(word: int, n: int) -> int:
    return (word >> n) & 1


def _rd(w: int) -> int:     return _bits(w, 11, 7)
def _funct3(w: int) -> int: return _bits(w, 14, 12)
def _rs1(w: int) -> int:    return _bits(w, 19, 15)
def _rs2(w: int) -> int:    return _bits(w, 24, 20)
def _funct7(w: int) -> int: return _bits(w, 31, 25)


def _sext(value: int, width: int) -> int:
    sign_bit = 1 << (width - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)
