"""RVC expansion — 16-bit compressed instructions → 32-bit RV64 equivalents.

Per RISC-V Volume I, Chapter 16 (C extension). Each compressed
instruction has a specific 32-bit equivalent; rotor decodes the
16-bit word, extracts the fields per its format (CR / CI / CSS /
CIW / CL / CS / CA / CB / CJ), and re-assembles the equivalent
32-bit RV64I/M encoding. Downstream, `rotor.btor2.riscv.decoder`
treats that 32-bit word the same way it would an uncompressed
instruction — the only trace that the instruction was compressed
is `Instruction.size == 2` / `Decoded.size == 2`, which the
lowering pipeline threads through to produce correct `pc + 2`
fall-through arithmetic (see rotor/btor2/riscv/isa.py::_fall).

Coverage in this first cut focuses on the compressed instructions
GCC/clang at `-O2 -march=rv64gc` emit for typical leaf and
shallow-call code: RV64G minus floating-point. Specifically:

  Quadrant 0 (op bits 1:0 == 00):
    c.addi4spn, c.lw, c.ld, c.sw, c.sd
  Quadrant 1 (op bits 1:0 == 01):
    c.nop, c.addi, c.addiw, c.li,
    c.addi16sp, c.lui,
    c.srli, c.srai, c.andi,
    c.sub, c.xor, c.or, c.and, c.subw, c.addw,
    c.j, c.beqz, c.bnez
  Quadrant 2 (op bits 1:0 == 10):
    c.slli, c.lwsp, c.ldsp,
    c.jr, c.mv, c.ebreak, c.jalr, c.add,
    c.swsp, c.sdsp

Floating-point RVC (c.flw/c.fld/c.fsw/c.fsd/c.fswsp/c.fsdsp/c.flwsp/
c.fldsp) returns None since rotor has no FP model. Reserved
encodings (all-zero first halfword, etc.) also return None.

Most RVC immediates are non-contiguous bit patterns; helpers below
extract each one per the spec tables rather than through a generic
shuffle, which keeps the mapping explicit and easy to audit.
"""

from __future__ import annotations

from typing import Optional


# --- bit helpers -------------------------------------------------------------

def _bit(w: int, i: int) -> int:
    return (w >> i) & 1


def _bits(w: int, hi: int, lo: int) -> int:
    return (w >> lo) & ((1 << (hi - lo + 1)) - 1)


def _sext(value: int, width: int) -> int:
    sign = 1 << (width - 1)
    return value - (1 << width) if value & sign else value


# RVC "short" register fields (3-bit): index is rs1'/rs2'/rd' → x8+reg.
def _rp(reg3: int) -> int:
    return 8 + (reg3 & 0b111)


# --- 32-bit encoding helpers -------------------------------------------------

def _enc_r(funct7: int, rs2: int, rs1: int, funct3: int, rd: int, opcode: int) -> int:
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _enc_i(imm12: int, rs1: int, funct3: int, rd: int, opcode: int) -> int:
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _enc_s(imm12: int, rs2: int, rs1: int, funct3: int, opcode: int) -> int:
    imm12 &= 0xFFF
    return (((imm12 >> 5) & 0x7F) << 25) | (rs2 << 20) | (rs1 << 15) | \
           (funct3 << 12) | ((imm12 & 0x1F) << 7) | opcode


def _enc_b(imm13: int, rs2: int, rs1: int, funct3: int) -> int:
    # imm13 bits: 12|10:5|4:1|11  (bit 0 is always 0 and not encoded).
    imm = imm13 & 0x1FFE
    b12 = (imm >> 12) & 1
    b11 = (imm >> 11) & 1
    b10_5 = (imm >> 5) & 0x3F
    b4_1 = (imm >> 1) & 0xF
    return (b12 << 31) | (b10_5 << 25) | (rs2 << 20) | (rs1 << 15) | \
           (funct3 << 12) | (b4_1 << 8) | (b11 << 7) | 0b1100011


def _enc_u(imm32: int, rd: int, opcode: int) -> int:
    return (imm32 & 0xFFFFF000) | (rd << 7) | opcode


def _enc_j(imm21: int, rd: int) -> int:
    # imm21 bits: 20|10:1|11|19:12
    imm = imm21 & 0x1FFFFE
    b20 = (imm >> 20) & 1
    b19_12 = (imm >> 12) & 0xFF
    b11 = (imm >> 11) & 1
    b10_1 = (imm >> 1) & 0x3FF
    return (b20 << 31) | (b10_1 << 21) | (b11 << 20) | (b19_12 << 12) | \
           (rd << 7) | 0b1101111


# RV64I/M opcodes.
_OP_IMM    = 0b0010011
_OP_IMM_32 = 0b0011011
_OP        = 0b0110011
_OP_32     = 0b0111011
_LUI       = 0b0110111
_JALR      = 0b1100111
_LOAD      = 0b0000011
_STORE     = 0b0100011
_SYSTEM    = 0b1110011


# --- public entry point -----------------------------------------------------

def expand_rvc(word16: int) -> Optional[int]:
    """Expand a 16-bit compressed instruction into its 32-bit equivalent.

    Returns None for unsupported or reserved encodings; callers treat
    None the same as an `Optional[Decoded]` miss — unsupported
    instruction.
    """
    word16 &= 0xFFFF
    q = word16 & 0b11
    if q == 0b11:                                  # not compressed
        return None
    if word16 == 0:                                # illegal encoding
        return None
    funct3 = _bits(word16, 15, 13)
    if q == 0b00:
        return _q0(word16, funct3)
    if q == 0b01:
        return _q1(word16, funct3)
    return _q2(word16, funct3)                     # q == 0b10


# --- Quadrant 0 -------------------------------------------------------------

def _q0(w: int, funct3: int) -> Optional[int]:
    rdp  = _rp(_bits(w,  4,  2))
    rs1p = _rp(_bits(w,  9,  7))
    # CIW uses rd'; CL uses rd'; CS uses rs2'; all same field position.
    if funct3 == 0b000:                            # c.addi4spn
        # imm[5:4|9:6|2|3] = bits 12:11 | 10:7 | 6 | 5, *4 to get byte offset
        imm = ((_bits(w, 12, 11) << 4)
               | (_bits(w, 10,  7) << 6)
               | (_bit (w,  6    ) << 2)
               | (_bit (w,  5    ) << 3))
        if imm == 0:                               # reserved
            return None
        return _enc_i(imm, 2, 0b000, rdp, _OP_IMM)   # addi rd', sp, imm
    if funct3 == 0b010:                            # c.lw
        imm = ((_bit (w,  5    ) << 6)
               | (_bits(w, 12, 10) << 3)
               | (_bit (w,  6    ) << 2))
        return _enc_i(imm, rs1p, 0b010, rdp, _LOAD)
    if funct3 == 0b011:                            # c.ld (RV64)
        imm = ((_bits(w,  6,  5) << 6)
               | (_bits(w, 12, 10) << 3))
        return _enc_i(imm, rs1p, 0b011, rdp, _LOAD)
    if funct3 == 0b110:                            # c.sw
        imm = ((_bit (w,  5    ) << 6)
               | (_bits(w, 12, 10) << 3)
               | (_bit (w,  6    ) << 2))
        return _enc_s(imm, rdp, rs1p, 0b010, _STORE)
    if funct3 == 0b111:                            # c.sd (RV64)
        imm = ((_bits(w,  6,  5) << 6)
               | (_bits(w, 12, 10) << 3))
        return _enc_s(imm, rdp, rs1p, 0b011, _STORE)
    return None                                    # FP variants not supported


# --- Quadrant 1 -------------------------------------------------------------

def _q1(w: int, funct3: int) -> Optional[int]:
    rd_rs1 = _bits(w, 11, 7)                       # full 5-bit for CI-format
    if funct3 == 0b000:                            # c.nop / c.addi
        imm = _sext((_bit(w, 12) << 5) | _bits(w, 6, 2), 6)
        return _enc_i(imm & 0xFFF, rd_rs1, 0b000, rd_rs1, _OP_IMM)
    if funct3 == 0b001:                            # c.addiw (RV64; hint when rd=0)
        if rd_rs1 == 0:
            return None                            # reserved
        imm = _sext((_bit(w, 12) << 5) | _bits(w, 6, 2), 6)
        return _enc_i(imm & 0xFFF, rd_rs1, 0b000, rd_rs1, _OP_IMM_32)
    if funct3 == 0b010:                            # c.li
        imm = _sext((_bit(w, 12) << 5) | _bits(w, 6, 2), 6)
        return _enc_i(imm & 0xFFF, 0, 0b000, rd_rs1, _OP_IMM)
    if funct3 == 0b011:
        if rd_rs1 == 2:                            # c.addi16sp
            imm = _sext(
                (_bit(w, 12) << 9)
                | (_bit(w,  6) << 4)
                | (_bit(w,  5) << 6)
                | (_bits(w, 4, 3) << 7)
                | (_bit(w,  2) << 5),
                10,
            )
            if imm == 0:
                return None
            return _enc_i(imm & 0xFFF, 2, 0b000, 2, _OP_IMM)
        # c.lui (hint when rd=0; reserved when rd=2 already filtered)
        if rd_rs1 == 0:
            return None
        imm20 = _sext((_bit(w, 12) << 5) | _bits(w, 6, 2), 6) << 12
        if imm20 == 0:
            return None
        return _enc_u(imm20 & 0xFFFFF000, rd_rs1, _LUI)
    if funct3 == 0b100:
        return _q1_alu(w)
    if funct3 == 0b101:                            # c.j
        imm = _sext(
            (_bit(w, 12) << 11)
            | (_bit(w, 11) << 4)
            | (_bits(w, 10, 9) << 8)
            | (_bit(w,  8) << 10)
            | (_bit(w,  7) << 6)
            | (_bit(w,  6) << 7)
            | (_bits(w, 5, 3) << 1)
            | (_bit(w,  2) << 5),
            12,
        )
        return _enc_j(imm & 0x1FFFFE, 0)           # jal x0, offset
    if funct3 == 0b110 or funct3 == 0b111:         # c.beqz / c.bnez
        rs1p = _rp(_bits(w, 9, 7))
        imm = _sext(
            (_bit(w, 12) << 8)
            | (_bits(w, 11, 10) << 3)
            | (_bits(w, 6, 5) << 6)
            | (_bits(w, 4, 3) << 1)
            | (_bit(w,  2) << 5),
            9,
        )
        f3 = 0b000 if funct3 == 0b110 else 0b001   # beq vs bne
        return _enc_b(imm & 0x1FFE, 0, rs1p, f3)
    return None


def _q1_alu(w: int) -> Optional[int]:
    rs1p = _rp(_bits(w, 9, 7))
    sub  = _bits(w, 11, 10)
    imm = (_bit(w, 12) << 5) | _bits(w, 6, 2)
    if sub == 0b00:                                # c.srli
        return _enc_i((0 << 6) | imm, rs1p, 0b101, rs1p, _OP_IMM)
    if sub == 0b01:                                # c.srai
        return _enc_i((0b010000 << 6) | imm, rs1p, 0b101, rs1p, _OP_IMM)
    if sub == 0b10:                                # c.andi
        imm_s = _sext(imm, 6) & 0xFFF
        return _enc_i(imm_s, rs1p, 0b111, rs1p, _OP_IMM)
    # sub == 0b11: c.sub / c.xor / c.or / c.and / c.subw / c.addw
    rs2p = _rp(_bits(w, 4, 2))
    bit12 = _bit(w, 12)
    sub2 = _bits(w, 6, 5)
    if bit12 == 0:
        m = {0b00: "sub", 0b01: "xor", 0b10: "or", 0b11: "and"}[sub2]
        funct3_map = {"sub": 0b000, "xor": 0b100, "or": 0b110, "and": 0b111}[m]
        funct7_map = 0b0100000 if m == "sub" else 0b0000000
        return _enc_r(funct7_map, rs2p, rs1p, funct3_map, rs1p, _OP)
    # bit12 == 1: RV64 OP-32 variants
    if sub2 == 0b00:                               # c.subw
        return _enc_r(0b0100000, rs2p, rs1p, 0b000, rs1p, _OP_32)
    if sub2 == 0b01:                               # c.addw
        return _enc_r(0b0000000, rs2p, rs1p, 0b000, rs1p, _OP_32)
    return None                                    # reserved


# --- Quadrant 2 -------------------------------------------------------------

def _q2(w: int, funct3: int) -> Optional[int]:
    rd_rs1 = _bits(w, 11, 7)
    rs2    = _bits(w,  6, 2)
    if funct3 == 0b000:                            # c.slli
        if rd_rs1 == 0:
            return None
        shamt = (_bit(w, 12) << 5) | rs2
        return _enc_i((0 << 6) | shamt, rd_rs1, 0b001, rd_rs1, _OP_IMM)
    if funct3 == 0b010:                            # c.lwsp
        if rd_rs1 == 0:
            return None
        imm = ((_bits(w, 3, 2) << 6)
               | (_bit (w, 12) << 5)
               | (_bits(w,  6, 4) << 2))
        return _enc_i(imm, 2, 0b010, rd_rs1, _LOAD)
    if funct3 == 0b011:                            # c.ldsp (RV64)
        if rd_rs1 == 0:
            return None
        imm = ((_bits(w, 4, 2) << 6)
               | (_bit (w, 12) << 5)
               | (_bits(w,  6, 5) << 3))
        return _enc_i(imm, 2, 0b011, rd_rs1, _LOAD)
    if funct3 == 0b100:
        bit12 = _bit(w, 12)
        if bit12 == 0:
            if rs2 == 0:
                if rd_rs1 == 0:
                    return None                    # reserved
                return _enc_i(0, rd_rs1, 0b000, 0, _JALR)  # c.jr: jalr x0, rs1, 0
            # c.mv: add rd, x0, rs2
            if rd_rs1 == 0:
                return None                        # hint / reserved
            return _enc_r(0, rs2, 0, 0b000, rd_rs1, _OP)
        # bit12 == 1
        if rd_rs1 == 0 and rs2 == 0:               # c.ebreak
            return _enc_i(1, 0, 0b000, 0, _SYSTEM)
        if rs2 == 0:                               # c.jalr: jalr x1, rs1, 0
            return _enc_i(0, rd_rs1, 0b000, 1, _JALR)
        # c.add: add rd, rd, rs2
        return _enc_r(0, rs2, rd_rs1, 0b000, rd_rs1, _OP)
    if funct3 == 0b110:                            # c.swsp
        imm = ((_bits(w, 8, 7) << 6)
               | (_bits(w, 12, 9) << 2))
        return _enc_s(imm, rs2, 2, 0b010, _STORE)
    if funct3 == 0b111:                            # c.sdsp (RV64)
        imm = ((_bits(w, 9, 7) << 6)
               | (_bits(w, 12, 10) << 3))
        return _enc_s(imm, rs2, 2, 0b011, _STORE)
    return None                                    # FP / reserved
