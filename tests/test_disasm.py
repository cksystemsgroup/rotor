"""Tests for the RISC-V disassembler."""

from __future__ import annotations

from rotor.riscv import disassemble, reg_name


# ──────────────────────────────────────────────────────────────────────────
# Helpers to encode instructions
# ──────────────────────────────────────────────────────────────────────────


def r_type(opcode: int, rd: int, f3: int, rs1: int, rs2: int, f7: int) -> int:
    return (
        opcode | (rd << 7) | (f3 << 12) | (rs1 << 15) | (rs2 << 20) | (f7 << 25)
    )


def i_type(opcode: int, rd: int, f3: int, rs1: int, imm: int) -> int:
    return opcode | (rd << 7) | (f3 << 12) | (rs1 << 15) | ((imm & 0xFFF) << 20)


def s_type(opcode: int, f3: int, rs1: int, rs2: int, imm: int) -> int:
    imm &= 0xFFF
    return (
        opcode
        | ((imm & 0x1F) << 7)
        | (f3 << 12)
        | (rs1 << 15)
        | (rs2 << 20)
        | (((imm >> 5) & 0x7F) << 25)
    )


def b_type(opcode: int, f3: int, rs1: int, rs2: int, imm: int) -> int:
    imm &= 0x1FFF  # 13-bit, bit 0 is implicit 0
    return (
        opcode
        | (((imm >> 11) & 1) << 7)
        | (((imm >> 1) & 0xF) << 8)
        | (f3 << 12)
        | (rs1 << 15)
        | (rs2 << 20)
        | (((imm >> 5) & 0x3F) << 25)
        | (((imm >> 12) & 1) << 31)
    )


def u_type(opcode: int, rd: int, imm20: int) -> int:
    return opcode | (rd << 7) | ((imm20 & 0xFFFFF) << 12)


def j_type(opcode: int, rd: int, imm: int) -> int:
    imm &= 0x1FFFFF  # 21-bit, bit 0 implicit 0
    return (
        opcode
        | (rd << 7)
        | (((imm >> 12) & 0xFF) << 12)
        | (((imm >> 11) & 1) << 20)
        | (((imm >> 1) & 0x3FF) << 21)
        | (((imm >> 20) & 1) << 31)
    )


# ──────────────────────────────────────────────────────────────────────────
# ABI names
# ──────────────────────────────────────────────────────────────────────────


def test_reg_names_spot_check() -> None:
    assert reg_name(0) == "zero"
    assert reg_name(1) == "ra"
    assert reg_name(2) == "sp"
    assert reg_name(8) == "s0"
    assert reg_name(10) == "a0"
    assert reg_name(15) == "a5"
    assert reg_name(31) == "t6"


# ──────────────────────────────────────────────────────────────────────────
# R-type
# ──────────────────────────────────────────────────────────────────────────


def test_add() -> None:
    assert disassemble(r_type(0x33, 10, 0, 11, 12, 0)) == "add a0, a1, a2"


def test_sub() -> None:
    assert disassemble(r_type(0x33, 10, 0, 11, 12, 0x20)) == "sub a0, a1, a2"


def test_mul() -> None:
    assert disassemble(r_type(0x33, 10, 0, 11, 12, 0x01)) == "mul a0, a1, a2"


def test_div_rem() -> None:
    assert disassemble(r_type(0x33, 10, 4, 11, 12, 0x01)) == "div a0, a1, a2"
    assert disassemble(r_type(0x33, 10, 6, 11, 12, 0x01)) == "rem a0, a1, a2"


def test_addw() -> None:
    assert disassemble(r_type(0x3B, 10, 0, 11, 12, 0)) == "addw a0, a1, a2"


def test_subw_mulw() -> None:
    assert disassemble(r_type(0x3B, 10, 0, 11, 12, 0x20)) == "subw a0, a1, a2"
    assert disassemble(r_type(0x3B, 10, 0, 11, 12, 0x01)) == "mulw a0, a1, a2"


# ──────────────────────────────────────────────────────────────────────────
# I-type
# ──────────────────────────────────────────────────────────────────────────


def test_addi() -> None:
    assert disassemble(i_type(0x13, 10, 0, 11, 42)) == "addi a0, a1, 42"


def test_addi_negative() -> None:
    assert disassemble(i_type(0x13, 10, 0, 11, -1)) == "addi a0, a1, -1"


def test_addi_zero_becomes_mv() -> None:
    assert disassemble(i_type(0x13, 10, 0, 11, 0)) == "mv a0, a1"


def test_addi_from_zero_becomes_li() -> None:
    assert disassemble(i_type(0x13, 10, 0, 0, 0)) == "li a0, 0"


def test_slli() -> None:
    assert disassemble(i_type(0x13, 10, 1, 11, 4)) == "slli a0, a1, 4"


def test_srai_via_f7() -> None:
    word = 0x13 | (10 << 7) | (5 << 12) | (11 << 15) | ((0x20 << 5 | 3) << 20)
    assert disassemble(word) == "srai a0, a1, 3"


def test_addiw() -> None:
    assert disassemble(i_type(0x1B, 10, 0, 11, 7)) == "addiw a0, a1, 7"


def test_sext_w() -> None:
    # addiw rd, rs1, 0 is the sext.w pseudo.
    assert disassemble(i_type(0x1B, 10, 0, 11, 0)) == "sext.w a0, a1"


# ──────────────────────────────────────────────────────────────────────────
# Loads / stores
# ──────────────────────────────────────────────────────────────────────────


def test_lw() -> None:
    assert disassemble(i_type(0x03, 10, 2, 2, 8)) == "lw a0, 8(sp)"


def test_ld() -> None:
    assert disassemble(i_type(0x03, 10, 3, 2, -16)) == "ld a0, -16(sp)"


def test_sd() -> None:
    assert disassemble(s_type(0x23, 3, 2, 10, 24)) == "sd a0, 24(sp)"


def test_sw() -> None:
    assert disassemble(s_type(0x23, 2, 2, 11, -4)) == "sw a1, -4(sp)"


# ──────────────────────────────────────────────────────────────────────────
# Control flow
# ──────────────────────────────────────────────────────────────────────────


def test_beq_with_pc() -> None:
    word = b_type(0x63, 0, 10, 11, 16)
    assert disassemble(word, pc=0x1000) == "beq a0, a1, 0x1010"


def test_jal_j_pseudo() -> None:
    # JAL with rd=0 is "j target".
    word = j_type(0x6F, 0, 8)
    assert disassemble(word, pc=0x2000) == "j 0x2008"


def test_jal_ra() -> None:
    word = j_type(0x6F, 1, 16)
    assert disassemble(word, pc=0x2000) == "jal ra, 0x2010"


def test_jalr_ret_pseudo() -> None:
    # jalr x0, ra, 0 → "ret"
    word = i_type(0x67, 0, 0, 1, 0)
    assert disassemble(word) == "ret"


# ──────────────────────────────────────────────────────────────────────────
# Upper / system
# ──────────────────────────────────────────────────────────────────────────


def test_lui() -> None:
    assert disassemble(u_type(0x37, 10, 0x12345)) == "lui a0, 0x12345"


def test_auipc() -> None:
    assert disassemble(u_type(0x17, 10, 0x1)) == "auipc a0, 1"


def test_ecall() -> None:
    assert disassemble(0x00000073) == "ecall"


def test_ebreak() -> None:
    assert disassemble(0x00100073) == "ebreak"


# ──────────────────────────────────────────────────────────────────────────
# Unknown
# ──────────────────────────────────────────────────────────────────────────


def test_unknown_opcode_falls_through() -> None:
    result = disassemble(0xFFFFFFFF)
    assert result.startswith(".word 0x")


def test_compressed_word_renders_as_hword() -> None:
    from rotor.binary import RISCVBinary  # noqa: F401

    # Can't easily test through RISCVBinary.disassemble without an ELF, so
    # we exercise the disassembler's fall-through directly.
    # A 16-bit instruction's low bits are not 11.
    word = 0x00000011  # low 2 bits = 01 (compressed)
    # disassemble() sees a 32-bit word regardless, so this goes to .word.
    result = disassemble(word)
    # It's not a valid RV64I opcode 0x11; falls through.
    assert result.startswith(".word")
