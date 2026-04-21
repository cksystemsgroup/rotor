"""Witness-simulator coverage for RV64M.

These tests step a single M-extension instruction concretely and
verify the register-file result matches the RISC-V spec, including
the edge cases (div-by-zero, INT_MIN/-1 overflow, sign-extended
-w variants).
"""

from __future__ import annotations

from rotor.btor2.riscv.decoder import decode
from rotor.witness import _STEP

MASK = (1 << 64) - 1


def _word(funct7: int, rs2: int, rs1: int, funct3: int, rd: int, opcode: int) -> int:
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def _step(word: int, *, x10: int = 0, x11: int = 0) -> int:
    d = decode(word)
    regs = [0] * 32
    regs[10] = x10 & MASK
    regs[11] = x11 & MASK
    _STEP[d.mnem](d, 0x1000, regs, {})
    return regs[d.rd]


# ---- mul ------------------------------------------------------------------

def test_mul_positive() -> None:
    w = _word(0b0000001, 11, 10, 0b000, 5, 0b0110011)
    assert _step(w, x10=3, x11=4) == 12


def test_mul_wraps_at_64_bits() -> None:
    w = _word(0b0000001, 11, 10, 0b000, 5, 0b0110011)
    # (2^32) * (2^32) = 2^64 → 0 in 64 bits
    assert _step(w, x10=1 << 32, x11=1 << 32) == 0


def test_mulh_signed_signed() -> None:
    w = _word(0b0000001, 11, 10, 0b001, 5, 0b0110011)
    # -1 * -1 = 1; upper 64 of 128-bit product is 0.
    assert _step(w, x10=MASK, x11=MASK) == 0
    # large positive * large positive: 2^63 * 2 = 2^64, so upper = 1 ...
    # but 2^63 interpreted as int64 is -2^63, so actually signed product
    # is -2^63 * 2 = -2^64 → in 128-bit two's complement, that's
    # 0xFFFFFFFFFFFFFFFF'0000000000000000, upper = 0xFF...FF.
    assert _step(w, x10=1 << 63, x11=2) == MASK


def test_mulhu_unsigned() -> None:
    w = _word(0b0000001, 11, 10, 0b011, 5, 0b0110011)
    # 2^63 * 2 = 2^64 → upper = 1
    assert _step(w, x10=1 << 63, x11=2) == 1


def test_mulhsu_signed_unsigned() -> None:
    w = _word(0b0000001, 11, 10, 0b010, 5, 0b0110011)
    # -1 (signed) * 1 (unsigned) = -1 → upper = all-ones in 128-bit sext.
    assert _step(w, x10=MASK, x11=1) == MASK


# ---- div / divu / rem / remu ----------------------------------------------

def test_divu_by_zero_returns_all_ones() -> None:
    w = _word(0b0000001, 11, 10, 0b101, 5, 0b0110011)
    assert _step(w, x10=42, x11=0) == MASK


def test_divu_normal() -> None:
    w = _word(0b0000001, 11, 10, 0b101, 5, 0b0110011)
    assert _step(w, x10=20, x11=3) == 6


def test_remu_by_zero_returns_dividend() -> None:
    w = _word(0b0000001, 11, 10, 0b111, 5, 0b0110011)
    assert _step(w, x10=42, x11=0) == 42


def test_remu_normal() -> None:
    w = _word(0b0000001, 11, 10, 0b111, 5, 0b0110011)
    assert _step(w, x10=20, x11=3) == 2


def test_div_by_zero_returns_minus_one() -> None:
    w = _word(0b0000001, 11, 10, 0b100, 5, 0b0110011)
    assert _step(w, x10=42, x11=0) == MASK


def test_div_int_min_by_minus_one_is_int_min() -> None:
    w = _word(0b0000001, 11, 10, 0b100, 5, 0b0110011)
    assert _step(w, x10=1 << 63, x11=MASK) == 1 << 63


def test_div_truncates_toward_zero() -> None:
    w = _word(0b0000001, 11, 10, 0b100, 5, 0b0110011)
    # -7 / 2 = -3 (truncated), not -4 (floor).
    assert _step(w, x10=(-7) & MASK, x11=2) == (-3) & MASK


def test_rem_by_zero_returns_dividend() -> None:
    w = _word(0b0000001, 11, 10, 0b110, 5, 0b0110011)
    assert _step(w, x10=42, x11=0) == 42


def test_rem_int_min_by_minus_one_is_zero() -> None:
    w = _word(0b0000001, 11, 10, 0b110, 5, 0b0110011)
    assert _step(w, x10=1 << 63, x11=MASK) == 0


def test_rem_matches_dividend_sign() -> None:
    w = _word(0b0000001, 11, 10, 0b110, 5, 0b0110011)
    # -7 % 2 = -1 (not +1 Python-style).
    assert _step(w, x10=(-7) & MASK, x11=2) == (-1) & MASK


# ---- -w variants ----------------------------------------------------------

def test_mulw_sign_extends() -> None:
    w = _word(0b0000001, 11, 10, 0b000, 5, 0b0111011)
    # Full 64-bit input: upper bits should be ignored.
    # Input x10 = 0x1_00000003, x11 = 2 → product low 32 = 6, sext to 64.
    assert _step(w, x10=(1 << 32) | 3, x11=2) == 6


def test_divw_by_zero() -> None:
    w = _word(0b0000001, 11, 10, 0b100, 5, 0b0111011)
    # divw by zero → all-ones 32-bit, sign-extended → MASK (all ones 64).
    assert _step(w, x10=7, x11=0) == MASK


def test_divw_int_min_by_minus_one_handled() -> None:
    # divw (signed 32-bit division): INT32_MIN / -1 overflow case.
    # Spec: result = INT32_MIN, sign-extended to 64 bits.
    w = _word(0b0000001, 11, 10, 0b100, 5, 0b0111011)
    result = _step(w, x10=1 << 31, x11=MASK)
    assert result == _signed32_sext(1 << 31)


def _signed32_sext(v: int) -> int:
    v &= 0xFFFFFFFF
    return v | 0xFFFFFFFF00000000 if v & (1 << 31) else v
