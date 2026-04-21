"""Decoder coverage for the RV64M (mul/div/rem) extension."""

from __future__ import annotations

from rotor.btor2.riscv.decoder import decode


def _r(funct7: int, rs2: int, rs1: int, funct3: int, rd: int, opcode: int) -> int:
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


_OP    = 0b0110011
_OP_32 = 0b0111011
_M     = 0b0000001                               # funct7 for M extension


def _assert(mnem: str, word: int, rd: int = 5, rs1: int = 10, rs2: int = 11) -> None:
    d = decode(word)
    assert d is not None, f"decode returned None for {mnem} (word=0x{word:08x})"
    assert d.mnem == mnem, f"expected {mnem}, got {d.mnem}"
    assert d.rd == rd and d.rs1 == rs1 and d.rs2 == rs2


def test_mul() -> None:
    _assert("mul", _r(_M, 11, 10, 0b000, 5, _OP))


def test_mulh() -> None:
    _assert("mulh", _r(_M, 11, 10, 0b001, 5, _OP))


def test_mulhsu() -> None:
    _assert("mulhsu", _r(_M, 11, 10, 0b010, 5, _OP))


def test_mulhu() -> None:
    _assert("mulhu", _r(_M, 11, 10, 0b011, 5, _OP))


def test_div() -> None:
    _assert("div", _r(_M, 11, 10, 0b100, 5, _OP))


def test_divu() -> None:
    _assert("divu", _r(_M, 11, 10, 0b101, 5, _OP))


def test_rem() -> None:
    _assert("rem", _r(_M, 11, 10, 0b110, 5, _OP))


def test_remu() -> None:
    _assert("remu", _r(_M, 11, 10, 0b111, 5, _OP))


def test_mulw() -> None:
    _assert("mulw", _r(_M, 11, 10, 0b000, 5, _OP_32))


def test_divw() -> None:
    _assert("divw", _r(_M, 11, 10, 0b100, 5, _OP_32))


def test_divuw() -> None:
    _assert("divuw", _r(_M, 11, 10, 0b101, 5, _OP_32))


def test_remw() -> None:
    _assert("remw", _r(_M, 11, 10, 0b110, 5, _OP_32))


def test_remuw() -> None:
    _assert("remuw", _r(_M, 11, 10, 0b111, 5, _OP_32))


def test_mulhw_not_a_thing() -> None:
    # mulh variants are only RV64 (no -w forms), so OP-32 + funct3=001
    # with funct7=M must not decode.
    d = decode(_r(_M, 11, 10, 0b001, 5, _OP_32))
    assert d is None
