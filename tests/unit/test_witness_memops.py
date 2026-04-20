"""Witness simulator coverage for RV64I loads and stores (M6).

These assert (Decoded -> memory/register effect) matches RISC-V
semantics for every load/store mnemonic, plus a store->load
round-trip through the dict-backed memory mirror. The decoder and
BTOR2 lowering get their own coverage elsewhere; this file pins down
the simulator half of the semantic contract.
"""

from __future__ import annotations

from rotor.btor2.riscv.decoder import Decoded
from rotor.witness import _step

XLEN = 64
MASK = (1 << XLEN) - 1


def _regs(**named: int) -> list[int]:
    r = [0] * 32
    for name, value in named.items():
        r[int(name[1:])] = value & MASK
    return r


# ---------------- stores lay bytes out little-endian ---------------- #

def test_sb_stores_low_byte() -> None:
    mem: dict[int, int] = {}
    r = _regs(x10=0x1000, x11=0xDEADBEEF)
    _step(Decoded("sb", 0, 10, 11, 0), 0, r, mem)
    assert mem == {0x1000: 0xEF}


def test_sh_writes_two_bytes_le() -> None:
    mem: dict[int, int] = {}
    r = _regs(x10=0x2000, x11=0xABCD)
    _step(Decoded("sh", 0, 10, 11, 0), 0, r, mem)
    assert mem == {0x2000: 0xCD, 0x2001: 0xAB}


def test_sw_writes_four_bytes_le() -> None:
    mem: dict[int, int] = {}
    r = _regs(x10=0x3000, x11=0x11223344)
    _step(Decoded("sw", 0, 10, 11, 0), 0, r, mem)
    assert mem == {0x3000: 0x44, 0x3001: 0x33, 0x3002: 0x22, 0x3003: 0x11}


def test_sd_writes_eight_bytes_le() -> None:
    mem: dict[int, int] = {}
    r = _regs(x10=0x4000, x11=0x0123456789ABCDEF)
    _step(Decoded("sd", 0, 10, 11, 0), 0, r, mem)
    bytes_out = [mem[0x4000 + i] for i in range(8)]
    assert bytes_out == [0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01]


# ---------------- loads compose bytes with correct extension ---------------- #

def test_lw_sign_extends_negative_32bit() -> None:
    # 0x80000000 in 32-bit is a negative number; lw must sign-extend to 64.
    mem = {0x1000 + i: b for i, b in enumerate([0x00, 0x00, 0x00, 0x80])}
    r = _regs(x10=0x1000)
    _step(Decoded("lw", 11, 10, 0, 0), 0, r, mem)
    assert r[11] == 0xFFFFFFFF80000000


def test_lwu_zero_extends() -> None:
    mem = {0x1000 + i: b for i, b in enumerate([0x00, 0x00, 0x00, 0x80])}
    r = _regs(x10=0x1000)
    _step(Decoded("lwu", 11, 10, 0, 0), 0, r, mem)
    assert r[11] == 0x80000000


def test_lb_sign_extends_byte() -> None:
    mem = {0x2000: 0xFE}
    r = _regs(x10=0x2000)
    _step(Decoded("lb", 11, 10, 0, 0), 0, r, mem)
    assert r[11] == (-2) & MASK


def test_lbu_zero_extends_byte() -> None:
    mem = {0x2000: 0xFE}
    r = _regs(x10=0x2000)
    _step(Decoded("lbu", 11, 10, 0, 0), 0, r, mem)
    assert r[11] == 0xFE


def test_lh_sign_extends_halfword() -> None:
    mem = {0x3000: 0xFF, 0x3001: 0xFF}
    r = _regs(x10=0x3000)
    _step(Decoded("lh", 11, 10, 0, 0), 0, r, mem)
    assert r[11] == MASK                             # all-ones


def test_lhu_zero_extends_halfword() -> None:
    mem = {0x3000: 0xFF, 0x3001: 0xFF}
    r = _regs(x10=0x3000)
    _step(Decoded("lhu", 11, 10, 0, 0), 0, r, mem)
    assert r[11] == 0xFFFF


def test_ld_reads_eight_bytes_le() -> None:
    mem = {0x4000 + i: b for i, b in enumerate([0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01])}
    r = _regs(x10=0x4000)
    _step(Decoded("ld", 11, 10, 0, 0), 0, r, mem)
    assert r[11] == 0x0123456789ABCDEF


# ---------------- store -> load round-trip ---------------- #

def test_store_then_load_roundtrip() -> None:
    """sw at offset N then lw at offset N must return the stored value."""
    mem: dict[int, int] = {}
    r = _regs(x10=0x5000, x11=0xCAFEBABE)
    _step(Decoded("sw", 0, 10, 11, 4), 0, r, mem)
    _step(Decoded("lw", 12, 10, 0, 4), 0, r, mem)
    assert r[12] == 0xFFFFFFFFCAFEBABE               # sign-extended (top bit set)


def test_store_then_load_unsigned_roundtrip() -> None:
    mem: dict[int, int] = {}
    r = _regs(x10=0x5000, x11=0x12345678)
    _step(Decoded("sw", 0, 10, 11, 0), 0, r, mem)
    _step(Decoded("lwu", 12, 10, 0, 0), 0, r, mem)
    assert r[12] == 0x12345678


def test_negative_offset_address_computation() -> None:
    """Store/load immediates are sign-extended; negative offsets work."""
    mem: dict[int, int] = {}
    r = _regs(x10=0x6000, x11=0xAA)
    _step(Decoded("sb", 0, 10, 11, -4), 0, r, mem)
    assert mem[0x5FFC] == 0xAA
    _step(Decoded("lbu", 12, 10, 0, -4), 0, r, mem)
    assert r[12] == 0xAA


def test_uninitialized_load_returns_zero() -> None:
    """Bytes not in the mirror default to 0 — matches the SMT picked
    a concrete 0 model for the free bytes in the witness."""
    mem: dict[int, int] = {}
    r = _regs(x10=0x7000)
    _step(Decoded("lw", 11, 10, 0, 0), 0, r, mem)
    assert r[11] == 0
