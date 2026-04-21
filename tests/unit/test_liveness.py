"""Unit tests for the L2 liveness analysis (rotor/ir/liveness.py)."""

from __future__ import annotations

from pathlib import Path

from rotor.binary import RISCVBinary
from rotor.ir.liveness import _reads_of, _write_of, dead_registers, live_registers

from rotor.btor2.riscv.decoder import decode

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _d(word: int):
    d = decode(word)
    assert d is not None
    return d


# ---- instruction-level classification ---------------------------------------

def test_r_type_add_reads_rs1_and_rs2_writes_rd() -> None:
    # add x5, x10, x11
    word = (0 << 25) | (11 << 20) | (10 << 15) | (0 << 12) | (5 << 7) | 0b0110011
    d = _d(word)
    assert _reads_of(d) == frozenset({10, 11})
    assert _write_of(d) == 5


def test_branch_reads_rs1_and_rs2_no_write() -> None:
    # beq x10, x11, +4
    word = (0 << 31) | (0 << 25) | (11 << 20) | (10 << 15) | (0 << 12) | (2 << 8) | (0 << 7) | 0b1100011
    d = _d(word)
    assert _reads_of(d) == frozenset({10, 11})
    assert _write_of(d) == 0


def test_store_reads_rs1_and_rs2_no_write() -> None:
    # sw x11, 0(x10)
    word = (0 << 25) | (11 << 20) | (10 << 15) | (2 << 12) | (0 << 7) | 0b0100011
    d = _d(word)
    assert _reads_of(d) == frozenset({10, 11})
    assert _write_of(d) == 0


def test_load_reads_rs1_writes_rd() -> None:
    # lw x5, 0(x10)
    word = (0 << 20) | (10 << 15) | (2 << 12) | (5 << 7) | 0b0000011
    d = _d(word)
    assert _reads_of(d) == frozenset({10})
    assert _write_of(d) == 5


def test_lui_reads_nothing_writes_rd() -> None:
    # lui x5, 0x12345
    word = (0x12345 << 12) | (5 << 7) | 0b0110111
    d = _d(word)
    assert _reads_of(d) == frozenset()
    assert _write_of(d) == 5


def test_jal_reads_nothing_writes_rd() -> None:
    # jal x1, +4
    word = (0 << 31) | (2 << 21) | (0 << 20) | (0 << 12) | (1 << 7) | 0b1101111
    d = _d(word)
    assert _reads_of(d) == frozenset()
    assert _write_of(d) == 1


def test_jalr_reads_rs1_writes_rd() -> None:
    # jalr x0, 0(x1)  (= ret when rd=0)
    word = (0 << 20) | (1 << 15) | (0 << 12) | (0 << 7) | 0b1100111
    d = _d(word)
    assert _reads_of(d) == frozenset({1})
    assert _write_of(d) == 0                    # rd=0 writes drop


def test_x0_destination_is_not_a_write() -> None:
    # addi x0, x10, 5 — writes x0, which is the zero register; no effect.
    word = (5 << 20) | (10 << 15) | (0 << 12) | (0 << 7) | 0b0010011
    d = _d(word)
    assert _write_of(d) == 0


# ---- function-level liveness on real fixtures -------------------------------

def test_add2_leaf_liveness_is_just_ra() -> None:
    # add2: `addw a0, a0, a1; ret`. The ret reads ra; no branch reads
    # a0/a1. a0 and a1 are read by addw but addw only writes a0, which
    # is not live, so the read of a1 does not pull a1 into the set.
    with RISCVBinary(FIXTURES / "add2.elf") as b:
        live = live_registers(b, b.function("add2"))
    assert live == frozenset({1})


def test_sign_pulls_a0_via_branch_reads() -> None:
    # sign: branches on a0 (x10), returns via ret. a0 is read directly
    # by the branches, so it's live from the initial set.
    with RISCVBinary(FIXTURES / "add2.elf") as b:
        live = live_registers(b, b.function("sign"))
    assert 10 in live and 1 in live


def test_bounded_counter_slice_drops_28_registers() -> None:
    # The M8 motivating fixture: Spacer times out on the full 31-reg
    # state; liveness must identify that only {ra, a0, a1} matter for
    # the dead-branch property so SsaEmitter can havoc 28 registers.
    with RISCVBinary(FIXTURES / "counter.elf") as b:
        live = live_registers(b, b.function("bounded_counter"))
        dead = dead_registers(b, b.function("bounded_counter"))
    assert live == frozenset({1, 10, 11})
    assert len(dead) == 28


def test_dead_plus_live_equal_all_nonzero_registers() -> None:
    # Invariant: liveness partitions x1..x31 into live ∪ dead, with x0
    # excluded from both (it's the hard-wired zero register).
    with RISCVBinary(FIXTURES / "counter.elf") as b:
        fn = b.function("tiny_mask")
        live = live_registers(b, fn)
        dead = dead_registers(b, fn)
    assert live.isdisjoint(dead)
    assert live | dead == frozenset(range(1, 32))
    assert 0 not in live and 0 not in dead
