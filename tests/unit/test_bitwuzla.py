"""Unit tests for the Bitwuzla BMC backend.

Like test_z3bv, these are built against synthetic in-memory BTOR2
models so tests don't depend on the RISC-V decoder or fixtures.
`tests/integration/test_ic3_counter.py` et al. cover the fixture
path via the portfolio.

The module is skipped entirely if the `bitwuzla` Python package
isn't installed — rotor does not hard-require it.
"""

from __future__ import annotations

import pytest

bz = pytest.importorskip("bitwuzla")

from rotor.btor2.nodes import ArraySort, Model, Sort  # noqa: E402
from rotor.solvers.bitwuzla import BitwuzlaBMC       # noqa: E402

BV1 = Sort(1)
BV8 = Sort(8)
BV64 = Sort(64)


def _counter(bad_at: int) -> Model:
    """8-bit counter that increments by 1; bad iff x >= bad_at."""
    m = Model()
    x = m.state(BV8, "x")
    m.init(x, m.const(BV8, 0))
    m.next(x, m.op("add", BV8, x, m.const(BV8, 1)))
    m.bad(m.op("ugte", BV1, x, m.const(BV8, bad_at)))
    return m


def test_reachable_counter_returns_step_and_backend() -> None:
    r = BitwuzlaBMC().check_reach(_counter(5), bound=5, timeout=10.0)
    assert r.verdict == "reachable"
    assert r.step == 5
    assert r.backend == "bitwuzla-bmc"


def test_unreachable_counter_within_bound() -> None:
    r = BitwuzlaBMC().check_reach(_counter(5), bound=3, timeout=10.0)
    assert r.verdict == "unreachable"
    assert r.bound == 3


def test_negative_bound_rejected() -> None:
    with pytest.raises(ValueError):
        BitwuzlaBMC().check_reach(_counter(5), bound=-1)


def test_initial_regs_extracted_for_reachable() -> None:
    # A 2-state system where whether bad fires depends on the free
    # initial value of a register-like state `r`.
    m = Model()
    pc = m.state(BV8, "pc")
    r = m.state(BV8, "r")
    m.init(pc, m.const(BV8, 0))
    # no init on r — free symbolic
    m.next(pc, m.op("add", BV8, pc, m.const(BV8, 1)))
    m.next(r, r)
    # bad when pc == 2 AND r == 42
    is_pc2 = m.op("eq", BV1, pc, m.const(BV8, 2))
    is_r42 = m.op("eq", BV1, r, m.const(BV8, 42))
    m.bad(m.op("and", BV1, is_pc2, is_r42))

    res = BitwuzlaBMC().check_reach(m, bound=4, timeout=10.0)
    assert res.verdict == "reachable"
    assert res.step == 2
    assert res.initial_regs["r"] == 42


def test_constraint_blocks_reach() -> None:
    # Same counter as _counter(5) but with a constraint that x != 5.
    # Over bound 5, bad (x >= 5) cannot fire: the only reachable
    # step-5 state has x == 5, which the constraint excludes.
    m = Model()
    x = m.state(BV8, "x")
    m.init(x, m.const(BV8, 0))
    m.next(x, m.op("add", BV8, x, m.const(BV8, 1)))
    m.constraint(m.op("neq", BV1, x, m.const(BV8, 5)))
    m.bad(m.op("ugte", BV1, x, m.const(BV8, 5)))
    r = BitwuzlaBMC().check_reach(m, bound=5, timeout=10.0)
    assert r.verdict == "unreachable"


def test_array_model_load_after_store() -> None:
    # One-cycle model that stores 42 at address 7 and checks bad iff
    # the byte at 7 is 42. Exercises the array-sort fold paths.
    m = Model()
    BV64 = Sort(64)
    BV8_ = Sort(8)
    mem_sort = ArraySort(index=BV64, element=BV8_)
    mem = m.state(mem_sort, "mem")
    free = m.state(mem_sort, "free")
    addr = m.const(BV64, 7)
    val = m.const(BV8_, 42)
    m.init(mem, m.write(free, addr, val))
    m.next(mem, mem)
    m.next(free, free)
    loaded = m.read(mem, addr)
    m.bad(m.op("eq", BV1, loaded, val))

    r = BitwuzlaBMC().check_reach(m, bound=0, timeout=10.0)
    assert r.verdict == "reachable"
    assert r.step == 0


def test_timeout_is_honored() -> None:
    # Force the backend to time out on a trivially-reachable model by
    # asking for a tiny timeout on a large bound. Verdict must be
    # either `reachable` (if bitwuzla is fast enough) or `unknown`
    # — never incorrect.
    r = BitwuzlaBMC().check_reach(_counter(2), bound=2, timeout=0.001)
    assert r.verdict in ("reachable", "unknown")
    assert r.backend == "bitwuzla-bmc"
