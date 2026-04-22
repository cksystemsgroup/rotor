"""Unit tests for the CVC5 BMC backend.

Built against synthetic in-memory BTOR2 models so tests don't
depend on the RISC-V decoder or fixtures. Integration coverage on
real fixtures is in tests/integration/test_cvc5_portfolio.py.

Module is skipped entirely when the `cvc5` pip package isn't
installed — rotor does not hard-require it.
"""

from __future__ import annotations

import pytest

cvc5 = pytest.importorskip("cvc5")

from rotor.btor2.nodes import ArraySort, Model, Sort  # noqa: E402
from rotor.solvers.cvc5bmc import CVC5BMC             # noqa: E402

BV1 = Sort(1)
BV8 = Sort(8)
BV64 = Sort(64)


def _counter(bad_at: int) -> Model:
    m = Model()
    x = m.state(BV8, "x")
    m.init(x, m.const(BV8, 0))
    m.next(x, m.op("add", BV8, x, m.const(BV8, 1)))
    m.bad(m.op("ugte", BV1, x, m.const(BV8, bad_at)))
    return m


def test_reachable_counter_returns_step_and_backend() -> None:
    r = CVC5BMC().check_reach(_counter(5), bound=5, timeout=10.0)
    assert r.verdict == "reachable"
    assert r.step == 5
    assert r.backend == "cvc5-bmc"


def test_unreachable_counter_within_bound() -> None:
    r = CVC5BMC().check_reach(_counter(5), bound=3, timeout=10.0)
    assert r.verdict == "unreachable"
    assert r.bound == 3


def test_negative_bound_rejected() -> None:
    with pytest.raises(ValueError):
        CVC5BMC().check_reach(_counter(5), bound=-1)


def test_initial_regs_extracted_for_reachable() -> None:
    # 2-state system where bad fires when pc==2 AND r==42.
    m = Model()
    pc = m.state(BV8, "pc")
    r = m.state(BV8, "r")
    m.init(pc, m.const(BV8, 0))
    m.next(pc, m.op("add", BV8, pc, m.const(BV8, 1)))
    m.next(r, r)
    is_pc2 = m.op("eq", BV1, pc, m.const(BV8, 2))
    is_r42 = m.op("eq", BV1, r, m.const(BV8, 42))
    m.bad(m.op("and", BV1, is_pc2, is_r42))

    res = CVC5BMC().check_reach(m, bound=4, timeout=10.0)
    assert res.verdict == "reachable"
    assert res.step == 2
    assert res.initial_regs["r"] == 42


def test_constraint_blocks_reach() -> None:
    # Counter with `x != 5` constraint; bad = x >= 5. Over bound 5,
    # the only reachable step-5 state has x==5, forbidden.
    m = Model()
    x = m.state(BV8, "x")
    m.init(x, m.const(BV8, 0))
    m.next(x, m.op("add", BV8, x, m.const(BV8, 1)))
    m.constraint(m.op("neq", BV1, x, m.const(BV8, 5)))
    m.bad(m.op("ugte", BV1, x, m.const(BV8, 5)))
    r = CVC5BMC().check_reach(m, bound=5, timeout=10.0)
    assert r.verdict == "unreachable"


def test_array_model_load_after_store() -> None:
    # Store 42 at addr 7 in cycle-0 init; check bad iff mem[7]==42.
    m = Model()
    mem_sort = ArraySort(index=BV64, element=BV8)
    mem = m.state(mem_sort, "mem")
    free = m.state(mem_sort, "free")
    addr = m.const(BV64, 7)
    val = m.const(BV8, 42)
    m.init(mem, m.write(free, addr, val))
    m.next(mem, mem)
    m.next(free, free)
    loaded = m.read(mem, addr)
    m.bad(m.op("eq", BV1, loaded, val))

    r = CVC5BMC().check_reach(m, bound=0, timeout=10.0)
    assert r.verdict == "reachable"
    assert r.step == 0


def test_mul_and_div_are_lowered() -> None:
    # Single-cycle sat query exercising BV_MULT and BV_UDIV paths in
    # the _apply_op dispatch.
    m = Model()
    a = m.state(BV8, "a")
    b = m.state(BV8, "b")
    m.init(a, m.const(BV8, 6))
    m.init(b, m.const(BV8, 2))
    m.next(a, a)
    m.next(b, b)
    product = m.op("mul", BV8, a, b)               # 12
    quotient = m.op("udiv", BV8, product, b)        # 6
    m.bad(m.op("eq", BV1, quotient, m.const(BV8, 6)))
    r = CVC5BMC().check_reach(m, bound=0, timeout=10.0)
    assert r.verdict == "reachable"


def test_timeout_returns_unknown_or_completes() -> None:
    # A tiny timeout on a trivial model may still complete; we only
    # assert the backend doesn't crash and returns a valid verdict.
    r = CVC5BMC().check_reach(_counter(2), bound=2, timeout=0.001)
    assert r.verdict in ("reachable", "unknown")
    assert r.backend == "cvc5-bmc"
