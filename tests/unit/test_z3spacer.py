"""Z3Spacer unit tests.

Spacer is exercised on hand-built BTOR2 counter Models — small,
array-free, and structurally what PDR was designed for. Large
array-backed models (the full L0 encoding of a RISC-V function)
are covered at integration level via the portfolio racing Spacer
against Z3BMC; Spacer may return `unknown` there, which is the
expected tradeoff, not a regression.
"""

from __future__ import annotations

from rotor.btor2.nodes import Model, Sort
from rotor.solvers import Z3Spacer

BV1 = Sort(1)
BV8 = Sort(8)


def _free_counter(bad_threshold: int) -> Model:
    """An unbounded 8-bit increment-by-1 counter; bad iff x >= threshold."""
    m = Model()
    x = m.state(BV8, "x")
    m.init(x, m.const(BV8, 0))
    m.next(x, m.op("add", BV8, x, m.const(BV8, 1)))
    m.bad(m.op("ugte", BV1, x, m.const(BV8, bad_threshold)))
    return m


def _capped_counter(cap: int) -> Model:
    """An 8-bit counter that stops incrementing at `cap`; bad iff x > cap."""
    m = Model()
    x = m.state(BV8, "x")
    m.init(x, m.const(BV8, 0))
    cap_node = m.const(BV8, cap)
    keep_going = m.op("ult", BV1, x, cap_node)
    m.next(x, m.ite(keep_going, m.op("add", BV8, x, m.const(BV8, 1)), x))
    m.bad(m.op("ugt", BV1, x, cap_node))
    return m


def test_unbounded_counter_is_reachable() -> None:
    r = Z3Spacer().check_reach(_free_counter(bad_threshold=6), bound=0, timeout=10.0)
    assert r.verdict == "reachable"
    assert r.backend == "z3-spacer"


def test_capped_counter_is_proved_safe() -> None:
    r = Z3Spacer().check_reach(_capped_counter(cap=5), bound=0, timeout=10.0)
    assert r.verdict == "proved"
    # Spacer emits an invariant expression; the exact form is engine
    # implementation detail, but it must be a non-empty string.
    assert r.invariant is not None and len(r.invariant) > 0


def test_no_bad_is_vacuously_proved() -> None:
    m = Model()
    x = m.state(BV8, "x")
    m.init(x, m.const(BV8, 0))
    m.next(x, x)
    r = Z3Spacer().check_reach(m, bound=0, timeout=5.0)
    assert r.verdict == "proved"


def test_timeout_returns_unknown() -> None:
    # A capped counter is easy; a very aggressive timeout might still
    # succeed. The shape of this test is mainly that the backend doesn't
    # crash on short timeouts — we accept proved or unknown.
    r = Z3Spacer().check_reach(_capped_counter(cap=3), bound=0, timeout=0.001)
    assert r.verdict in ("proved", "unknown")
    assert r.backend == "z3-spacer"


def test_bound_parameter_is_ignored() -> None:
    # PDR is unbounded; passing different bounds must yield the same verdict.
    r1 = Z3Spacer().check_reach(_capped_counter(cap=5), bound=0)
    r2 = Z3Spacer().check_reach(_capped_counter(cap=5), bound=1000)
    assert r1.verdict == r2.verdict == "proved"
