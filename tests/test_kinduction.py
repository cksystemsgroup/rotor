"""Tests for k-induction (rotor.solvers.kind)."""

from __future__ import annotations

import pytest

bitwuzla = pytest.importorskip("bitwuzla")

from rotor.btor2 import BTOR2Builder, BTOR2Printer
from rotor.solvers.kind import KInductionSolver


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _run_kind(builder: BTOR2Builder, *, max_k: int = 8, bound: int = 8):
    text = BTOR2Printer().render(builder.dag)
    solver = KInductionSolver(max_k=max_k)
    return solver.check(text, bound=bound)


# ──────────────────────────────────────────────────────────────────────────
# Base-case routing
# ──────────────────────────────────────────────────────────────────────────


def test_violation_within_bound_returns_sat() -> None:
    """If BMC finds a CEX, k-induction should surface it as SAT."""
    b = BTOR2Builder()
    bv4 = b.bitvec(4)
    x = b.state(bv4, "x")
    b.init(bv4, x, b.zero(bv4))
    b.next(bv4, x, b.add(x, b.one(bv4)))
    b.bad(b.eq(x, b.constd(bv4, 3)), "reach-3")
    result = _run_kind(b, max_k=6, bound=6)
    assert result.verdict == "sat"


# ──────────────────────────────────────────────────────────────────────────
# 1-inductive properties
# ──────────────────────────────────────────────────────────────────────────


def test_constant_invariant_is_1_inductive() -> None:
    """A latch stuck at 0 trivially holds forever."""
    b = BTOR2Builder()
    bv1 = b.bitvec(1)
    flag = b.state(bv1, "flag")
    b.init(bv1, flag, b.zero(bv1))
    b.next(bv1, flag, flag)  # never changes
    b.bad(flag, "flag-set")
    result = _run_kind(b)
    assert result.verdict == "unsat"
    assert result.invariant is not None
    assert "1-inductive" in result.invariant


def test_monotonic_counter_bounded_below_max_is_inductive() -> None:
    """A counter that increments only while below a bound never exceeds it."""
    b = BTOR2Builder()
    bv4 = b.bitvec(4)
    x = b.state(bv4, "x")
    b.init(bv4, x, b.zero(bv4))
    # x := x < 8 ? x+1 : x
    lt_8 = b.ult(x, b.constd(bv4, 8))
    incr = b.ite(lt_8, b.add(x, b.one(bv4)), x)
    b.next(bv4, x, incr)
    # bad: x > 8 (should be unreachable).
    b.bad(b.ugt(x, b.constd(bv4, 8)), "x>8")
    result = _run_kind(b, max_k=8, bound=8)
    assert result.verdict == "unsat"


# ──────────────────────────────────────────────────────────────────────────
# Properties that aren't trivially k-inductive
# ──────────────────────────────────────────────────────────────────────────


def _not_x_model():
    """x := ~x, init 0, bad x==3 on a 3-bit value. The property (x != 3)
    is not 1-inductive: from x=4 (which satisfies x != 3) the transition
    produces ~4 = 3 (which doesn't). But ~(~x) = x, so the property is
    2-inductive: if x != 3 held at step 0 and step 1, the double
    complement back to x_0 at step 2 preserves the property."""
    b = BTOR2Builder()
    bv3 = b.bitvec(3)
    x = b.state(bv3, "x")
    b.init(bv3, x, b.zero(bv3))
    b.next(bv3, x, b.not_(x))
    b.bad(b.eq(x, b.constd(bv3, 3)), "x==3")
    return b


def test_property_not_1_inductive_returns_unknown() -> None:
    b = _not_x_model()
    result = _run_kind(b, max_k=1, bound=3)
    assert result.verdict == "unknown"
    assert "not k-inductive" in (result.stderr or "")


def test_property_2_inductive_with_max_k_2() -> None:
    b = _not_x_model()
    result = _run_kind(b, max_k=2, bound=3)
    assert result.verdict == "unsat"
    assert result.invariant is not None
    assert "2-inductive" in result.invariant


# ──────────────────────────────────────────────────────────────────────────
# Solver registry
# ──────────────────────────────────────────────────────────────────────────


def test_kind_registered_in_make_solver() -> None:
    from rotor.solvers import make_solver
    s = make_solver("kind")
    assert s.name == "kind"
    assert s.supports_unbounded()


def test_no_bads_means_trivially_safe() -> None:
    """A model without any bad property is trivially safe at all bounds."""
    b = BTOR2Builder()
    bv8 = b.bitvec(8)
    x = b.state(bv8, "x")
    b.init(bv8, x, b.zero(bv8))
    b.next(bv8, x, b.add(x, b.one(bv8)))
    # No bad installed.
    result = _run_kind(b)
    assert result.verdict == "unsat"
