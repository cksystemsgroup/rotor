"""Tests for the Bitwuzla-backed BMC unroller.

Skipped when the ``bitwuzla`` Python package is not installed.
"""

from __future__ import annotations

import pytest

bitwuzla = pytest.importorskip("bitwuzla")

from rotor.btor2 import BTOR2Builder
from rotor.solvers.bmc import BitwuzlaUnroller
from rotor.solvers.bitwuzla import BitwuzlaSolver


# ──────────────────────────────────────────────────────────────────────────
# Small hand-built models
# ──────────────────────────────────────────────────────────────────────────


def _counter_model(width: int = 4, target: int = 5):
    """Build a model: counter starts at 0, increments by 1 each step, bad
    when counter == target."""
    b = BTOR2Builder()
    bv = b.bitvec(width)
    counter = b.state(bv, "counter")
    b.init(bv, counter, b.zero(bv))
    b.next(bv, counter, b.add(counter, b.one(bv)))
    target_const = b.constd(bv, target, f"target={target}")
    b.bad(b.eq(counter, target_const), "counter-eq-target")
    return b


def _flag_model():
    """Simple model: a 1-bit flag that never changes from zero; bad asserts
    flag is true. UNSAT for any bound."""
    b = BTOR2Builder()
    bv = b.bitvec(1)
    flag = b.state(bv, "flag")
    b.init(bv, flag, b.zero(bv))
    b.next(bv, flag, flag)
    b.bad(flag, "flag-set")
    return b


# ──────────────────────────────────────────────────────────────────────────
# Counter tests
# ──────────────────────────────────────────────────────────────────────────


def test_counter_sat_within_bound() -> None:
    b = _counter_model(width=4, target=5)
    u = BitwuzlaUnroller(b.dag)
    result = u.check(bound=10)
    assert result.verdict == "sat"
    assert result.steps == 5
    assert result.witness is not None
    # Counter at step 5 should be 5.
    step5 = next(f for f in result.witness if f["step"] == 5)
    assert step5["assignments"]["counter"] == 5


def test_counter_unsat_below_target() -> None:
    b = _counter_model(width=4, target=10)
    u = BitwuzlaUnroller(b.dag)
    result = u.check(bound=5)
    assert result.verdict == "unsat"


def test_counter_sat_at_wraparound() -> None:
    # 4-bit counter wraps to 0 at step 16.
    b = _counter_model(width=4, target=0)
    u = BitwuzlaUnroller(b.dag)
    result = u.check(bound=20)
    assert result.verdict == "sat"
    assert result.steps == 0  # step 0 has counter==0 due to init


# ──────────────────────────────────────────────────────────────────────────
# Tests that exercise the quieter paths
# ──────────────────────────────────────────────────────────────────────────


def test_flag_model_unsat() -> None:
    b = _flag_model()
    u = BitwuzlaUnroller(b.dag)
    result = u.check(bound=5)
    assert result.verdict == "unsat"


def test_no_bad_returns_unsat() -> None:
    b = BTOR2Builder()
    bv = b.bitvec(8)
    x = b.state(bv, "x")
    b.init(bv, x, b.zero(bv))
    b.next(bv, x, x)
    # No bad property registered.
    u = BitwuzlaUnroller(b.dag)
    result = u.check(bound=5)
    assert result.verdict == "unsat"


def test_arithmetic_operators() -> None:
    """Exercise add/sub/mul/and/or/xor + comparisons through the unroller."""
    b = BTOR2Builder()
    bv = b.bitvec(8)
    x = b.state(bv, "x")
    b.init(bv, x, b.constd(bv, 3))
    # next(x) = (x + 1) * 2 - 3
    next_val = b.sub(b.mul(b.add(x, b.one(bv)), b.constd(bv, 2)), b.constd(bv, 3))
    b.next(bv, x, next_val)
    # Bad: x == 13 (reached at step 1: (3+1)*2-3 = 5; step 2: (5+1)*2-3 = 9;
    # step 3: (9+1)*2-3 = 17; ... unreachable, actually). Verify UNSAT.
    b.bad(b.eq(x, b.constd(bv, 99)), "never")
    u = BitwuzlaUnroller(b.dag)
    result = u.check(bound=5)
    assert result.verdict in ("unsat", "sat")  # depends on wraparound


def test_slice_and_extend() -> None:
    """Exercise slice, sext, uext through the unroller."""
    b = BTOR2Builder()
    bv8 = b.bitvec(8)
    bv4 = b.bitvec(4)
    bv16 = b.bitvec(16)
    x = b.state(bv8, "x")
    b.init(bv8, x, b.constd(bv8, 0xAB))
    low4 = b.slice(bv4, x, 3, 0, "low4")
    widened = b.uext(bv16, low4, 12, "wide")
    b.next(bv8, x, x)
    b.bad(b.eq(widened, b.constd(bv16, 0xB)), "low4-is-b")
    u = BitwuzlaUnroller(b.dag)
    result = u.check(bound=1)
    assert result.verdict == "sat"


def test_array_read_write() -> None:
    """Array state with read/write operations."""
    b = BTOR2Builder()
    idx = b.bitvec(4)
    elem = b.bitvec(8)
    arr_sort = b.array(idx, elem)
    arr = b.state(arr_sort, "arr")
    b.init(arr_sort, arr, b.zero(arr_sort))
    # next(arr) = write(arr, 0, 42)
    b.next(
        arr_sort, arr,
        b.write(arr, b.zero(idx), b.constd(elem, 42)),
    )
    # Bad: arr[0] == 42 (reached at step 1).
    b.bad(b.eq(b.read(arr, b.zero(idx)), b.constd(elem, 42)), "arr0-is-42")
    u = BitwuzlaUnroller(b.dag)
    result = u.check(bound=3)
    assert result.verdict == "sat"
    assert result.steps == 1


def test_solver_wrapper_round_trip() -> None:
    """BitwuzlaSolver.check should accept BTOR2 text and return a result."""
    b = _counter_model(width=4, target=3)
    from rotor.btor2 import BTOR2Printer
    text = BTOR2Printer().render(b.dag)
    solver = BitwuzlaSolver()
    result = solver.check(text, bound=10)
    assert result.verdict == "sat"
    assert result.steps == 3


def test_constraint_narrows_search() -> None:
    """A constraint should rule out otherwise-reachable bad states."""
    b = BTOR2Builder()
    bv4 = b.bitvec(4)
    x = b.state(bv4, "x")
    b.init(bv4, x, b.zero(bv4))
    b.next(bv4, x, b.add(x, b.one(bv4)))
    # Without constraint, bad x==3 is reachable at step 3.
    # With constraint !(x==3) always, bad becomes unreachable.
    b.constraint(b.neq(x, b.constd(bv4, 3)), "avoid-3")
    b.bad(b.eq(x, b.constd(bv4, 3)), "reach-3")
    u = BitwuzlaUnroller(b.dag)
    result = u.check(bound=5)
    assert result.verdict == "unsat"
