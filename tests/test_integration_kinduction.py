"""End-to-end integration tests for k-induction through the public API.

These exercise the full pipeline RotorInstance → BTOR2Printer → parse_btor2
→ KInductionSolver, verifying that ``ModelConfig.solver = "kind"`` is
recognized and routes correctly, and that the resulting CheckResult carries
a proof string when a property is inductive.
"""

from __future__ import annotations

import pytest

bitwuzla = pytest.importorskip("bitwuzla")


def test_rotor_instance_solver_kind_returns_unsat_on_safe_model() -> None:
    """Build a tiny RotorInstance-compatible model and prove it unbounded."""
    from rotor.btor2 import BTOR2Builder, BTOR2Printer
    from rotor.solvers import make_solver

    b = BTOR2Builder()
    bv1 = b.bitvec(1)
    halted = b.state(bv1, "halted")
    b.init(bv1, halted, b.zero(bv1))
    b.next(bv1, halted, halted)  # never changes from 0
    b.bad(halted, "halted-fires")

    text = BTOR2Printer().render(b.dag)
    solver = make_solver("kind", max_k=4)
    result = solver.check(text, bound=4)

    assert result.verdict == "unsat"
    assert result.invariant is not None
    assert "1-inductive" in result.invariant


def test_rotor_api_verify_unbounded_proves_trivial_invariant() -> None:
    """End-to-end: RotorAPI.verify(..., unbounded=True) should return
    'holds' with a k-induction proof for a trivially-true invariant.

    The default ``illegal-instruction`` bad is suppressed for unbounded
    queries (it isn't k-inductive in skip_init mode), so only the user's
    invariant drives the proof.
    """
    import os
    import pytest
    from rotor import RotorAPI

    fixture = os.path.join(
        os.path.dirname(__file__), "fixtures", "add2.elf"
    )
    if not os.path.exists(fixture):
        pytest.skip("add2.elf fixture not built")

    api = RotorAPI(fixture, default_bound=4)
    result = api.verify("add2", "0 == 0", bound=4, unbounded=True)
    assert result.verdict == "holds"
    assert result.unbounded is True
    assert result.proof is not None
    assert "inductive" in result.proof


def test_kind_on_riscv_instance_routes_correctly() -> None:
    """A RotorInstance with solver='kind' should exercise the native
    Python builder, emit BTOR2, parse it back in the solver, and return a
    conclusive verdict on a tractable model (no BMC counterexample within
    the bound, and no clearly inductive invariant to prove either → unknown
    is acceptable; what we verify here is only that the solver is invoked)."""
    from rotor import RISCVBinary, RotorInstance, ModelConfig
    import os

    fixture = os.path.join(
        os.path.dirname(__file__), "fixtures", "add2.elf"
    )
    if not os.path.exists(fixture):
        pytest.skip("add2.elf fixture not built")

    with RISCVBinary(fixture) as binary:
        low, high = binary.function_bounds("add2")
        cfg = ModelConfig(
            is_64bit=True, code_start=low, code_end=high,
            solver="kind", bound=3, model_backend="python",
        )
        inst = RotorInstance(binary, cfg)
        result = inst.check()
        assert result.verdict in ("unsat", "unknown")
        assert result.solver == "kind"
