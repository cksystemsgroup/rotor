"""Tests for the `havoc_regs` CEGAR abstraction primitive on
`build_reach`.

The knob replaces named registers with per-cycle BTOR2 `input` nodes
instead of stateful `state` nodes with computed next-state expressions.
Reads see fresh symbolic values every cycle; writes are dropped.

These tests cover the BTOR2 shape that `build_reach` emits. Integration
with CEGAR's refinement loop is exercised in test_cegar.py once that
lands.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rotor.binary import RISCVBinary
from rotor.btor2.builder import build_reach
from rotor.ir.spec import ReachSpec
from rotor.solvers import Z3BMC

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "add2.elf"


def _regs_by_kind(model, kind: str) -> set[str]:
    return {n.name for n in model.nodes if n.kind == kind and (n.name or "").startswith("x")}


def test_default_build_has_31_register_states_and_no_register_inputs() -> None:
    with RISCVBinary(FIXTURE) as b:
        m = build_reach(b, ReachSpec(function="add2", target_pc=b.function("add2").start))
    state_regs = _regs_by_kind(m, "state")
    input_regs = _regs_by_kind(m, "input")
    assert state_regs == {f"x{i}" for i in range(1, 32)}
    assert input_regs == set()


def test_havoc_moves_named_registers_to_inputs() -> None:
    with RISCVBinary(FIXTURE) as b:
        m = build_reach(
            b,
            ReachSpec(function="add2", target_pc=b.function("add2").start),
            havoc_regs={10, 11},
        )
    state_regs = _regs_by_kind(m, "state")
    input_regs = _regs_by_kind(m, "input")
    assert "x10" in input_regs and "x11" in input_regs
    assert "x10" not in state_regs and "x11" not in state_regs
    # Other regs untouched.
    assert "x1" in state_regs and "x12" in state_regs


def test_havoc_x0_is_silently_ignored() -> None:
    # x0 is a hard-wired constant, not a state or input. Requesting havoc
    # on it must not perturb the encoding.
    with RISCVBinary(FIXTURE) as b:
        m = build_reach(
            b,
            ReachSpec(function="add2", target_pc=b.function("add2").start),
            havoc_regs={0, 10},
        )
    input_regs = _regs_by_kind(m, "input")
    assert "x0" not in input_regs
    assert "x10" in input_regs


def test_havoc_ra_elides_entry_constraint() -> None:
    # The ra-outside-fn-pc-range constraint is meaningless when ra is
    # havoc'd (a fresh value every cycle has no "initial" to constrain),
    # so build_reach should skip emitting it. Concretely: the default
    # model has exactly one constraint node; havoc'ing x1 removes it.
    with RISCVBinary(FIXTURE) as b:
        default = build_reach(b, ReachSpec(function="add2", target_pc=b.function("add2").start))
        havoc_ra = build_reach(
            b,
            ReachSpec(function="add2", target_pc=b.function("add2").start),
            havoc_regs={1},
        )
    default_constraints = [n for n in default.nodes if n.kind == "constraint"]
    havoc_constraints = [n for n in havoc_ra.nodes if n.kind == "constraint"]
    assert len(default_constraints) == 1
    assert len(havoc_constraints) == 0


def test_havoc_all_regs_makes_every_in_function_pc_reachable() -> None:
    # Fully-abstracted model: every register is havoc'd. The PC-dispatch
    # tree still runs, but every read returns an arbitrary value, so the
    # abstraction admits every branching path. A mid-function PC that is
    # unreachable in the concrete model is reachable in this abstraction.
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("sign")
        # sign's +4 (li a5, 0) is unreachable at bound 0 in the concrete
        # model (see corpus: sign-unreach-within-bound at bound=0). With
        # full havoc it's still unreachable at bound 0 (pc hasn't moved
        # yet), but at bound 2 the abstract model reaches the mid-function
        # arms freely.
        m = build_reach(
            b,
            ReachSpec(function="sign", target_pc=fn.start + 0x10),
            havoc_regs=set(range(1, 32)),
        )
        r = Z3BMC().check_reach(m, bound=2, timeout=10.0)
        assert r.verdict == "reachable"


def test_havoc_preserves_concrete_reachability() -> None:
    # Abstraction over-approximates: anything reachable in the concrete
    # model must still be reachable in the havoc'd model. Pick a PC that
    # is trivially reachable in 1 step (the ret of add2) and confirm both
    # encodings agree.
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("add2")
        spec = ReachSpec(function="add2", target_pc=fn.start + 4)
        concrete = build_reach(b, spec)
        abstract = build_reach(b, spec, havoc_regs={10, 11, 12})
        r_concrete = Z3BMC().check_reach(concrete, bound=1)
        r_abstract = Z3BMC().check_reach(abstract, bound=1)
    assert r_concrete.verdict == "reachable"
    assert r_abstract.verdict == "reachable"
