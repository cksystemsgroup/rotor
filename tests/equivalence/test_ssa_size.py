"""M8 exit criterion: SSA slicing shrinks BTOR2 below DAG on every
fixture that has dead registers to slice.

The L0-equivalence harness covers correctness (same verdict, same
step, across the full corpus). This file covers productivity. For a
reach question `can_reach(fn, target_pc)`, the live-register set
bounds the Model's register-state count; when some registers are
dead, SsaEmitter produces a strictly-smaller Model than DagEmitter
(which already hash-conses but keeps every state).

Monotonicity gate: SSA ≤ DAG ≤ L0 on every entry, period.
"""

from __future__ import annotations

from pathlib import Path

from rotor import DagEmitter, IdentityEmitter, SsaEmitter
from rotor.binary import RISCVBinary
from rotor.ir.liveness import dead_registers
from rotor.ir.spec import ReachSpec
from tests.equivalence.corpus import CORPUS

REPO_ROOT = Path(__file__).resolve().parents[2]


def _nodes(emitter, function: str, target_pc: int) -> int:
    return len(emitter.emit(ReachSpec(function=function, target_pc=target_pc)).nodes)


def test_ssa_never_exceeds_dag_or_identity() -> None:
    for entry in CORPUS:
        path = REPO_ROOT / entry.binary_relpath
        with RISCVBinary(path) as b:
            fn = b.function(entry.function)
            target = fn.start + entry.target_offset
            l0 = _nodes(IdentityEmitter(b), entry.function, target)
            l1 = _nodes(DagEmitter(b),      entry.function, target)
            l2 = _nodes(SsaEmitter(b),      entry.function, target)
        assert l2 <= l1 <= l0, (
            f"{entry.name}: l2={l2} l1={l1} l0={l0} (must be monotonically non-increasing)"
        )


def test_ssa_shrinks_when_dead_registers_exist() -> None:
    """If the function has at least one dead register for this reach
    question, SsaEmitter must produce strictly fewer nodes than
    DagEmitter (the closest non-slicing IR)."""
    shrunk = 0
    total_with_dead = 0
    for entry in CORPUS:
        path = REPO_ROOT / entry.binary_relpath
        with RISCVBinary(path) as b:
            fn = b.function(entry.function)
            target = fn.start + entry.target_offset
            dead = dead_registers(b, fn)
            if not dead:
                continue
            total_with_dead += 1
            l1 = _nodes(DagEmitter(b), entry.function, target)
            l2 = _nodes(SsaEmitter(b), entry.function, target)
            if l2 < l1:
                shrunk += 1
            assert l2 <= l1, f"{entry.name}: l2={l2} > l1={l1} despite {len(dead)} dead regs"

    assert total_with_dead > 0, "corpus has no fixture with dead registers — liveness trivial"
    assert shrunk == total_with_dead, (
        f"SsaEmitter shrank only {shrunk}/{total_with_dead} entries with dead regs"
    )
