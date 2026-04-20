"""M7 exit criterion: DAG IR shrinks BTOR2 on a majority of corpus cases.

The L0-equivalence harness covers correctness; this file covers
productivity. PLAN.md's M7 gate: "measurable BTOR2-size reduction on
at least 50% of corpus cases." We compare Model node counts under
IdentityEmitter vs DagEmitter across CORPUS and assert the threshold.
"""

from __future__ import annotations

from pathlib import Path

from rotor import DagEmitter, IdentityEmitter
from rotor.binary import RISCVBinary
from rotor.ir.spec import ReachSpec
from tests.equivalence.corpus import CORPUS

REPO_ROOT = Path(__file__).resolve().parents[2]


def _node_count(emitter, binary: RISCVBinary, function: str, target_pc: int) -> int:
    model = emitter.emit(ReachSpec(function=function, target_pc=target_pc))
    return len(model.nodes)


def test_dag_reduces_btor2_on_majority_of_corpus() -> None:
    reductions: list[tuple[str, int, int]] = []
    for entry in CORPUS:
        path = REPO_ROOT / entry.binary_relpath
        with RISCVBinary(path) as b:
            fn = b.function(entry.function)
            target = fn.start + entry.target_offset
            base = _node_count(IdentityEmitter(b), b, entry.function, target)
            dag  = _node_count(DagEmitter(b),      b, entry.function, target)
        reductions.append((entry.name, base, dag))

    shrunk = [r for r in reductions if r[2] < r[1]]
    # Every DagEmitter model must be no larger than the IdentityEmitter one.
    for name, base, dag in reductions:
        assert dag <= base, f"{name}: dag {dag} > identity {base}"
    # At least half the corpus must actually shrink.
    assert len(shrunk) * 2 >= len(reductions), (
        f"only {len(shrunk)}/{len(reductions)} corpus entries shrank under "
        f"DagEmitter; expected >= 50%"
    )


def test_dag_never_grows_btor2() -> None:
    """Hash-consing + rewrites are monotonic: the DAG output can be
    identical to L0 (nothing to dedupe) but must never be larger."""
    for entry in CORPUS:
        path = REPO_ROOT / entry.binary_relpath
        with RISCVBinary(path) as b:
            fn = b.function(entry.function)
            target = fn.start + entry.target_offset
            base = _node_count(IdentityEmitter(b), b, entry.function, target)
            dag  = _node_count(DagEmitter(b),      b, entry.function, target)
        assert dag <= base, f"{entry.name}: dag {dag} > identity {base}"
