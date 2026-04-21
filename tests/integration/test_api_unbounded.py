"""Phase 6.5: RotorAPI's unbounded + CEGAR reach modes.

Exercises the new `unbounded=True` kwarg on `api.can_reach` and the
separate `api.cegar_reach` method end-to-end, including that the
returned `ReachResult` carries the Spacer invariant in both cases.
"""

from __future__ import annotations

from pathlib import Path

from rotor import RotorAPI
from rotor.cegar import CegarConfig

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "counter.elf"
TINY_MASK_DEAD_PC = 0x1117c


def test_can_reach_unbounded_proves_tiny_mask_dead_branch() -> None:
    with RotorAPI(FIXTURE) as api:
        r = api.can_reach(
            function="tiny_mask",
            target_pc=TINY_MASK_DEAD_PC,
            unbounded=True,
            timeout=30.0,
        )
    assert r.verdict == "proved"
    assert r.backend == "z3-spacer"
    assert r.invariant is not None and len(r.invariant) > 0
    assert r.trace is None                          # no CEX trace for `proved`


def test_cegar_reach_proves_tiny_mask_dead_branch() -> None:
    with RotorAPI(FIXTURE) as api:
        r = api.cegar_reach(
            function="tiny_mask",
            target_pc=TINY_MASK_DEAD_PC,
            config=CegarConfig(max_iterations=6, spacer_timeout=20.0),
        )
    assert r.verdict == "proved"
    assert "cegar" in r.backend
    assert r.invariant is not None


def test_can_reach_default_remains_bounded_bmc() -> None:
    # Regression: adding unbounded/cegar must not change the default
    # behavior. `can_reach(...)` without kwargs is still bounded BMC.
    with RotorAPI(FIXTURE, default_bound=5) as api:
        r = api.can_reach(
            function="tiny_mask",
            target_pc=TINY_MASK_DEAD_PC,
        )
    assert r.verdict == "unreachable"
    assert r.backend == "z3-bmc"
    assert r.invariant is None
