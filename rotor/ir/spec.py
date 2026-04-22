"""Question specifications for rotor.

A QuestionSpec is what an emitter compiles into a BTOR2 Model. L0
ships `ReachSpec` (can_reach) and `VerifySpec` (verify);
`FindInputSpec` and `EquivalenceSpec` are pending Track D follow-ups.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Register-comparison operators used by VerifySpec and future verbs.
# Names mirror BTOR2 op names so the builder can pass them straight
# through to `Model.op(...)`.
Comparison = Literal[
    "eq", "neq",
    "slt", "slte", "sgt", "sgte",
    "ult", "ulte", "ugt", "ugte",
]


@dataclass(frozen=True)
class QuestionSpec:
    """Marker base for all emitter-consumable question specs."""


@dataclass(frozen=True)
class ReachSpec(QuestionSpec):
    """can_reach obligation: is target_pc reachable within the BMC bound?"""
    function: str
    target_pc: int


@dataclass(frozen=True)
class VerifySpec(QuestionSpec):
    """verify obligation: does the function's return value satisfy a
    constraint at every return site?

    The predicate is the conjunction of a comparison operator and a
    64-bit integer right-hand side, evaluated against the named
    register when PC reaches any `ret` (jalr x0, x1, 0) inside the
    function. The emitter negates it into a `bad` expression:

        bad = (pc at any ret in fn) ∧ ¬(regs[register] OP rhs)

    So a `proved` verdict means "the predicate holds on every
    execution path that returns from the function"; a `reachable`
    verdict means "some input makes the predicate fail at a return
    site" and initial_regs plus the trace identify the
    counterexample.

    `register` is the ABI register index (1..31). Use `10` for `a0`,
    rotor's return-value convention. `comparison` is a BTOR2-level
    op name so the builder can wire it directly. `rhs` is
    interpreted per the comparison's signedness.
    """
    function: str
    register: int
    comparison: Comparison
    rhs: int
