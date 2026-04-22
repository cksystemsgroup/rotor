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


@dataclass(frozen=True)
class EquivalenceSpec(QuestionSpec):
    """are_equivalent obligation: product construction over two
    function bodies sharing an initial register state.

    Two copies of the machine model run in parallel in the same
    BTOR2 Model (state-prefixed `a_` and `b_`). Their cycle-0
    registers are constrained to the same symbolic input values —
    so the question is "do both sides produce the same output
    register at their respective return sites, given identical
    inputs?"

        bad = (pc_a at any ret in fn_a)
            ∧ (pc_b at any ret in fn_b)
            ∧ (regs_a[output_register] ≠ regs_b[output_register])

    - `reachable` → an input makes the two sides disagree; the
      CEX is in initial_regs (prefixed with `a_` / `b_` by the
      solver, but since arg regs are equated their values match
      on both sides).
    - `unreachable` (bounded) → no disagreement up to the BMC
      bound. Not a global equivalence proof; a larger bound or
      Spacer would be needed for that.
    - `proved` (unbounded, future) → sides are equivalent on
      every input.

    Current scope: leaf functions only (no shared-memory story
    yet). `fn_a` and `fn_b` must each contain at least one `ret`.
    The two binaries must not have overlapping PC ranges for
    their functions, otherwise the ra-outside-fn entry assumptions
    conflict.
    """
    binary_a_path: str
    function_a: str
    binary_b_path: str
    function_b: str
    output_register: int = 10                       # a0 by default


@dataclass(frozen=True)
class FindInputSpec(QuestionSpec):
    """find_input obligation: synthesize an initial-register
    assignment such that `regs[register] OP rhs` holds at a return
    site of `function`.

    Shape is identical to VerifySpec; the only difference is
    polarity — the emitter does NOT negate the predicate:

        bad = (pc at any ret in fn) ∧ (regs[register] OP rhs)

    So `reachable` means "the predicate is achievable — initial_regs
    constitute a witness input to the function"; `unreachable`
    (bounded) or `proved` (unbounded via Spacer) means "no input
    satisfies the predicate at a return site within the given
    bound / on any execution path".

    Equivalent to `VerifySpec` with the comparison negated, but
    exposed as a separate spec so the API and CLI can present the
    verdict with its natural find-input interpretation.
    """
    function: str
    register: int
    comparison: Comparison
    rhs: int
