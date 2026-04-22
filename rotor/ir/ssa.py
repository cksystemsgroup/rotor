"""L2: goal-directed slicing emitter.

`SsaEmitter` is rotor's third BTOR2Emitter — after `IdentityEmitter`
(raw L0) and `DagEmitter` (L1 hash-consing + local simplification).
Its job is to shrink the state-variable set the BTOR2 Model carries,
so unbounded engines (Z3 Spacer today; future IC3 bridges) can close
invariants on a smaller PDR state space.

The mechanism is `rotor.ir.liveness`: a flow-insensitive backward
analysis that classifies each register as either *live for
reachability* (its value can affect whether `bad` ever holds) or
*dead* (provably irrelevant). Dead registers are then passed as
`havoc_regs` to `build_reach`, which replaces them with per-cycle
BTOR2 inputs — the 6.4a primitive originally built for CEGAR.

Soundness: dropping a dead register is safe — its value never flows
into `bad` (by definition of dead), so havoc'ing it cannot change
the set of reachable `bad`-states. The L0-equivalence harness
confirms this per-verdict and per-step across the corpus.

Scope note: this is SSA in name rather than in full machinery.
PLAN.md's M8 description anticipates per-instruction defs, φ-nodes
at joins, and a dominator tree; those become useful for slicing
*below* the register granularity (e.g. dropping instructions that
only write dead registers) and for future IC3 predicate inference.
The L0-equivalence harness asserts BMC step counts match L0 exactly,
which rules out instruction-level slicing at this layer — register
liveness is the deepest slice the contract admits. Richer SSA
structure lands when a consumer demands it.
"""

from __future__ import annotations

from rotor.binary import RISCVBinary
from rotor.btor2.builder import build_find_input, build_reach, build_verify
from rotor.btor2.nodes import Model
from rotor.ir.dag import DagBuilder
from rotor.ir.liveness import dead_registers
from rotor.ir.spec import FindInputSpec, QuestionSpec, ReachSpec, VerifySpec


class SsaEmitter:
    """L2 emitter: L1 (hash-consing + simplification) plus liveness-sliced
    register state.

    SsaEmitter is strictly layered *on top of* DagEmitter's hash-consed
    builder — dropping the hash-consing would lose the per-input and
    per-constant sharing that keeps the sliced model actually smaller.
    The only new ingredient versus DagEmitter is the `havoc_regs` set
    derived from `rotor.ir.liveness`.
    """
    name = "ssa"

    def __init__(self, binary: RISCVBinary) -> None:
        self._binary = binary

    @property
    def binary(self) -> RISCVBinary:
        return self._binary

    def emit(self, spec: QuestionSpec) -> Model:
        fn_name = getattr(spec, "function", None)
        if fn_name is None:
            raise TypeError(
                f"{type(self).__name__} does not support spec type "
                f"{type(spec).__name__}"
            )
        fn = self._binary.function(fn_name)
        dead = set(dead_registers(self._binary, fn))
        if isinstance(spec, ReachSpec):
            return build_reach(
                self._binary, spec,
                builder=DagBuilder(), havoc_regs=dead,
            )
        if isinstance(spec, VerifySpec):
            # The verify predicate reads spec.register at return sites,
            # so that register must stay live even if the function itself
            # never branches on it.
            dead.discard(spec.register)
            return build_verify(
                self._binary, spec,
                builder=DagBuilder(), havoc_regs=dead,
            )
        if isinstance(spec, FindInputSpec):
            # Same carve-out as VerifySpec — the find_input predicate
            # also reads spec.register at return sites.
            dead.discard(spec.register)
            return build_find_input(
                self._binary, spec,
                builder=DagBuilder(), havoc_regs=dead,
            )
        raise TypeError(
            f"{type(self).__name__} does not support spec type "
            f"{type(spec).__name__}"
        )
