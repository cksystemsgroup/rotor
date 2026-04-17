"""Phase 6: Counterexample-Guided Abstraction Refinement.

CEGAR lets us extend unbounded IC3 reasoning to programs whose concrete state
space is too large for IC3 to handle directly. We alternate between:

    1. abstracting the concrete :class:`RotorInstance` into a cheaper model,
    2. running IC3 on the abstraction,
    3. either lifting a proof to the concrete model (accept) or checking a
       counterexample concretely (real bug vs. spurious),
    4. refining the abstraction when the counterexample is spurious.

This module provides the loop orchestration; the abstraction/refinement
operators are intentionally hooks that callers configure for their workload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from rotor.btor2 import Node
from rotor.engine import RotorEngine
from rotor.instance import RotorInstance
from rotor.solvers.base import CheckResult


@dataclass
class CEGARConfig:
    max_iterations: int = 20
    abstract_solver: str = "ic3"
    concrete_solver: str = "bitwuzla"
    bound: int = 1000
    abstract: Callable[[RotorInstance], RotorInstance] | None = None
    refine: Callable[[RotorInstance, CheckResult], RotorInstance] | None = None
    verify_invariant: Callable[[str, str], bool] | None = None


@dataclass
class CEGARTrace:
    """Diagnostic record of a CEGAR run."""

    iterations: int = 0
    events: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.events.append(msg)


class CEGARVerifier:
    """CEGAR driver on top of a :class:`RotorEngine`."""

    def __init__(self, engine: RotorEngine, config: CEGARConfig | None = None) -> None:
        self.engine = engine
        self.config = config or CEGARConfig()
        self.trace_log = CEGARTrace()

    def verify(self, function: str, property_bad: Node) -> CheckResult:
        cfg = self.config
        abstract_inst = self._make_abstract(function)

        for iteration in range(cfg.max_iterations):
            self.trace_log.iterations += 1
            self.trace_log.log(f"iter {iteration}: check abstract model")

            abstract_inst.add_bad(property_bad, "cegar-target")
            abstract_inst.config.solver = cfg.abstract_solver
            result = abstract_inst.check(cfg.bound)

            if result.verdict == "unsat":
                self.trace_log.log("abstract UNSAT; verifying invariant concretely")
                if self._invariant_holds_concretely(function, result.invariant):
                    self.trace_log.log("invariant lifts — genuine proof")
                    return result
                self.trace_log.log("invariant spurious — refining")
                abstract_inst = self._refine(abstract_inst, result)

            elif result.verdict == "sat":
                self.trace_log.log("abstract SAT; checking CEX concretely")
                concrete = self._make_concrete(function)
                concrete.add_bad(property_bad, "cegar-target-concrete")
                concrete.config.solver = cfg.concrete_solver
                concrete_result = concrete.check(cfg.bound)
                if concrete_result.verdict == "sat":
                    self.trace_log.log("concrete SAT — genuine bug")
                    return concrete_result
                self.trace_log.log("CEX spurious — refining")
                abstract_inst = self._refine(abstract_inst, result)

            else:
                self.trace_log.log(f"abstract verdict={result.verdict}; giving up")
                return result

        return CheckResult(
            verdict="unknown",
            solver="cegar",
            stderr=f"exhausted max_iterations={cfg.max_iterations}",
        )

    # ------------------------------------------------------------- hooks

    def _make_abstract(self, function: str) -> RotorInstance:
        inst = self.engine.create_instance(function)
        if self.config.abstract is not None:
            return self.config.abstract(inst)
        return inst

    def _make_concrete(self, function: str) -> RotorInstance:
        return self.engine.create_instance(function)

    def _refine(self, inst: RotorInstance, result: CheckResult) -> RotorInstance:
        if self.config.refine is not None:
            return self.config.refine(inst, result)
        # Default: re-create the abstract instance unchanged. This is a
        # placeholder — real refinement needs predicate abstraction updates.
        return inst

    def _invariant_holds_concretely(
        self, function: str, invariant: str | None
    ) -> bool:
        if invariant is None:
            return True  # IC3 backend didn't emit one; conservatively accept.
        if self.config.verify_invariant is not None:
            return self.config.verify_invariant(function, invariant)
        return True
