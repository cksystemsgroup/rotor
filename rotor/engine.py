"""Orchestration: compile a question to BTOR2 via an emitter, dispatch to a backend.

RotorEngine owns a RISCVBinary plus three choices:

    emitter_factory  — which IR produces the BTOR2 Model (default: IdentityEmitter).
    backend          — which single solver answers the obligation.
    portfolio        — alternative to `backend`: race multiple solvers.

Callers invoke one method per question (`check_reach`, future
`check_verify` / ...). The engine compiles the question through the
emitter seam and hands the Model to the chosen executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

from rotor.binary import RISCVBinary
from rotor.ir.emitter import BTOR2Emitter, IdentityEmitter
from rotor.ir.spec import Comparison, FindInputSpec, ReachSpec, VerifySpec
from rotor.solvers.base import SolverBackend, SolverResult
from rotor.solvers.portfolio import Portfolio
from rotor.solvers.z3bv import Z3BMC
from rotor.solvers.z3spacer import Z3Spacer

# cegar imported after rotor.ir to avoid a circular-import race:
# cegar → rotor.btor2.builder → rotor.ir.spec, while rotor.ir.__init__
# also triggers rotor.btor2.builder via ir.emitter. Loading rotor.ir
# first ensures rotor.ir.spec is in sys.modules before the builder
# cycle kicks in.
from rotor.cegar import CegarConfig, cegar_reach


@dataclass
class EngineConfig:
    emitter_factory: Callable[[RISCVBinary], BTOR2Emitter] = IdentityEmitter
    backend: Optional[SolverBackend] = None
    portfolio: Optional[Portfolio] = None
    default_bound: int = 20
    default_timeout: Optional[float] = None


class RotorEngine:
    def __init__(
        self,
        binary: RISCVBinary,
        config: Optional[EngineConfig] = None,
    ) -> None:
        self.binary = binary
        self.config = config or EngineConfig()
        self._emitter: BTOR2Emitter = self.config.emitter_factory(binary)

    @property
    def emitter(self) -> BTOR2Emitter:
        return self._emitter

    def _executor(self) -> Union[SolverBackend, Portfolio]:
        if self.config.portfolio is not None:
            return self.config.portfolio
        return self.config.backend or Z3BMC()

    def check_reach(
        self,
        function: str,
        target_pc: int,
        bound: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> SolverResult:
        spec = ReachSpec(function=function, target_pc=target_pc)
        model = self._emitter.emit(spec)
        executor = self._executor()
        if isinstance(executor, Portfolio):
            return executor.check_reach(model)
        return executor.check_reach(
            model,
            bound=bound if bound is not None else self.config.default_bound,
            timeout=timeout if timeout is not None else self.config.default_timeout,
        )

    def check_reach_unbounded(
        self,
        function: str,
        target_pc: int,
        timeout: Optional[float] = None,
    ) -> SolverResult:
        """Unbounded reachability via Z3 Spacer (PDR/IC3).

        Ignores the engine's configured backend / portfolio / bound —
        this is an explicit PDR call. Returns `proved` with invariant,
        `reachable` with no step info (Spacer's Python API doesn't
        surface CEX traces), or `unknown`.
        """
        spec = ReachSpec(function=function, target_pc=target_pc)
        model = self._emitter.emit(spec)
        return Z3Spacer().check_reach(
            model,
            bound=0,
            timeout=timeout if timeout is not None else self.config.default_timeout,
        )

    def check_verify(
        self,
        function: str,
        register: int,
        comparison: Comparison,
        rhs: int,
        bound: Optional[int] = None,
        timeout: Optional[float] = None,
        unbounded: bool = False,
    ) -> SolverResult:
        """Verify that `regs[register] <comparison> rhs` holds at every
        return site of `function`.

        Bounded by default (BMC). `unbounded=True` routes to Z3 Spacer
        for an inductive-invariant answer. Both modes return `proved`
        when the predicate holds on every path (Spacer; BMC answers
        `unreachable` up to the bound instead), `reachable` when the
        predicate can fail, or `unknown`.
        """
        spec = VerifySpec(
            function=function, register=register,
            comparison=comparison, rhs=rhs,
        )
        model = self._emitter.emit(spec)
        if unbounded:
            return Z3Spacer().check_reach(
                model, bound=0,
                timeout=timeout if timeout is not None else self.config.default_timeout,
            )
        executor = self._executor()
        if isinstance(executor, Portfolio):
            return executor.check_reach(model)
        return executor.check_reach(
            model,
            bound=bound if bound is not None else self.config.default_bound,
            timeout=timeout if timeout is not None else self.config.default_timeout,
        )

    def check_find_input(
        self,
        function: str,
        register: int,
        comparison: Comparison,
        rhs: int,
        bound: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> SolverResult:
        """Synthesize an initial-register assignment such that
        `regs[register] <comparison> rhs` holds at a return site of
        `function`.

        Bounded BMC only — unbounded PDR on find_input doesn't have
        a useful interpretation (Spacer's `proved` verdict would mean
        "no input achieves the predicate on any execution", which is
        a verify question in disguise — use `check_verify(...,
        unbounded=True)` with the negated comparison if that's the
        question you want).
        """
        spec = FindInputSpec(
            function=function, register=register,
            comparison=comparison, rhs=rhs,
        )
        model = self._emitter.emit(spec)
        executor = self._executor()
        if isinstance(executor, Portfolio):
            return executor.check_reach(model)
        return executor.check_reach(
            model,
            bound=bound if bound is not None else self.config.default_bound,
            timeout=timeout if timeout is not None else self.config.default_timeout,
        )

    def check_reach_cegar(
        self,
        function: str,
        target_pc: int,
        config: Optional[CegarConfig] = None,
    ) -> SolverResult:
        """CEGAR-driven reachability.

        Starts with every register havoc'd; iteratively refines the
        abstraction using Z3Spacer + concrete witness replay until
        converged. Bypasses the configured emitter — CEGAR drives
        build_reach directly so it can vary havoc_regs per iteration.
        """
        spec = ReachSpec(function=function, target_pc=target_pc)
        return cegar_reach(self.binary, spec, config)
