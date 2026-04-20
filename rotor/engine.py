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
from rotor.ir.spec import ReachSpec
from rotor.solvers.base import SolverBackend, SolverResult
from rotor.solvers.portfolio import Portfolio
from rotor.solvers.z3bv import Z3BMC


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
