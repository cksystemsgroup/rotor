"""Orchestration: compile a question to BTOR2, dispatch to a backend.

RotorEngine owns a RISCVBinary plus a backend choice. Callers invoke
one method per question (`check_reach`, future `check_verify` / ...).
The engine compiles the question to a BTOR2 Model and hands it to a
single backend or a Portfolio racer.

RotorAPI is a thin wrapper on top that adds default bounds, DWARF
lookup, and witness-to-trace rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from rotor.binary import RISCVBinary
from rotor.btor2.builder import ReachSpec, build_reach
from rotor.solvers.base import SolverBackend, SolverResult
from rotor.solvers.portfolio import Portfolio
from rotor.solvers.z3bv import Z3BMC


@dataclass
class EngineConfig:
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
        model = build_reach(self.binary, spec)
        executor = self._executor()
        if isinstance(executor, Portfolio):
            return executor.check_reach(model)
        return executor.check_reach(
            model,
            bound=bound if bound is not None else self.config.default_bound,
            timeout=timeout if timeout is not None else self.config.default_timeout,
        )
