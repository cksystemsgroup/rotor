"""High-level entry point for rotor.

M2 exposes can_reach with a source-lifted Trace on reachable verdicts.
Additional verbs (find_input, verify, are_equivalent) land later under
the same architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from rotor.binary import RISCVBinary
from rotor.dwarf import DwarfLineMap
from rotor.engine import EngineConfig, RotorEngine
from rotor.solvers.base import SolverBackend
from rotor.solvers.portfolio import Portfolio
from rotor.trace import Trace, build_trace


@dataclass(frozen=True)
class ReachResult:
    verdict: str                       # "reachable" | "unreachable" | "unknown"
    bound: int
    step: Optional[int]
    initial_regs: dict[str, int]
    elapsed: float
    backend: str
    trace: Optional[Trace] = None      # populated when verdict == "reachable"


class RotorAPI:
    def __init__(
        self,
        binary_path: Union[str, Path],
        *,
        default_bound: int = 20,
        backend: Optional[SolverBackend] = None,
        portfolio: Optional[Portfolio] = None,
    ) -> None:
        self._binary_path = Path(binary_path)
        self._binary = RISCVBinary(self._binary_path)
        self._dwarf = DwarfLineMap(self._binary_path)
        self._engine = RotorEngine(
            self._binary,
            config=EngineConfig(
                backend=backend,
                portfolio=portfolio,
                default_bound=default_bound,
            ),
        )

    @property
    def binary(self) -> RISCVBinary:
        return self._binary

    @property
    def engine(self) -> RotorEngine:
        return self._engine

    def close(self) -> None:
        self._binary.close()

    def __enter__(self) -> "RotorAPI":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def can_reach(
        self,
        function: str,
        target_pc: int,
        bound: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> ReachResult:
        result = self._engine.check_reach(function, target_pc, bound=bound, timeout=timeout)
        trace = None
        if result.verdict == "reachable":
            trace = build_trace(
                binary=self._binary,
                function=function,
                target_pc=target_pc,
                verdict=result.verdict,
                bound=result.bound,
                reached_at=result.step,
                elapsed=result.elapsed,
                backend=result.backend,
                initial_regs=result.initial_regs,
                dwarf=self._dwarf,
            )
        return ReachResult(
            verdict=result.verdict,
            bound=result.bound,
            step=result.step,
            initial_regs=result.initial_regs,
            elapsed=result.elapsed,
            backend=result.backend,
            trace=trace,
        )
