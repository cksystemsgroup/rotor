"""High-level entry point for rotor.

M1 exposes only can_reach. Additional verbs (find_input, verify,
are_equivalent) land in later milestones under the same architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from rotor.binary import RISCVBinary
from rotor.instance import RotorInstance
from rotor.solvers.base import SolverBackend, SolverResult


@dataclass(frozen=True)
class ReachResult:
    verdict: str                       # "reachable" | "unreachable" | "unknown"
    bound: int
    step: Optional[int]
    initial_regs: dict[str, int]
    elapsed: float
    backend: str


class RotorAPI:
    def __init__(
        self,
        binary_path: Union[str, Path],
        *,
        default_bound: int = 20,
        backend: Optional[SolverBackend] = None,
    ) -> None:
        self._binary_path = Path(binary_path)
        self._binary = RISCVBinary(self._binary_path)
        self._default_bound = default_bound
        self._backend = backend

    @property
    def binary(self) -> RISCVBinary:
        return self._binary

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
        instance = RotorInstance.for_reach(
            self._binary, function, target_pc, backend=self._backend
        )
        result: SolverResult = instance.check(
            bound=bound if bound is not None else self._default_bound,
            timeout=timeout,
        )
        return ReachResult(
            verdict=result.verdict,
            bound=result.bound,
            step=result.step,
            initial_regs=result.initial_regs,
            elapsed=result.elapsed,
            backend=result.backend,
        )
