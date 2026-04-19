"""Solver backend interface.

Backends are pluggable: Z3 (in-process) for M1, subprocess bridges to
Bitwuzla / BtorMC / rIC3 / AVR later. All backends consume a BTOR2
Model and return a SolverResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

from rotor.btor2.nodes import Model

Verdict = Literal["reachable", "unreachable", "unknown"]


@dataclass(frozen=True)
class SolverResult:
    verdict: Verdict
    bound: int
    step: Optional[int] = None                    # step at which bad held, if reachable
    initial_regs: dict[str, int] = field(default_factory=dict)
    elapsed: float = 0.0
    backend: str = ""
    reason: Optional[str] = None                  # for "unknown"


class SolverBackend(Protocol):
    name: str

    def check_reach(
        self,
        model: Model,
        bound: int,
        timeout: Optional[float] = None,
    ) -> SolverResult: ...
