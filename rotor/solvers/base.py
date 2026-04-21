"""Solver backend interface.

Backends are pluggable: Z3 (in-process BMC) is the shipping engine;
additional engines (Z3 Spacer for unbounded PDR, subprocess bridges to
Bitwuzla / BtorMC / rIC3 / AVR) land under this same Protocol.

Verdict semantics:

    reachable    — concrete counterexample exists. Globally conclusive.
    unreachable  — safe up to `bound`. Bounded engines (BMC) can only
                   ever return this; a larger bound could still find
                   a bug.
    proved       — safe for all bounds. Only unbounded engines (IC3,
                   k-induction that closes) return this. `invariant`
                   carries an optional certificate.
    unknown      — timeout or solver-internal failure.

Both `reachable` and `proved` are globally conclusive; portfolios
short-circuit on either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

from rotor.btor2.nodes import Model

Verdict = Literal["reachable", "unreachable", "proved", "unknown"]


@dataclass(frozen=True)
class SolverResult:
    verdict: Verdict
    bound: int
    step: Optional[int] = None                    # step at which bad held, if reachable
    initial_regs: dict[str, int] = field(default_factory=dict)
    elapsed: float = 0.0
    backend: str = ""
    reason: Optional[str] = None                  # for "unknown"
    invariant: Optional[str] = None               # certificate for "proved"


class SolverBackend(Protocol):
    name: str

    def check_reach(
        self,
        model: Model,
        bound: int,
        timeout: Optional[float] = None,
    ) -> SolverResult: ...
