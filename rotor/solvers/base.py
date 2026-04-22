"""Solver backend interface.

Backends are pluggable. Under this same Protocol rotor ships:

    Z3BMC         — bounded BMC via Z3                      (always)
    Z3Spacer      — unbounded PDR via Z3                    (always)
    BitwuzlaBMC   — bounded BMC via Bitwuzla                (pip install bitwuzla)
    CVC5BMC       — bounded BMC via CVC5                    (pip install cvc5)
    Pono          — multi-engine subprocess bridge
                    (bmc / ind / mbic3 / ic3ia / ic3sa / interp)   (build from source)

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
