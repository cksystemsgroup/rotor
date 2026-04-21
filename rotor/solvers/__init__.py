"""Solver backends. Each backend consumes a BTOR2 Model and answers
reach/verify/... questions.

    Z3BMC        — bounded BMC via Z3:         `reachable` | `unreachable` | `unknown`
    BitwuzlaBMC  — bounded BMC via Bitwuzla:   typically 3–5× faster on BV workloads
    Z3Spacer     — unbounded PDR via Z3:       `reachable` | `proved` | `unknown`
    Ric3         — subprocess bridge to rIC3 (Rust PDR);      unbounded
    BtorMC       — subprocess bridge to BtorMC (BMC/k-ind);   both modes

Subprocess bridges (Ric3, BtorMC) report `unknown` with a clear
reason when the external binary is not on PATH, so portfolios that
include them work in environments where only a subset of the tools
are installed.
"""

from typing import Optional

from rotor.solvers.base import SolverBackend, SolverResult, Verdict
from rotor.solvers.bitwuzla import BitwuzlaBMC
from rotor.solvers.btormc import BtorMC
from rotor.solvers.portfolio import Portfolio, PortfolioEntry
from rotor.solvers.ric3 import Ric3
from rotor.solvers.z3bv import Z3BMC
from rotor.solvers.z3spacer import Z3Spacer


def default_portfolio(bound: int = 20, timeout: Optional[float] = 30.0) -> Portfolio:
    """Build a portfolio that races every available solver backend.

    Always-available (in-process Python): Z3BMC, Z3Spacer, plus
    BitwuzlaBMC when the `bitwuzla` Python package is importable.
    Subprocess bridges (Ric3, BtorMC) are added if their binary is
    on PATH — otherwise omitted, since including a guaranteed-
    `unknown` racer would dilute the portfolio's verdict strength.

    `bound` applies to the bounded engines; unbounded engines
    (Z3Spacer, Ric3) ignore it. `timeout` is per-entry.
    """
    p = Portfolio()
    p.add(Z3BMC(), bound=bound, timeout=timeout)
    p.add(Z3Spacer(), bound=0, timeout=timeout)
    try:
        import bitwuzla                              # noqa: F401
        p.add(BitwuzlaBMC(), bound=bound, timeout=timeout)
    except ImportError:                              # pragma: no cover
        pass
    ric3 = Ric3()
    if ric3.available:
        p.add(ric3, bound=0, timeout=timeout)
    btormc = BtorMC()
    if btormc.available:
        p.add(btormc, bound=bound, timeout=timeout)
    return p


__all__ = [
    "SolverBackend",
    "SolverResult",
    "Verdict",
    "Z3BMC",
    "Z3Spacer",
    "BitwuzlaBMC",
    "Ric3",
    "BtorMC",
    "Portfolio",
    "PortfolioEntry",
    "default_portfolio",
]
