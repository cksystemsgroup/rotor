"""Solver backends. Each backend consumes a BTOR2 Model and answers
reach/verify/... questions.

    Z3BMC     — bounded BMC: `reachable` | `unreachable` | `unknown`
    Z3Spacer  — unbounded PDR: `reachable` | `proved` | `unknown`
"""

from rotor.solvers.base import SolverBackend, SolverResult, Verdict
from rotor.solvers.portfolio import Portfolio, PortfolioEntry
from rotor.solvers.z3bv import Z3BMC
from rotor.solvers.z3spacer import Z3Spacer

__all__ = [
    "SolverBackend",
    "SolverResult",
    "Verdict",
    "Z3BMC",
    "Z3Spacer",
    "Portfolio",
    "PortfolioEntry",
]
