"""Solver backends. Each backend consumes a BTOR2 Model and answers
reach/verify/... questions.

    Z3BMC        — bounded BMC via Z3:         `reachable` | `unreachable` | `unknown`
    BitwuzlaBMC  — bounded BMC via Bitwuzla:   typically 3–5× faster on BV workloads
    Z3Spacer     — unbounded PDR via Z3:       `reachable` | `proved` | `unknown`
"""

from rotor.solvers.base import SolverBackend, SolverResult, Verdict
from rotor.solvers.bitwuzla import BitwuzlaBMC
from rotor.solvers.portfolio import Portfolio, PortfolioEntry
from rotor.solvers.z3bv import Z3BMC
from rotor.solvers.z3spacer import Z3Spacer

__all__ = [
    "SolverBackend",
    "SolverResult",
    "Verdict",
    "Z3BMC",
    "Z3Spacer",
    "BitwuzlaBMC",
    "Portfolio",
    "PortfolioEntry",
]
