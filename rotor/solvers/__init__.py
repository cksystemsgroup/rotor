"""Solver backends. Each backend consumes a BTOR2 Model and answers
reach/verify/... questions. M1 ships one BMC backend built on Z3."""

from rotor.solvers.base import SolverBackend, SolverResult, Verdict
from rotor.solvers.portfolio import Portfolio, PortfolioEntry
from rotor.solvers.z3bv import Z3BMC

__all__ = ["SolverBackend", "SolverResult", "Verdict", "Z3BMC", "Portfolio", "PortfolioEntry"]
