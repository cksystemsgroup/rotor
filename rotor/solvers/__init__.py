"""Solver backends.

Each backend implements :class:`~rotor.solvers.base.SolverBackend`. The
:func:`make_solver` factory returns an instance for a given solver name.
"""

from __future__ import annotations

from rotor.solvers.base import CheckResult, SolverBackend
from rotor.solvers.bitwuzla import BitwuzlaSolver
from rotor.solvers.btormc import BtorMCSolver
from rotor.solvers.ic3 import IC3Solver
from rotor.solvers.portfolio import PortfolioSolver


_BACKENDS = {
    "bitwuzla": BitwuzlaSolver,
    "btormc": BtorMCSolver,
    "ic3": IC3Solver,
    "portfolio": PortfolioSolver,
}


def make_solver(name: str, **kwargs: object) -> SolverBackend:
    """Instantiate a solver backend by name."""
    try:
        cls = _BACKENDS[name]
    except KeyError as err:
        raise ValueError(
            f"Unknown solver {name!r}; available: {sorted(_BACKENDS)}"
        ) from err
    return cls(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "CheckResult",
    "SolverBackend",
    "BitwuzlaSolver",
    "BtorMCSolver",
    "IC3Solver",
    "PortfolioSolver",
    "make_solver",
]
