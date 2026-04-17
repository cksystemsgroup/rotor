"""Solver-backend abstract base class and result type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckResult:
    """Outcome of a BMC/IC3 check.

    ``verdict`` is one of ``'sat'``, ``'unsat'``, ``'unknown'``.
    On ``sat`` the solver provides ``steps`` (BMC depth at which a bad
    property was reached) and ``witness`` (a list of per-step frames of
    concrete state). On ``unsat`` an IC3-class solver additionally provides
    an inductive ``invariant`` as a BTOR2 textual clause set.
    """

    verdict: str
    steps: int | None = None
    witness: list[dict[str, Any]] | None = None
    invariant: str | None = None
    solver: str = ""
    elapsed: float = 0.0
    stdout: str = ""
    stderr: str = ""
    branch_points: list[Any] = field(default_factory=list)

    def is_conclusive(self) -> bool:
        return self.verdict in ("sat", "unsat")


class SolverBackend(ABC):
    """Abstract backend that takes BTOR2 text and produces a :class:`CheckResult`."""

    name: str = "base"

    @abstractmethod
    def check(self, btor2: str, bound: int) -> CheckResult:
        """Run the backend on ``btor2`` text up to ``bound`` steps."""

    def supports_unbounded(self) -> bool:
        """Does this backend return UNSAT proofs beyond a fixed ``bound``?"""
        return False
