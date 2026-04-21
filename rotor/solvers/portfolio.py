"""Parallel race over (backend, bound) configurations.

Verdict semantics constrain how the race resolves:

    reachable     — globally conclusive (a concrete counterexample exists).
    proved        — globally conclusive (safe for all bounds, with
                    optional invariant certificate).
    unreachable   — only means "safe up to bound k"; a larger bound
                    could still find a bug. Not globally conclusive.
    unknown       — weakest; timeout or solver-internal failure.

The portfolio short-circuits on the first globally-conclusive result
(`reachable` or `proved`). `reachable` wins over `proved` if both
arrive, since a counterexample is a stronger form of evidence than a
certificate for the user: it is concrete. If no globally-conclusive
verdict appears, the portfolio waits for every config to settle and
returns the deepest `unreachable` (the strongest safe-up-to claim
available). If nothing conclusive remains, the result is `unknown`.

Thread cancellation in Python is best-effort: a running Z3 check
cannot be preempted. `future.cancel()` only affects configs that have
not yet started; timeouts on individual solvers are the principal
control.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Optional

from rotor.btor2.nodes import Model
from rotor.solvers.base import SolverBackend, SolverResult


@dataclass(frozen=True)
class PortfolioEntry:
    backend: SolverBackend
    bound: int
    timeout: Optional[float] = None


@dataclass
class Portfolio:
    entries: list[PortfolioEntry] = field(default_factory=list)
    max_workers: Optional[int] = None

    name = "portfolio"

    def add(self, backend: SolverBackend, bound: int, timeout: Optional[float] = None) -> "Portfolio":
        self.entries.append(PortfolioEntry(backend=backend, bound=bound, timeout=timeout))
        return self

    def check_reach(self, model: Model) -> SolverResult:
        if not self.entries:
            raise ValueError("portfolio is empty")

        pool = ThreadPoolExecutor(max_workers=self.max_workers or len(self.entries))
        futures = {
            pool.submit(e.backend.check_reach, model, e.bound, e.timeout): e
            for e in self.entries
        }

        collected: list[SolverResult] = []
        pending = set(futures)
        conclusive: Optional[SolverResult] = None
        try:
            while pending and conclusive is None:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    try:
                        result = fut.result()
                    except Exception as exc:                  # pragma: no cover
                        entry = futures[fut]
                        result = SolverResult(
                            verdict="unknown",
                            bound=entry.bound,
                            backend=entry.backend.name,
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                    collected.append(result)
                    if result.verdict in ("reachable", "proved"):
                        conclusive = result
                        break
        finally:
            for fut in pending:
                fut.cancel()
            pool.shutdown(wait=False, cancel_futures=True)

        if conclusive is not None:
            return conclusive

        unreachable = [r for r in collected if r.verdict == "unreachable"]
        if unreachable:
            return max(unreachable, key=lambda r: r.bound)

        assert collected, "portfolio must produce at least one result"
        return collected[0]
