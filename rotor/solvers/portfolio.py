"""Parallel solver portfolio.

Runs several :class:`SolverBackend` instances concurrently on the same BTOR2
input and returns the first conclusive result. If all backends return
``unknown``, the portfolio returns ``unknown`` with the aggregated stderr.
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import Iterable

from rotor.solvers.base import CheckResult, SolverBackend


class PortfolioSolver(SolverBackend):
    """Run multiple backends in parallel; return the first conclusive result."""

    name = "portfolio"

    def __init__(
        self,
        config: object | None = None,
        backends: Iterable[SolverBackend] | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.config = config
        self._backends = list(backends or [])
        self.timeout = timeout

    def add(self, backend: SolverBackend) -> None:
        self._backends.append(backend)

    def supports_unbounded(self) -> bool:
        return any(b.supports_unbounded() for b in self._backends)

    def check(self, btor2: str, bound: int) -> CheckResult:
        if not self._backends:
            return CheckResult(
                verdict="unknown",
                solver=self.name,
                elapsed=0.0,
                stderr="portfolio: no backends configured",
            )

        start = time.monotonic()
        agg_stderr: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(self._backends)
        ) as pool:
            future_to_backend = {
                pool.submit(backend.check, btor2, bound): backend
                for backend in self._backends
            }
            try:
                for future in concurrent.futures.as_completed(
                    future_to_backend, timeout=self.timeout
                ):
                    result = future.result()
                    agg_stderr.append(f"[{result.solver}] {result.stderr}")
                    if result.is_conclusive():
                        for other in future_to_backend:
                            if other is not future:
                                other.cancel()
                        result.elapsed = time.monotonic() - start
                        return result
            except concurrent.futures.TimeoutError:
                pass

        return CheckResult(
            verdict="unknown",
            solver=self.name,
            elapsed=time.monotonic() - start,
            stderr="\n".join(agg_stderr),
        )
