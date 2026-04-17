"""Bitwuzla BMC backend.

Bitwuzla is an SMT solver; it does not understand BTOR2's sequential
extensions (``state``, ``init``, ``next``, ``bad``, ``constraint``). This
backend works around that by parsing the BTOR2 text into a
:class:`~rotor.btor2.NodeDAG`, then feeding the DAG to
:class:`~rotor.solvers.bmc.BitwuzlaUnroller`, which materializes each state's
value at every BMC step as a fresh Bitwuzla term and queries the solver at
each depth.
"""

from __future__ import annotations

import time

from rotor.solvers.base import CheckResult, SolverBackend


class BitwuzlaSolver(SolverBackend):
    """BMC backend backed by Bitwuzla's Python API via incremental unrolling."""

    name = "bitwuzla"

    def __init__(self, config: object | None = None) -> None:
        self.config = config

    def supports_unbounded(self) -> bool:
        return False

    def check(self, btor2: str, bound: int) -> CheckResult:
        start = time.monotonic()
        try:
            import bitwuzla  # noqa: F401
        except ImportError:
            return CheckResult(
                verdict="unknown",
                solver=self.name,
                elapsed=time.monotonic() - start,
                stderr="bitwuzla package not installed",
            )

        from rotor.btor2 import parse_btor2
        from rotor.solvers.bmc import BitwuzlaUnroller

        try:
            dag = parse_btor2(btor2)
        except Exception as err:
            return CheckResult(
                verdict="unknown",
                solver=self.name,
                elapsed=time.monotonic() - start,
                stderr=f"BTOR2 parse failure: {err}",
            )

        unroller = BitwuzlaUnroller(dag)
        result = unroller.check(bound)
        # Preserve our own timing since the unroller reports its own.
        result.elapsed = time.monotonic() - start
        return result
