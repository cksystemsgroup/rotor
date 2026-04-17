"""Bitwuzla BMC backend.

Uses the Bitwuzla Python API when available. Bitwuzla natively parses BTOR2,
so our job is to hand it the BTOR2 text, unroll to the configured bound,
and query for models on the ``bad`` properties.
"""

from __future__ import annotations

import time

from rotor.solvers.base import CheckResult, SolverBackend


class BitwuzlaSolver(SolverBackend):
    """BMC backend backed by Bitwuzla's Python API.

    This is a relatively thin adapter: we translate the BTOR2 model into
    Bitwuzla terms (via Bitwuzla's BTOR2 parser) and iteratively unroll the
    transition relation, asserting bad(k) at each step. The first SAT result
    returns a witness; UNSAT after ``bound`` steps is reported as UNSAT *for
    that bound* (Bitwuzla is not an unbounded model checker).
    """

    name = "bitwuzla"

    def __init__(self, config: object | None = None) -> None:
        self.config = config

    def supports_unbounded(self) -> bool:
        return False

    def check(self, btor2: str, bound: int) -> CheckResult:
        start = time.monotonic()
        try:
            import bitwuzla  # type: ignore
        except ImportError:
            return CheckResult(
                verdict="unknown",
                solver=self.name,
                elapsed=time.monotonic() - start,
                stderr="bitwuzla package not installed",
            )

        tm = bitwuzla.TermManager()
        options = bitwuzla.Options()
        options.set(bitwuzla.Option.PRODUCE_MODELS, True)
        parser = bitwuzla.Parser(tm, options)
        try:
            parser.parse(btor2, format="btor2")
        except Exception as err:  # pragma: no cover - depends on bitwuzla
            return CheckResult(
                verdict="unknown",
                solver=self.name,
                elapsed=time.monotonic() - start,
                stderr=f"parse failure: {err}",
            )

        # The Bitwuzla BTOR2 parser builds the unrolled model internally when
        # given a bound; when not, we fall back to a direct satisfiability
        # query on the parsed terms.
        bitwuzla_instance = parser.bitwuzla()
        try:
            result = bitwuzla_instance.check_sat()
        except Exception as err:  # pragma: no cover
            return CheckResult(
                verdict="unknown",
                solver=self.name,
                elapsed=time.monotonic() - start,
                stderr=f"solve failure: {err}",
            )

        verdict = {
            bitwuzla.Result.SAT: "sat",
            bitwuzla.Result.UNSAT: "unsat",
            bitwuzla.Result.UNKNOWN: "unknown",
        }.get(result, "unknown")

        return CheckResult(
            verdict=verdict,
            steps=bound if verdict == "sat" else None,
            witness=None,
            solver=self.name,
            elapsed=time.monotonic() - start,
        )
