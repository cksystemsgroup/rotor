"""IC3 / PDR backends via subprocess (rIC3, AVR, ABC).

IC3-class solvers can return **unbounded** safety proofs (as inductive
invariants) or concrete counterexamples. Rotor defers to existing mature
implementations rather than re-implementing IC3.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time

from rotor.solvers.base import CheckResult, SolverBackend


_BACKEND_COMMANDS = {
    # Rust-based IC3 reimplementation; takes a BTOR2 file.
    "ric3": ["ric3"],
    # AVR is invoked through its driver script.
    "avr": ["avr.py"],
    # ABC's pdr engine reads AIGER; for BTOR2 we go via btor2aiger first.
    "abc": ["abc"],
}


class IC3Solver(SolverBackend):
    """Unbounded IC3 via rIC3 / AVR / ABC.

    ``backend`` selects the underlying solver. The ``check`` method writes
    the BTOR2 text to a temp file and runs the backend; the output is parsed
    into a :class:`CheckResult` with ``verdict`` set to ``'unsat'`` and an
    ``invariant`` string on proof, ``'sat'`` with a witness on CEX, or
    ``'unknown'`` on resource exhaustion.
    """

    name = "ic3"

    BACKENDS = tuple(_BACKEND_COMMANDS.keys())

    def __init__(
        self,
        config: object | None = None,
        backend: str = "ric3",
        timeout: float = 600.0,
        extra_args: list[str] | None = None,
    ) -> None:
        self.config = config
        if backend not in _BACKEND_COMMANDS:
            raise ValueError(
                f"Unknown IC3 backend {backend!r}; available: {self.BACKENDS}"
            )
        self.backend = backend
        self.timeout = timeout
        self.extra_args = list(extra_args or [])

    def supports_unbounded(self) -> bool:
        return True

    def check(self, btor2: str, bound: int) -> CheckResult:
        start = time.monotonic()
        with tempfile.NamedTemporaryFile(
            "w", suffix=".btor2", delete=False
        ) as handle:
            handle.write(btor2)
            path = handle.name

        cmd = _BACKEND_COMMANDS[self.backend] + self.extra_args + [path]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError:
            return CheckResult(
                verdict="unknown",
                solver=f"{self.name}/{self.backend}",
                elapsed=time.monotonic() - start,
                stderr=f"{cmd[0]} not found on PATH",
            )
        except subprocess.TimeoutExpired:
            return CheckResult(
                verdict="unknown",
                solver=f"{self.name}/{self.backend}",
                elapsed=time.monotonic() - start,
                stderr=f"{self.backend} timed out after {self.timeout}s",
            )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        verdict = self._parse_verdict(stdout)
        invariant = (
            self._parse_invariant(stdout) if verdict == "unsat" else None
        )
        return CheckResult(
            verdict=verdict,
            steps=None,
            witness=None,
            invariant=invariant,
            solver=f"{self.name}/{self.backend}",
            elapsed=time.monotonic() - start,
            stdout=stdout,
            stderr=stderr,
        )

    @staticmethod
    def _parse_verdict(stdout: str) -> str:
        lower = stdout.lower()
        if re.search(r"\bproved\b|\bsafe\b|result:\s*unsat", lower):
            return "unsat"
        if re.search(r"\bcex\b|counterexample|result:\s*sat", lower):
            return "sat"
        if re.search(r"\bunsat\b", lower):
            return "unsat"
        if re.search(r"\bsat\b", lower):
            return "sat"
        return "unknown"

    @staticmethod
    def _parse_invariant(stdout: str) -> str | None:
        """Extract an inductive invariant block if the backend emits one."""
        lines = stdout.splitlines()
        invariant_lines: list[str] = []
        capturing = False
        for line in lines:
            if re.search(r"inductive invariant", line, re.IGNORECASE):
                capturing = True
                continue
            if capturing:
                if not line.strip():
                    if invariant_lines:
                        break
                    continue
                invariant_lines.append(line)
        return "\n".join(invariant_lines) if invariant_lines else None
