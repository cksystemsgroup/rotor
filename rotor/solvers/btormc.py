"""BtorMC BMC backend (subprocess)."""

from __future__ import annotations

import re
import subprocess
import tempfile
import time

from rotor.solvers.base import CheckResult, SolverBackend


_BTORMC_SAT_RE = re.compile(r"\bsat\b", re.IGNORECASE)
_BTORMC_UNSAT_RE = re.compile(r"\bunsat\b", re.IGNORECASE)
_BTORMC_BAD_RE = re.compile(r"^b(\d+)\b", re.MULTILINE)


class BtorMCSolver(SolverBackend):
    """BtorMC via subprocess.

    BtorMC is a BMC-first solver; it can be driven to exhaustive BMC for a
    fixed bound. When SAT, its stdout contains a witness in a format that
    other rotor components parse into :class:`MachineState` frames.
    """

    name = "btormc"

    def __init__(
        self,
        config: object | None = None,
        binary: str = "btormc",
        timeout: float = 300.0,
    ) -> None:
        self.config = config
        self.binary = binary
        self.timeout = timeout

    def supports_unbounded(self) -> bool:
        return False

    def check(self, btor2: str, bound: int) -> CheckResult:
        start = time.monotonic()
        with tempfile.NamedTemporaryFile(
            "w", suffix=".btor2", delete=False
        ) as handle:
            handle.write(btor2)
            path = handle.name
        try:
            proc = subprocess.run(
                [self.binary, f"--kmax={bound}", "--trace-gen-full", path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError:
            return CheckResult(
                verdict="unknown",
                solver=self.name,
                elapsed=time.monotonic() - start,
                stderr=f"{self.binary} not found on PATH",
            )
        except subprocess.TimeoutExpired:
            return CheckResult(
                verdict="unknown",
                solver=self.name,
                elapsed=time.monotonic() - start,
                stderr="btormc timed out",
            )

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        verdict, steps, witness = self._parse(stdout)
        return CheckResult(
            verdict=verdict,
            steps=steps,
            witness=witness,
            solver=self.name,
            elapsed=time.monotonic() - start,
            stdout=stdout,
            stderr=stderr,
        )

    @staticmethod
    def _parse(stdout: str) -> tuple[str, int | None, list[dict] | None]:
        if _BTORMC_SAT_RE.search(stdout):
            # Parse the witness into frames. BtorMC prints '@k' lines to
            # delimit frame k; state/input assignments follow as
            # '<nid> <value>' or '<nid> <value> <symbol>'.
            frames: list[dict] = []
            current: dict | None = None
            for line in stdout.splitlines():
                stripped = line.strip()
                if stripped.startswith("@"):
                    if current is not None:
                        frames.append(current)
                    try:
                        k = int(stripped[1:])
                    except ValueError:
                        k = len(frames)
                    current = {"step": k, "assignments": {}}
                elif current is not None and stripped and stripped[0].isdigit():
                    tokens = stripped.split()
                    if len(tokens) >= 2:
                        current["assignments"][tokens[0]] = tokens[1]
            if current is not None:
                frames.append(current)
            return "sat", len(frames) - 1 if frames else 0, frames or None
        if _BTORMC_UNSAT_RE.search(stdout):
            return "unsat", None, None
        return "unknown", None, None
