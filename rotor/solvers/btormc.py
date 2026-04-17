"""BtorMC BMC backend (subprocess)."""

from __future__ import annotations

import subprocess
import tempfile
import time

from rotor.solvers.base import CheckResult, SolverBackend
from rotor.witness import parse_btor2_witness


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
        witness = parse_btor2_witness(stdout)
        if witness.verdict == "sat":
            frames: list[dict] = []
            for frame in witness.frames:
                assignments = {}
                for a in frame.assignments:
                    key = a.symbol or str(a.nid)
                    if a.index is not None:
                        assignments[f"{key}[{a.index}]"] = a.value
                    else:
                        assignments[key] = a.value
                frames.append(
                    {
                        "step": frame.step,
                        "kind": frame.kind,
                        "assignments": assignments,
                    }
                )
            steps = (
                max((f.step for f in witness.state_frames()), default=0)
                if witness.state_frames()
                else 0
            )
            return "sat", steps, frames or None
        if witness.verdict == "unsat":
            return "unsat", None, None
        return "unknown", None, None
