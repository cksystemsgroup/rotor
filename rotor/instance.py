"""A RotorInstance is one BTOR2 proof obligation plus a default backend.

M1 wraps the build-reach + solve pipeline. Higher layers will add other
question kinds and alternative emitters; the instance is the thing that
carries a Model + emitter + backend through a single check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rotor.binary import RISCVBinary
from rotor.btor2.builder import ReachSpec, build_reach
from rotor.btor2.nodes import Model
from rotor.solvers.base import SolverBackend, SolverResult
from rotor.solvers.z3bv import Z3BMC


@dataclass
class RotorInstance:
    binary: RISCVBinary
    model: Model
    backend: SolverBackend

    @classmethod
    def for_reach(
        cls,
        binary: RISCVBinary,
        function: str,
        target_pc: int,
        backend: Optional[SolverBackend] = None,
    ) -> "RotorInstance":
        spec = ReachSpec(function=function, target_pc=target_pc)
        model = build_reach(binary, spec)
        return cls(binary=binary, model=model, backend=backend or Z3BMC())

    def check(self, bound: int, timeout: Optional[float] = None) -> SolverResult:
        return self.backend.check_reach(self.model, bound=bound, timeout=timeout)
