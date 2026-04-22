"""High-level entry point for rotor.

M2 exposes can_reach with a source-lifted Trace on reachable verdicts;
Phase 6.5 adds the `unbounded=True` shortcut (Z3Spacer) and a
`cegar_reach` method for abstraction-refinement. Additional verbs
(find_input, verify, are_equivalent) land later under the same
architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from rotor.binary import RISCVBinary
from rotor.dwarf import DwarfLineMap
from rotor.engine import EngineConfig, RotorEngine
from rotor.solvers.base import SolverBackend
from rotor.solvers.portfolio import Portfolio
from rotor.trace import Trace, build_trace

if TYPE_CHECKING:
    from rotor.cegar import CegarConfig


@dataclass(frozen=True)
class ReachResult:
    verdict: str                       # "reachable" | "unreachable" | "proved" | "unknown"
    bound: int
    step: Optional[int]
    initial_regs: dict[str, int]
    elapsed: float
    backend: str
    trace: Optional[Trace] = None      # populated when verdict == "reachable"
    invariant: Optional[str] = None    # certificate when verdict == "proved"


# `verify` and `find_input` reuse `ReachResult` — all three verbs
# share the same verdict vocabulary and counterexample shape. The
# asymmetry is interpretation:
#   can_reach:   `reachable` = PC hit the target.
#   verify:      `reachable` = predicate FAILED (CEX found).
#   find_input:  `reachable` = predicate HOLDS (synthesis witness).
# Use the verb you called to interpret the result.
VerifyResult = ReachResult
FindInputResult = ReachResult


class RotorAPI:
    def __init__(
        self,
        binary_path: Union[str, Path],
        *,
        default_bound: int = 20,
        backend: Optional[SolverBackend] = None,
        portfolio: Optional[Portfolio] = None,
    ) -> None:
        self._binary_path = Path(binary_path)
        self._binary = RISCVBinary(self._binary_path)
        self._dwarf = DwarfLineMap(self._binary_path)
        self._engine = RotorEngine(
            self._binary,
            config=EngineConfig(
                backend=backend,
                portfolio=portfolio,
                default_bound=default_bound,
            ),
        )

    @property
    def binary(self) -> RISCVBinary:
        return self._binary

    @property
    def engine(self) -> RotorEngine:
        return self._engine

    def close(self) -> None:
        self._binary.close()

    def __enter__(self) -> "RotorAPI":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def can_reach(
        self,
        function: str,
        target_pc: int,
        bound: Optional[int] = None,
        timeout: Optional[float] = None,
        unbounded: bool = False,
    ) -> ReachResult:
        """Reachability check. Default is bounded BMC.

        Set `unbounded=True` to route through Z3 Spacer (PDR/IC3)
        instead; the `bound` parameter is then ignored and the answer
        may be `proved` with an inductive invariant. Spacer's Python
        API does not expose counterexample traces, so `reachable`
        verdicts from unbounded mode have no step/trace.
        """
        if unbounded:
            result = self._engine.check_reach_unbounded(
                function, target_pc, timeout=timeout,
            )
        else:
            result = self._engine.check_reach(
                function, target_pc, bound=bound, timeout=timeout,
            )
        return self._finalize(function, target_pc, result)

    def verify(
        self,
        function: str,
        register: int,
        comparison: str,
        rhs: int,
        bound: Optional[int] = None,
        timeout: Optional[float] = None,
        unbounded: bool = False,
    ) -> VerifyResult:
        """Verify that `regs[register] <comparison> rhs` holds at every
        `ret` of `function`.

        Interpretation mirrors `can_reach` with the polarity flipped —
        `reachable` here means the predicate can FAIL on some input,
        with `initial_regs` / `trace` identifying the counterexample;
        `unreachable` means safe up to the bound; `proved` means safe
        on every execution path (only from `unbounded=True`).

        `register` is the ABI index (10 = a0 = RISC-V return value).
        `comparison` is one of: eq, neq, slt, slte, sgt, sgte, ult,
        ulte, ugt, ugte. Signedness is encoded in the comparison name.
        `rhs` is the 64-bit integer right-hand side; negative values
        are masked into two's-complement automatically.
        """
        result = self._engine.check_verify(
            function, register, comparison, rhs,
            bound=bound, timeout=timeout, unbounded=unbounded,
        )
        return self._finalize_verify(function, register, comparison, rhs, result)

    def _finalize_verify(
        self, function: str, register: int, comparison: str, rhs: int, result,
    ):
        # Verify CEXs live at a return site with a specific register
        # failing the predicate. We don't produce a trace today —
        # which specific ret the CEX hit isn't surfaced by the solver
        # backends — but initial_regs still identify the failing input.
        return ReachResult(
            verdict=result.verdict,
            bound=result.bound,
            step=result.step,
            initial_regs=result.initial_regs,
            elapsed=result.elapsed,
            backend=result.backend,
            trace=None,
            invariant=result.invariant,
        )

    def find_input(
        self,
        function: str,
        register: int,
        comparison: str,
        rhs: int,
        bound: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> "FindInputResult":
        """Synthesize an input that makes `regs[register] <comparison>
        rhs` hold at a return site of `function`.

        - `reachable` → witness found; `initial_regs` is the input.
        - `unreachable` → no input achieves the predicate within the
          given bound. (Larger bound may still find one.)
        - `unknown` → solver gave up.

        Bounded BMC only. For an unbounded "is the predicate
        achievable on any execution?" question, use `verify(...,
        comparison=negated_op, unbounded=True)` — `reachable` there
        answers the same question.
        """
        result = self._engine.check_find_input(
            function, register, comparison, rhs,
            bound=bound, timeout=timeout,
        )
        return ReachResult(
            verdict=result.verdict,
            bound=result.bound,
            step=result.step,
            initial_regs=result.initial_regs,
            elapsed=result.elapsed,
            backend=result.backend,
            trace=None,
            invariant=None,
        )

    def cegar_reach(
        self,
        function: str,
        target_pc: int,
        config: Optional[CegarConfig] = None,
    ) -> ReachResult:
        """Counterexample-guided abstraction refinement.

        Drives Z3 Spacer against progressively-refined abstractions;
        confirms any abstract counterexample against rotor's concrete
        witness simulator. Returns `proved` + invariant, `reachable`
        with a concrete step index, or `unknown` on unrefinable CEX
        or iteration-budget exhaustion.
        """
        result = self._engine.check_reach_cegar(function, target_pc, config=config)
        return self._finalize(function, target_pc, result)

    def _finalize(self, function: str, target_pc: int, result) -> ReachResult:
        trace = None
        if result.verdict == "reachable" and result.step is not None:
            trace = build_trace(
                binary=self._binary,
                function=function,
                target_pc=target_pc,
                verdict=result.verdict,
                bound=result.bound,
                reached_at=result.step,
                elapsed=result.elapsed,
                backend=result.backend,
                initial_regs=result.initial_regs,
                dwarf=self._dwarf,
            )
        return ReachResult(
            verdict=result.verdict,
            bound=result.bound,
            step=result.step,
            initial_regs=result.initial_regs,
            elapsed=result.elapsed,
            backend=result.backend,
            trace=trace,
            invariant=result.invariant,
        )
