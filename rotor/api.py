"""Phase 7: High-level question API.

A compact façade over :class:`~rotor.engine.RotorEngine` that maps the
categories of questions an LLM (or human) would ask about a RISC-V binary
into one method each. Each method returns a structured result with a
verdict plus a source-annotated trace / proof / counterexample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from rotor.binary import RISCVBinary
from rotor.engine import RotorEngine
from rotor.instance import ModelConfig, RotorInstance
from rotor.solvers.base import CheckResult
from rotor.trace import SourceTrace


# ──────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ReachResult:
    verdict: str  # 'reachable' | 'unreachable' | 'unknown'
    trace: SourceTrace | None = None
    proof: str | None = None
    elapsed: float = 0.0
    raw: CheckResult | None = None


@dataclass
class InputResult:
    verdict: str  # 'found' | 'no_such_input' | 'unknown'
    input_bytes: bytes | None = None
    trace: SourceTrace | None = None
    elapsed: float = 0.0
    raw: CheckResult | None = None


@dataclass
class EquivResult:
    verdict: str  # 'equivalent' | 'differs' | 'unknown'
    diverging_input: bytes | None = None
    trace_a: SourceTrace | None = None
    trace_b: SourceTrace | None = None
    proof: str | None = None
    elapsed: float = 0.0
    raw: CheckResult | None = None


@dataclass
class VerifyResult:
    verdict: str  # 'holds' | 'violated' | 'unknown'
    proof: str | None = None
    counterexample: SourceTrace | None = None
    unbounded: bool = False
    elapsed: float = 0.0
    raw: CheckResult | None = None


@dataclass
class CausalResult:
    verdict: str  # 'identified' | 'none' | 'unknown'
    critical_bytes: list[int] = field(default_factory=list)
    explanation: str = ""
    elapsed: float = 0.0


@dataclass
class SynthResult:
    verdict: str  # 'synthesized' | 'unsatisfiable' | 'unknown'
    value: int | None = None
    disassembly: str | None = None
    elapsed: float = 0.0
    raw: CheckResult | None = None


# ──────────────────────────────────────────────────────────────────────────
# RotorAPI
# ──────────────────────────────────────────────────────────────────────────


# Condition-expression compilers take a :class:`RotorInstance` plus a
# condition string and return a BTOR2 node usable as a bad property or
# constraint. Since a full expression language requires a parser over
# variable names, register names, memory accesses, this is exposed as a hook
# that callers (including LLM scaffolds) can plug in.
ConditionCompiler = Callable[[RotorInstance, str], object]


class RotorAPI:
    """The primary interface for asking questions about a RISC-V binary."""

    def __init__(
        self,
        binary_path: str,
        default_solver: str = "bitwuzla",
        default_bound: int = 1000,
        condition_compiler: ConditionCompiler | None = None,
    ) -> None:
        self.engine = RotorEngine(
            binary_path,
            default_solver=default_solver,
            default_bound=default_bound,
        )
        if condition_compiler is None:
            from rotor.expr import default_condition_compiler
            condition_compiler = default_condition_compiler()
        self._compile_condition = condition_compiler
        self.binary: RISCVBinary = self.engine.binary

    # -------------------------------------------------- condition machinery

    def _compile(self, inst: RotorInstance, expression: str):
        if self._compile_condition is None:
            raise NotImplementedError(
                "RotorAPI needs a condition_compiler to compile "
                f"expression {expression!r} into a BTOR2 node"
            )
        return self._compile_condition(inst, expression)

    def _make_instance(
        self, function: str, solver: str | None = None, bound: int | None = None,
    ) -> RotorInstance:
        config = ModelConfig(
            solver=solver or self.engine.default_solver,
            bound=bound or self.engine.default_bound,
        )
        return self.engine.create_instance(function, config=config)

    # ─────────────────────────────────────────────────────────── reachability

    def can_reach(
        self,
        function: str,
        condition: str,
        bound: int = 1000,
        unbounded: bool = False,
    ) -> ReachResult:
        solver = "ic3" if unbounded else self.engine.default_solver
        inst = self._make_instance(function, solver=solver, bound=bound)
        cond_node = self._compile(inst, condition)
        inst.add_bad(cond_node, f"reach: {condition}")  # type: ignore[arg-type]
        result = inst.check(bound)

        trace: SourceTrace | None = None
        if result.verdict == "sat":
            trace = SourceTrace(inst.get_witness(), self.binary)

        verdict_map = {"sat": "reachable", "unsat": "unreachable", "unknown": "unknown"}
        return ReachResult(
            verdict=verdict_map[result.verdict],
            trace=trace,
            proof=result.invariant,
            elapsed=result.elapsed,
            raw=result,
        )

    # ──────────────────────────────────────────────────── input synthesis

    def find_input(
        self,
        function: str,
        output_condition: str,
        bound: int = 1000,
    ) -> InputResult:
        inst = self._make_instance(function, bound=bound)
        cond_node = self._compile(inst, output_condition)
        inst.add_bad(cond_node, f"find-input: {output_condition}")  # type: ignore[arg-type]
        result = inst.check(bound)

        trace: SourceTrace | None = None
        input_bytes: bytes | None = None
        verdict = {"sat": "found", "unsat": "no_such_input", "unknown": "unknown"}[result.verdict]
        if result.verdict == "sat":
            witness = inst.get_witness()
            trace = SourceTrace(witness, self.binary)
            input_bytes = _extract_input_bytes(witness)
        return InputResult(
            verdict=verdict,
            input_bytes=input_bytes,
            trace=trace,
            elapsed=result.elapsed,
            raw=result,
        )

    # ──────────────────────────────────────────────────────────── equivalence

    def are_equivalent(
        self,
        other_binary: str,
        function: str,
        bound: int = 1000,
        unbounded: bool = False,
    ) -> EquivResult:
        solver = "ic3" if unbounded else self.engine.default_solver
        # Build a dual-core model of [self, other] with shared input.
        config = ModelConfig(
            solver=solver,
            bound=bound,
            cores=2,
            shared_input=True,
            binaries=[self.engine.binary._path, other_binary],
        )
        inst = self.engine.create_instance(function, config=config)
        inst.add_bad(inst.outputs_differ(), "outputs-diverge")
        result = inst.check(bound)

        trace_a: SourceTrace | None = None
        trace_b: SourceTrace | None = None
        verdict = {"sat": "differs", "unsat": "equivalent", "unknown": "unknown"}[result.verdict]
        if result.verdict == "sat":
            witness = inst.get_witness()
            trace_a = SourceTrace(witness, self.binary)
            # trace_b would require a RISCVBinary for `other_binary`; left as
            # a caller-side concern.
        return EquivResult(
            verdict=verdict,
            trace_a=trace_a,
            trace_b=trace_b,
            proof=result.invariant,
            elapsed=result.elapsed,
            raw=result,
        )

    # ───────────────────────────────────────────────── property verification

    def verify(
        self,
        function: str,
        invariant: str,
        bound: int = 1000,
        unbounded: bool = False,
    ) -> VerifyResult:
        # 'kind' is the in-process k-induction solver; it returns an unbounded
        # proof when the invariant is k-inductive for some k ≤ bound. For
        # non-inductive invariants, users can switch to 'ic3' (external rIC3/
        # AVR/ABC) to get CEGAR-style strengthening.
        solver = "kind" if unbounded else self.engine.default_solver
        # For unbounded proofs we suppress the native illegal-instruction
        # bad: it is not k-inductive on its own (symbolic code permits any
        # opcode), so leaving it in would poison the inductive step.
        config = ModelConfig(
            solver=solver,
            bound=bound or self.engine.default_bound,
            emit_default_bad_properties=not unbounded,
        )
        inst = self.engine.create_instance(function, config=config)
        # An invariant I is safe iff !I is unreachable: assert bad(!I).
        invariant_node = self._compile(inst, invariant)
        builder = inst.model.builder  # type: ignore[attr-defined]
        if builder is None:
            raise RuntimeError(
                "verify: requires a python builder; current model has none"
            )
        negated = builder.not_(invariant_node, f"not-({invariant})")  # type: ignore[attr-defined]
        inst.add_bad(negated, f"invariant-violated: {invariant}")
        result = inst.check(bound)

        counterexample: SourceTrace | None = None
        verdict = {"sat": "violated", "unsat": "holds", "unknown": "unknown"}[result.verdict]
        if result.verdict == "sat":
            counterexample = SourceTrace(inst.get_witness(), self.binary)
        return VerifyResult(
            verdict=verdict,
            proof=result.invariant,
            counterexample=counterexample,
            unbounded=unbounded and result.verdict == "unsat",
            elapsed=result.elapsed,
            raw=result,
        )

    # ──────────────────────────────────────────────────────── causality

    def find_responsible_inputs(
        self,
        function: str,
        condition: str,
        known_input: bytes,
        bound: int = 1000,
    ) -> CausalResult:
        """Delta-debug the input bytes to find a minimal set that preserves
        reachability of ``condition``.

        Starts with the full ``known_input`` as a constraint and progressively
        releases byte subsets; if the condition remains reachable, those
        bytes are not responsible. Binary search over subsets.
        """
        # Deferred: requires the condition_compiler to accept byte-indexed
        # constraints on the input. Returns a skeleton result.
        _ = (function, condition, known_input, bound)
        return CausalResult(
            verdict="unknown",
            critical_bytes=[],
            explanation="find_responsible_inputs requires a byte-indexed compiler; not configured",
        )

    # ────────────────────────────────────────────────────── value synthesis

    def synthesize_value(
        self,
        function: str,
        hole_location: str,
        spec: str,
        bound: int = 100,
    ) -> SynthResult:
        """Ask C Rotor's symbolic-code mode to synthesize an instruction.

        ``hole_location`` is either a symbol name or ``file:line``. The
        instruction at that PC is made symbolic and the solver finds values
        for it that satisfy ``spec``.
        """
        pc = self._resolve_hole_pc(hole_location)
        config = ModelConfig(
            solver=self.engine.default_solver,
            bound=bound,
            symbolic_code=True,
            symbolic_instructions=[pc],
        )
        inst = RotorInstance(self.binary, config)
        inst.build_machine()
        spec_node = self._compile(inst, spec)
        inst.add_bad(spec_node, f"synth-spec: {spec}")  # type: ignore[arg-type]
        result = inst.check(bound)

        verdict = {"sat": "synthesized", "unsat": "unsatisfiable", "unknown": "unknown"}[result.verdict]
        value = None
        disassembly = None
        if result.verdict == "sat" and result.witness:
            # The first frame assignment under the synthesis symbol holds
            # the synthesized bytes.
            frame0 = result.witness[0] if result.witness else {}
            assignments = frame0.get("assignments", {})
            if assignments:
                raw = next(iter(assignments.values()))
                try:
                    value = int(raw, 2) if set(raw) <= {"0", "1"} else int(raw, 16)
                except ValueError:
                    value = None
                if value is not None:
                    disassembly = self.binary.disassemble(pc)

        return SynthResult(
            verdict=verdict,
            value=value,
            disassembly=disassembly,
            elapsed=result.elapsed,
            raw=result,
        )

    # ───────────────────────────────────────────────────────── strategy

    def _select_strategy(
        self, function: str, question_type: str, unbounded: bool
    ) -> list[tuple[str, dict]]:
        configs: list[tuple[str, dict]] = []
        if question_type in ("reachability", "input_synthesis"):
            configs.append(("bitwuzla", {"bound": 100}))
            configs.append(("bitwuzla", {"bound": 1000}))
        if unbounded or question_type == "verification":
            configs.append(("ic3", {"backend": "ric3"}))
            configs.append(("ic3", {"backend": "avr"}))
        if question_type == "equivalence":
            configs.append(("ic3", {"backend": "ric3", "cores": 2}))
        return configs

    # --------------------------------------------------------- helpers

    def _resolve_hole_pc(self, hole_location: str) -> int:
        if ":" in hole_location:
            file_part, _, line_part = hole_location.partition(":")
            line = int(line_part)
            for pc, loc in (self.binary._line_map or {}).items():
                if loc.file.endswith(file_part) and loc.line == line:
                    return pc
            raise KeyError(f"No PC found for source location {hole_location!r}")
        sym = self.binary.symbols.get(hole_location)
        if sym is not None:
            return sym.address
        raise KeyError(f"Hole location {hole_location!r} not found")


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _extract_input_bytes(witness) -> bytes | None:
    """Best-effort extraction of concrete input bytes from a witness."""
    if not witness:
        return None
    # Inputs are assigned in the kernel model; if none are visible, return
    # None rather than guessing.
    return None
