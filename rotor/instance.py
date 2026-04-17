"""Phase 3: RotorInstance — one BTOR2 model instance with solve support."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rotor.btor2 import (
    BTOR2Printer,
    CRotorBackend,
    MachineModel,
    Node,
    NodeDAG,
    RISCVMachineBuilder,
)
from rotor.solvers import make_solver
from rotor.solvers.base import CheckResult, SolverBackend

if TYPE_CHECKING:  # pragma: no cover
    from rotor.binary import RISCVBinary
    from rotor.trace import MachineState


# ──────────────────────────────────────────────────────────────────────────
# Model configuration
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """All knobs that affect model generation and solving.

    The defaults mirror C Rotor's ``-m64`` target.
    """

    # Target architecture
    is_64bit: bool = True
    word_size: int = 64

    # Code window
    code_start: int = 0
    code_end: int = 0
    symbolic_code: bool = False
    symbolic_instructions: list[int] = field(default_factory=list)

    # Memory
    heap_allowance: int = 2048
    stack_allowance: int = 2048
    virtual_address_space: int = 32

    # Multicore
    cores: int = 1
    shared_input: bool = False
    binaries: list[str] = field(default_factory=list)

    # ISA extensions
    enable_rvc: bool = True
    enable_rv64m: bool = True
    riscu_only: bool = False

    # Initial state
    # If True, the register file is init'd to all zeros (default). Set False
    # for synthesis-style questions ("does any a0 cause X?"), which need
    # argument registers to be symbolic at step 0.
    init_registers_to_zero: bool = True

    # Solver
    solver: str = "bitwuzla"
    bound: int = 1000

    # Default checks
    check_division_by_zero: bool = True
    check_seg_faults: bool = True
    check_invalid_addresses: bool = True
    check_bad_exit_code: bool = True

    # Backend selection for model construction. ``"python"`` uses the
    # native Python :class:`RISCVMachineBuilder` (self-sufficient, default);
    # ``"crotor"`` delegates to the external C Rotor binary via subprocess
    # for cases where the native builder's ISA coverage is insufficient.
    model_backend: str = "python"


# ──────────────────────────────────────────────────────────────────────────
# RotorInstance
# ──────────────────────────────────────────────────────────────────────────


class RotorInstance:
    """One BTOR2 model of one RISC-V binary under one :class:`ModelConfig`.

    The workflow is:

        1. Construct with a :class:`RISCVBinary` + :class:`ModelConfig`.
        2. Call :meth:`build_machine` to generate the BTOR2 DAG.
        3. Optionally add user-defined bad/constraint nodes via
           :meth:`add_bad` / :meth:`add_constraint`.
        4. Call :meth:`check` to run the configured solver.
        5. On SAT, call :meth:`get_witness` to retrieve the trace.
    """

    def __init__(self, binary: "RISCVBinary", config: ModelConfig) -> None:
        self.binary = binary
        self.config = config
        self._model: MachineModel | None = None
        self._extra_bads: list[tuple[Node, str]] = []
        self._extra_constraints: list[tuple[Node, str]] = []
        self._last_result: CheckResult | None = None

    # ------------------------------------------------------------ properties

    @property
    def model(self) -> MachineModel:
        if self._model is None:
            raise RuntimeError(
                "RotorInstance.model accessed before build_machine()"
            )
        return self._model

    # --------------------------------------------------------- model building

    def build_machine(self) -> MachineModel:
        """Construct the BTOR2 model for this binary + config."""
        if self.config.model_backend == "crotor":
            backend = CRotorBackend()
            dag = backend.build(self.binary._path, self.config)
            # Translate the parsed DAG into a MachineModel shell; individual
            # solver paths only need the DAG + bad/constraint lists.
            model = MachineModel(
                dag=dag,
                property_nodes=[n for n in dag.nodes() if n.op == "bad"],
                constraint_nodes=[n for n in dag.nodes() if n.op == "constraint"],
            )
        elif self.config.model_backend == "python":
            builder = RISCVMachineBuilder(self.config)
            model = builder.build()
            # Bake the ELF's .text bytes into the code segment so the
            # fetch stage reads real instructions rather than symbolic ones.
            if self.binary.code is not None:
                code_seg = self.binary.code
                low = max(code_seg.start, self.config.code_start or code_seg.start)
                high = self.config.code_end or (code_seg.start + code_seg.size)
                high = min(high, code_seg.start + code_seg.size)
                if high > low:
                    offset = low - code_seg.start
                    length = high - low
                    for core in range(self.config.cores):
                        builder.initialize_code_segment(
                            core, low, code_seg.data[offset:offset + length]
                        )
            # Bake read-only data so loads of string literals / constant
            # tables see the real values rather than zero.
            if self.binary.rodata is not None and self.binary.rodata.size > 0:
                for core in range(self.config.cores):
                    builder.initialize_data_segment(
                        core, self.binary.rodata.start, self.binary.rodata.data
                    )
            if self.binary.data is not None and self.binary.data.size > 0:
                for core in range(self.config.cores):
                    builder.initialize_data_segment(
                        core, self.binary.data.start, self.binary.data.data
                    )
        else:
            raise ValueError(
                f"Unknown model_backend {self.config.model_backend!r}"
            )
        self._model = model
        return model

    # ------------------------------------------------------------- assertions

    def add_bad(self, condition: Node, name: str = "") -> None:
        """Register a user-specified bad property."""
        self._extra_bads.append((condition, name))

    def add_constraint(self, condition: Node, name: str = "") -> None:
        """Register a user-specified constraint (assumed invariant)."""
        self._extra_constraints.append((condition, name))

    # ----------------------------------------------------- convenience nodes

    def pc_equals(self, target_pc: int) -> Node:
        """Return a BTOR2 node that is true when ``PC == target_pc``."""
        if self._model is None or self._model.builder is None:
            raise RuntimeError(
                "pc_equals: requires build_machine() with model_backend='python'"
            )
        builder = self._model.builder  # type: ignore[attr-defined]
        pc_node = self._model.state_nodes["pc"]
        return builder.eq(
            pc_node,
            builder.consth(
                builder.SID_MACHINE_WORD,
                target_pc,
                f"pc == 0x{target_pc:x}",
            ),
        )

    def outputs_differ(self) -> Node:
        """Node that is true when two cores' ``a0`` return values diverge."""
        if self.config.cores < 2:
            raise ValueError("outputs_differ requires cores >= 2")
        if self._model is None or self._model.builder is None:
            raise RuntimeError(
                "outputs_differ: requires build_machine() with model_backend='python'"
            )
        a0_core0 = self._model.register_value(core=0, reg=10)
        a0_core1 = self._model.register_value(core=1, reg=10)
        return self._model.builder.neq(  # type: ignore[attr-defined]
            a0_core0, a0_core1, "a0 of core 0 != core 1"
        )

    # ------------------------------------------------------------ solving

    def check(self, bound: int | None = None) -> CheckResult:
        """Emit the model and run the configured solver."""
        if self._model is None:
            self.build_machine()
        solver = self._make_solver()
        btor2_text = self._emit_btor2()
        result = solver.check(btor2_text, bound or self.config.bound)
        self._last_result = result
        return result

    def get_witness(self) -> list["MachineState"]:
        """Return the concrete execution trace from the last SAT result."""
        from rotor.trace import MachineState

        if self._last_result is None or self._last_result.witness is None:
            return []
        trace: list[MachineState] = []
        registers: dict[int, int] = {}
        memory: dict[int, int] = {}
        for frame in self._last_result.witness:
            if frame.get("kind") != "state":
                continue
            step = int(frame.get("step", len(trace)))
            assignments = frame.get("assignments", {})
            pc = 0
            for key, value in assignments.items():
                if key.endswith("pc") or key == "pc":
                    pc = int(value)
                elif "register-file[" in key:
                    idx = int(key[key.index("[") + 1: key.index("]")])
                    registers[idx] = int(value)
            trace.append(
                MachineState(
                    step=step,
                    pc=pc,
                    registers=dict(registers),
                    memory=dict(memory),
                )
            )
        return trace

    # ------------------------------------------------------------ emission

    def _emit_btor2(self) -> str:
        assert self._model is not None
        printer = BTOR2Printer()
        buf = io.StringIO()
        # Extra bads/constraints live in the same DAG via add_bad() having
        # inserted them; ensure they exist as 'bad'/'constraint' nodes.
        dag = self._model.dag
        for cond, name in self._extra_bads:
            dag.get_or_create("bad", cond.sort, [cond], symbol=name)
        for cond, name in self._extra_constraints:
            dag.get_or_create("constraint", cond.sort, [cond], symbol=name)
        printer.print(dag, buf)
        return buf.getvalue()

    def emit_btor2(self) -> str:
        """Public accessor for the current BTOR2 text."""
        if self._model is None:
            self.build_machine()
        return self._emit_btor2()

    def _make_solver(self) -> SolverBackend:
        return make_solver(self.config.solver, config=self.config)
