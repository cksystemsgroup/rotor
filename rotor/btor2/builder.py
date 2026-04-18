"""BTOR2 builder: ergonomic constructors over the node DAG.

The :class:`BTOR2Builder` exposes one method per BTOR2 operator. The
:class:`RISCVMachineBuilder` subclass is a skeleton for the full Python
reimplementation of C Rotor's machine semantics; the initial Phase 2
deliverable uses :class:`CRotorBackend` to generate BTOR2 via the C tool and
parses it back with :func:`rotor.btor2.parser.parse_btor2`.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from rotor.btor2.nodes import MachineModel, Node, NodeDAG, Sort

if TYPE_CHECKING:  # pragma: no cover
    from rotor.instance import ModelConfig


# ──────────────────────────────────────────────────────────────────────────
# BTOR2Builder
# ──────────────────────────────────────────────────────────────────────────


class BTOR2Builder:
    """Ergonomic constructor over a :class:`NodeDAG`.

    All methods return :class:`~rotor.btor2.nodes.Node` and rely on the DAG's
    structural sharing so that repeated calls with the same arguments produce
    the same object.
    """

    def __init__(self, dag: NodeDAG | None = None) -> None:
        self.dag = dag or NodeDAG()

    # ------------------------------------------------------------------ sorts

    def bitvec(self, width: int, comment: str = "") -> Sort:
        return self.dag.intern_sort("bitvec", width=width, comment=comment)

    def array(self, index_sort: Sort, elem_sort: Sort, comment: str = "") -> Sort:
        return self.dag.intern_sort(
            "array", index_sort=index_sort, elem_sort=elem_sort, comment=comment,
        )

    # -------------------------------------------------------------- constants

    def zero(self, sort: Sort, comment: str = "") -> Node:
        return self.dag.get_or_create("zero", sort, comment=comment)

    def one(self, sort: Sort, comment: str = "") -> Node:
        return self.dag.get_or_create("one", sort, comment=comment)

    def ones(self, sort: Sort, comment: str = "") -> Node:
        return self.dag.get_or_create("ones", sort, comment=comment)

    def constd(self, sort: Sort, value: int, comment: str = "") -> Node:
        return self.dag.get_or_create("constd", sort, params=[value], comment=comment)

    def consth(self, sort: Sort, value: int, comment: str = "") -> Node:
        return self.dag.get_or_create("consth", sort, params=[value], comment=comment)

    def const(self, sort: Sort, binary: str, comment: str = "") -> Node:
        # Binary constants pass the integer value of the bitstring as a param
        # along with its width; the printer renders it back in binary.
        value = int(binary, 2) if binary else 0
        return self.dag.get_or_create("const", sort, params=[value, len(binary)], comment=comment)

    # ---------------------------------------------------------- unary / misc

    def not_(self, a: Node, comment: str = "") -> Node:
        return self.dag.get_or_create("not", a.sort, [a], comment=comment)

    def inc(self, a: Node, comment: str = "") -> Node:
        return self.dag.get_or_create("inc", a.sort, [a], comment=comment)

    def dec(self, a: Node, comment: str = "") -> Node:
        return self.dag.get_or_create("dec", a.sort, [a], comment=comment)

    def neg(self, a: Node, comment: str = "") -> Node:
        return self.dag.get_or_create("neg", a.sort, [a], comment=comment)

    def redor(self, a: Node, comment: str = "") -> Node:
        return self.dag.get_or_create("redor", self.bitvec(1), [a], comment=comment)

    def redand(self, a: Node, comment: str = "") -> Node:
        return self.dag.get_or_create("redand", self.bitvec(1), [a], comment=comment)

    def redxor(self, a: Node, comment: str = "") -> Node:
        return self.dag.get_or_create("redxor", self.bitvec(1), [a], comment=comment)

    # ---------------------------------------------------- extension & slicing

    def sext(self, sort: Sort, a: Node, w: int, comment: str = "") -> Node:
        return self.dag.get_or_create("sext", sort, [a], params=[w], comment=comment)

    def uext(self, sort: Sort, a: Node, w: int, comment: str = "") -> Node:
        return self.dag.get_or_create("uext", sort, [a], params=[w], comment=comment)

    def slice(self, sort: Sort, a: Node, u: int, l: int, comment: str = "") -> Node:
        return self.dag.get_or_create("slice", sort, [a], params=[u, l], comment=comment)

    # ---------------------------------------------------- binary arith / logic

    def _binop(self, op: str, a: Node, b: Node, comment: str, sort: Sort | None = None) -> Node:
        sort = sort or a.sort
        return self.dag.get_or_create(op, sort, [a, b], comment=comment)

    def add(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._binop("add", a, b, comment)

    def sub(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._binop("sub", a, b, comment)

    def mul(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._binop("mul", a, b, comment)

    def udiv(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._binop("udiv", a, b, comment)

    def urem(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._binop("urem", a, b, comment)

    def sdiv(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._binop("sdiv", a, b, comment)

    def srem(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._binop("srem", a, b, comment)

    def and_(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._binop("and", a, b, comment)

    def or_(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._binop("or", a, b, comment)

    def xor(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._binop("xor", a, b, comment)

    def sll(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._binop("sll", a, b, comment)

    def srl(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._binop("srl", a, b, comment)

    def sra(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._binop("sra", a, b, comment)

    def concat(self, a: Node, b: Node, comment: str = "") -> Node:
        width = (a.sort.width or 0) + (b.sort.width or 0)
        return self._binop("concat", a, b, comment, sort=self.bitvec(width))

    # ---------------------------------------------------------------- compare

    def _cmp(self, op: str, a: Node, b: Node, comment: str) -> Node:
        return self.dag.get_or_create(op, self.bitvec(1), [a, b], comment=comment)

    def eq(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._cmp("eq", a, b, comment)

    def neq(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._cmp("neq", a, b, comment)

    def ult(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._cmp("ult", a, b, comment)

    def ulte(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._cmp("ulte", a, b, comment)

    def ugt(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._cmp("ugt", a, b, comment)

    def ugte(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._cmp("ugte", a, b, comment)

    def slt(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._cmp("slt", a, b, comment)

    def slte(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._cmp("slte", a, b, comment)

    def sgt(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._cmp("sgt", a, b, comment)

    def sgte(self, a: Node, b: Node, comment: str = "") -> Node:
        return self._cmp("sgte", a, b, comment)

    # --------------------------------------------------------------- ternary

    def ite(self, c: Node, a: Node, b: Node, comment: str = "") -> Node:
        return self.dag.get_or_create("ite", a.sort, [c, a, b], comment=comment)

    # ---------------------------------------------------------------- arrays

    def read(self, array: Node, index: Node, comment: str = "") -> Node:
        elem = array.sort.elem_sort
        if elem is None:
            raise TypeError("read: array operand must have array sort")
        return self.dag.get_or_create("read", elem, [array, index], comment=comment)

    def write(self, array: Node, index: Node, value: Node, comment: str = "") -> Node:
        return self.dag.get_or_create("write", array.sort, [array, index, value], comment=comment)

    # -------------------------------------------------------- sequential logic

    def state(self, sort: Sort, symbol: str, comment: str = "") -> Node:
        return self.dag.get_or_create("state", sort, symbol=symbol, comment=comment)

    def input(self, sort: Sort, symbol: str, comment: str = "") -> Node:
        return self.dag.get_or_create("input", sort, symbol=symbol, comment=comment)

    def init(self, sort: Sort, state: Node, value: Node, comment: str = "") -> Node:
        return self.dag.get_or_create("init", sort, [state, value], comment=comment)

    def next(self, sort: Sort, state: Node, value: Node, comment: str = "") -> Node:
        return self.dag.get_or_create("next", sort, [state, value], comment=comment)

    # -------------------------------------------------------------- properties

    def bad(self, cond: Node, symbol: str = "", comment: str = "") -> Node:
        return self.dag.get_or_create("bad", cond.sort, [cond], symbol=symbol, comment=comment)

    def constraint(self, cond: Node, symbol: str = "", comment: str = "") -> Node:
        return self.dag.get_or_create("constraint", cond.sort, [cond], symbol=symbol, comment=comment)

    def fair(self, cond: Node, symbol: str = "", comment: str = "") -> Node:
        return self.dag.get_or_create("fair", cond.sort, [cond], symbol=symbol, comment=comment)

    def justice(self, conds: Iterable[Node], symbol: str = "", comment: str = "") -> Node:
        conds_list = list(conds)
        sort = conds_list[0].sort if conds_list else self.bitvec(1)
        return self.dag.get_or_create(
            "justice", sort, conds_list, params=[len(conds_list)],
            symbol=symbol, comment=comment,
        )


# ──────────────────────────────────────────────────────────────────────────
# RISCVMachineBuilder (skeleton — full semantics are deferred)
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class _MachineBuildState:
    """Working state accumulated during :meth:`RISCVMachineBuilder.build`."""

    state_nodes: dict[str, Node] = field(default_factory=dict)
    input_nodes: dict[str, Node] = field(default_factory=dict)
    init_nodes: list[Node] = field(default_factory=list)
    next_nodes: list[Node] = field(default_factory=list)
    property_nodes: list[Node] = field(default_factory=list)
    constraint_nodes: list[Node] = field(default_factory=list)


class RISCVMachineBuilder(BTOR2Builder):
    """Python skeleton mirroring ``rotor.c``'s machine builder.

    This class defines the overall structure (sort/constant pool, register
    file, memory segments, fetch/decode/execute, kernel model, properties)
    but intentionally leaves the per-instruction semantics as ``TODO``. The
    initial Phase 2 deliverable uses :class:`CRotorBackend` to generate BTOR2
    externally and parses it into a DAG.
    """

    def __init__(self, config: "ModelConfig") -> None:
        super().__init__()
        self.config = config
        self._state = _MachineBuildState()

        # Commonly-referenced sorts and constants; populated in _build_sorts.
        self.SID_BOOLEAN: Sort | None = None
        self.SID_BYTE: Sort | None = None
        self.SID_HALF_WORD: Sort | None = None
        self.SID_SINGLE_WORD: Sort | None = None
        self.SID_DOUBLE_WORD: Sort | None = None
        self.SID_MACHINE_WORD: Sort | None = None
        self.SID_VIRTUAL_ADDRESS: Sort | None = None
        self.SID_REGISTER_ADDRESS: Sort | None = None
        self.SID_REGISTER_STATE: Sort | None = None

        self.NID_FALSE: Node | None = None
        self.NID_TRUE: Node | None = None

        # Per-core register-file state nodes: core → state Node.
        self._register_file: dict[int, Node] = {}
        # Per-core memory segment state nodes.
        self._memory_segments: dict[tuple[int, str], Node] = {}
        # Per-core program counter state nodes.
        self._pc_nodes: dict[int, Node] = {}
        # Per-core halted latches (set to 1 by ECALL).
        self._halted_nodes: dict[int, Node] = {}
        # Per-core exit-code latches (set by the exit syscall).
        self._exit_code_nodes: dict[int, Node] = {}
        # Per-core fresh-input byte nodes (read syscall input source).
        self._input_byte_nodes: dict[int, Node] = {}

    # -------------------------------------------------------------- lifecycle

    def build(self) -> MachineModel:
        self._build_sorts_and_constants()
        self._build_register_file()
        self._build_memory_segments()
        self._build_fetch_decode_execute()
        self._build_kernel_model()
        self._build_properties()
        return MachineModel(
            dag=self.dag,
            state_nodes=self._state.state_nodes,
            input_nodes=self._state.input_nodes,
            init_nodes=self._state.init_nodes,
            next_nodes=self._state.next_nodes,
            property_nodes=self._state.property_nodes,
            constraint_nodes=self._state.constraint_nodes,
            builder=self,
        )

    # ----------------------------------------------------------- subsystems

    def _build_sorts_and_constants(self) -> None:
        w = 64 if self.config.is_64bit else 32
        self.SID_BOOLEAN = self.bitvec(1, "Boolean")
        self.SID_BYTE = self.bitvec(8, "byte")
        self.SID_HALF_WORD = self.bitvec(16, "half word")
        self.SID_SINGLE_WORD = self.bitvec(32, "single word")
        self.SID_DOUBLE_WORD = self.bitvec(64, "double word")
        self.SID_MACHINE_WORD = self.bitvec(w, f"{w}-bit machine word")
        self.SID_VIRTUAL_ADDRESS = self.bitvec(
            self.config.virtual_address_space, "virtual address"
        )
        self.SID_REGISTER_ADDRESS = self.bitvec(5, "register address")
        assert self.SID_REGISTER_ADDRESS is not None
        assert self.SID_MACHINE_WORD is not None
        self.SID_REGISTER_STATE = self.array(
            self.SID_REGISTER_ADDRESS, self.SID_MACHINE_WORD, "register state"
        )
        self.NID_FALSE = self.constd(self.SID_BOOLEAN, 0, "false")
        self.NID_TRUE = self.constd(self.SID_BOOLEAN, 1, "true")

    def _build_register_file(self) -> None:
        assert self.SID_REGISTER_STATE is not None
        for core in range(self.config.cores):
            symbol = f"core{core}-register-file" if self.config.cores > 1 else "register-file"
            regs = self.state(self.SID_REGISTER_STATE, symbol, "register file state")
            self._register_file[core] = regs
            self._state.state_nodes[symbol] = regs
            if getattr(self.config, "init_registers_to_zero", True):
                zero_array = self.zero(self.SID_REGISTER_STATE, "zeroed registers")
                self._state.init_nodes.append(
                    self.init(self.SID_REGISTER_STATE, regs, zero_array,
                              f"init {symbol}")
                )

    def _build_memory_segments(self) -> None:
        # Architecturally the machine has a single flat virtual address space.
        # We model ``code`` separately so code synthesis can mark it symbolic
        # without affecting the data heap/stack; everything else lives in a
        # unified byte-addressable ``memory`` state node.
        assert self.SID_VIRTUAL_ADDRESS is not None and self.SID_BYTE is not None
        mem_sort = self.array(self.SID_VIRTUAL_ADDRESS, self.SID_BYTE, "memory")
        for core in range(self.config.cores):
            code_symbol = f"core{core}-code" if self.config.cores > 1 else "code"
            mem_symbol = f"core{core}-memory" if self.config.cores > 1 else "memory"

            code = self.state(mem_sort, code_symbol, "code segment")
            self._memory_segments[(core, "code")] = code
            self._state.state_nodes[code_symbol] = code

            memory = self.state(mem_sort, mem_symbol, "data/heap/stack")
            self._memory_segments[(core, "memory")] = memory
            self._state.state_nodes[mem_symbol] = memory

            # Initialize unified memory to zero.
            zero_mem = self.zero(mem_sort, "zeroed memory")
            self._state.init_nodes.append(
                self.init(mem_sort, memory, zero_mem, f"init {mem_symbol}")
            )

    def initialize_code_segment(
        self, core: int, base_addr: int, data: bytes
    ) -> None:
        """Bake the bytes of ``data`` at ``base_addr`` into the code segment
        via a chain of BTOR2 ``write`` ops on a zero-initialized array.

        Called from :class:`~rotor.instance.RotorInstance` once the machine
        has been built and the binary's ``.text`` bytes are available.
        """
        self._initialize_segment(core, "code", base_addr, data, "code")

    def initialize_data_segment(
        self, core: int, base_addr: int, data: bytes
    ) -> None:
        """Bake ELF ``.data`` / ``.rodata`` bytes into the unified memory."""
        self._initialize_segment(core, "memory", base_addr, data, "memory")

    def _initialize_segment(
        self, core: int, segment: str, base_addr: int, data: bytes, label: str,
    ) -> None:
        assert self.SID_VIRTUAL_ADDRESS is not None and self.SID_BYTE is not None
        mem_sort = self.array(self.SID_VIRTUAL_ADDRESS, self.SID_BYTE, "memory")
        node = self._memory_segments[(core, segment)]

        # Start from the existing init value (zero array, or a prior
        # initialization). We rebuild by layering writes on top of zero.
        acc = self.zero(mem_sort, f"zero {label} segment")
        for offset, byte in enumerate(data):
            addr = self.constd(
                self.SID_VIRTUAL_ADDRESS, base_addr + offset,
                f"{label}[0x{base_addr + offset:x}]",
            )
            value = self.constd(self.SID_BYTE, byte, f"0x{byte:02x}")
            acc = self.write(acc, addr, value, f"{label}-byte")

        # Remove any prior init for this state, then install the new one.
        self._state.init_nodes = [
            n for n in self._state.init_nodes
            if not (n.op == "init" and n.args and n.args[0] is node)
        ]
        self._state.init_nodes.append(
            self.init(mem_sort, node, acc, f"init {label} segment")
        )

    def _build_fetch_decode_execute(self) -> None:
        """Create PC state nodes and wire up fetch/decode/execute.

        Full instruction semantics live in :mod:`rotor.btor2.riscv`. This
        method creates the per-core program-counter and halt latches, then
        delegates transition construction to the ISA module.
        """
        from rotor.btor2.riscv import build_fetch_decode_execute

        assert self.SID_MACHINE_WORD is not None and self.SID_BOOLEAN is not None
        for core in range(self.config.cores):
            pc_symbol = f"core{core}-pc" if self.config.cores > 1 else "pc"
            pc = self.state(self.SID_MACHINE_WORD, pc_symbol, "program counter")
            self._pc_nodes[core] = pc
            self._state.state_nodes[pc_symbol] = pc
            init_pc = self.consth(
                self.SID_MACHINE_WORD,
                self.config.code_start,
                f"initial pc = 0x{self.config.code_start:x}",
            )
            self._state.init_nodes.append(
                self.init(self.SID_MACHINE_WORD, pc, init_pc, f"init {pc_symbol}")
            )

            halt_symbol = f"core{core}-halted" if self.config.cores > 1 else "halted"
            halted = self.state(self.SID_BOOLEAN, halt_symbol, "halted")
            self._halted_nodes[core] = halted
            self._state.state_nodes[halt_symbol] = halted
            assert self.NID_FALSE is not None
            self._state.init_nodes.append(
                self.init(self.SID_BOOLEAN, halted, self.NID_FALSE,
                          f"init {halt_symbol}")
            )

            exit_symbol = f"core{core}-exit-code" if self.config.cores > 1 else "exit-code"
            exit_code = self.state(self.SID_MACHINE_WORD, exit_symbol, "exit code")
            self._exit_code_nodes[core] = exit_code
            self._state.state_nodes[exit_symbol] = exit_code
            self._state.init_nodes.append(
                self.init(self.SID_MACHINE_WORD, exit_code,
                          self.zero(self.SID_MACHINE_WORD, "0"),
                          f"init {exit_symbol}")
            )

            assert self.SID_BYTE is not None
            input_symbol = f"core{core}-input-byte" if self.config.cores > 1 else "input-byte"
            input_byte = self.input(self.SID_BYTE, input_symbol, "read syscall byte")
            self._input_byte_nodes[core] = input_byte
            self._state.input_nodes[input_symbol] = input_byte

        # Delegate per-core transition logic to the RISC-V ISA module.
        build_fetch_decode_execute(self)

    def _build_kernel_model(self) -> None:
        # Syscalls (exit, brk, openat, read, write) are deferred.
        pass

    def _build_properties(self) -> None:
        # Default property list is added by the caller via add_bad().
        pass

    # ----------------------------------------------------------- accessors

    def register_value(self, core: int, reg: int) -> Node:
        """Return the Node representing register ``reg`` of ``core``."""
        assert self.SID_REGISTER_ADDRESS is not None
        regs = self._register_file[core]
        idx = self.constd(self.SID_REGISTER_ADDRESS, reg, f"x{reg}")
        return self.read(regs, idx, f"core{core}.x{reg}")


# ──────────────────────────────────────────────────────────────────────────
# CRotorBackend (subprocess wrapper)
# ──────────────────────────────────────────────────────────────────────────


class CRotorBackend:
    """Invoke the C Rotor binary to generate BTOR2 for an ELF, then parse it.

    This is the primary Phase 2 path: we delegate the machine semantics to
    C Rotor and focus the Python layer on orchestration and reasoning.
    """

    def __init__(self, rotor_binary: str | None = None) -> None:
        if rotor_binary is None:
            rotor_binary = os.environ.get("ROTOR_BINARY", "rotor")
        self.binary = rotor_binary

    def available(self) -> bool:
        """Return True if the C Rotor binary is callable in the current env."""
        try:
            subprocess.run(
                [self.binary, "-h"],
                check=False,
                capture_output=True,
                timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    def build(self, elf_path: str, config: "ModelConfig") -> NodeDAG:
        """Generate BTOR2 via a subprocess and parse it into a NodeDAG."""
        from rotor.btor2.parser import parse_btor2

        args = self._config_to_args(config)
        result = subprocess.run(
            [self.binary, "-c", elf_path, *args, "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        return parse_btor2(result.stdout)

    def build_text(self, elf_path: str, config: "ModelConfig") -> str:
        """Generate BTOR2 text via a subprocess without parsing it."""
        args = self._config_to_args(config)
        result = subprocess.run(
            [self.binary, "-c", elf_path, *args, "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    @staticmethod
    def _config_to_args(config: "ModelConfig") -> list[str]:
        args: list[str] = []
        if config.is_64bit:
            args.append("-m64")
        else:
            args.append("-m32")
        if config.cores > 1:
            args.extend(["-cores", str(config.cores)])
        if config.heap_allowance:
            args.extend(["-heap", str(config.heap_allowance)])
        if config.stack_allowance:
            args.extend(["-stack", str(config.stack_allowance)])
        if config.riscu_only:
            args.append("-riscuonly")
        if not config.enable_rvc:
            args.append("-norvc")
        if config.symbolic_code:
            args.append("-codewordsize")
        if config.check_division_by_zero:
            args.append("-check-division-by-zero")
        return args
