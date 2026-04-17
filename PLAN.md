# Python Rotor: Implementation Plan

This document is the implementation plan for Python Rotor — an orchestration
layer and reasoning engine built on top of C Rotor. It is organized as a
sequence of phases, each delivering a usable increment of capability.

---

## Project Structure

```
rotor/
├── README.md
├── PLAN.md                        ← this file
├── pyproject.toml
├── rotor/
│   ├── __init__.py
│   ├── binary.py                  ← Phase 1: ELF + DWARF parsing
│   ├── btor2/
│   │   ├── __init__.py
│   │   ├── nodes.py               ← Phase 2: BTOR2 node DAG
│   │   ├── builder.py             ← Phase 2: RISC-V machine construction
│   │   └── printer.py             ← Phase 2: emit BTOR2 text
│   ├── instance.py                ← Phase 3: RotorInstance
│   ├── engine.py                  ← Phase 4: RotorEngine
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── base.py                ← Phase 3: solver interface
│   │   ├── bitwuzla.py            ← Phase 3: BMC backend
│   │   ├── btormc.py              ← Phase 3: BtorMC backend
│   │   ├── ic3.py                 ← Phase 6: IC3/PDR backends
│   │   └── portfolio.py           ← Phase 4: parallel portfolio
│   ├── trace.py                   ← Phase 5: source-level trace
│   ├── dwarf.py                   ← Phase 5: DWARF location evaluation
│   ├── cegar.py                   ← Phase 6: abstraction refinement
│   └── api.py                     ← Phase 7: high-level question API
├── tests/
│   ├── fixtures/                  ← small RISC-V ELF binaries for testing
│   ├── test_binary.py
│   ├── test_btor2.py
│   ├── test_instance.py
│   ├── test_engine.py
│   └── test_trace.py
├── examples/
│   ├── buffer_overflow.py
│   ├── equivalence_check.py
│   ├── input_synthesis.py
│   └── ic3_proof.py
└── tools/
    └── rotor.c                    ← symlink or submodule to selfie/tools/rotor.c
```

---

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `pyelftools` | ELF parsing and DWARF debug information |
| `bitwuzla` | Python bindings for Bitwuzla SMT solver (BMC backend) |
| `z3-solver` | Z3 Python API (fallback BMC backend) |
| `subprocess` | Interface to BtorMC, rIC3, AVR, ABC (external solvers) |

### Build / Dev

| Package | Purpose |
|---------|---------|
| `pytest` | Test runner |
| `riscv-gnu-toolchain` | Compile test fixtures to RISC-V ELF |
| `gcc` / `clang` with RISC-V target | Alternative compiler for fixtures |

### Optional

| Package | Purpose |
|---------|---------|
| `cffi` / `ctypes` | C extension binding to rotor.c for performance |
| `sarif-tools` | SARIF output for IDE integration |

---

## Phase 1: ELF and DWARF Parsing

**Goal:** Given a RISC-V ELF binary, extract everything needed to build a
model instance and to present results back in source terms.

**Deliverable:** `rotor/binary.py`

### 1.1 ELF Loading

```python
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection
from dataclasses import dataclass

@dataclass
class Segment:
    start: int      # virtual address
    size: int
    data: bytes

@dataclass
class Symbol:
    name: str
    address: int
    size: int

class RISCVBinary:
    def __init__(self, path: str):
        self._path = path
        self._elf = ELFFile(open(path, 'rb'))
        self._check_arch()
        self.code = self._load_segment('.text')
        self.data = self._load_segment('.data')
        self.rodata = self._load_segment('.rodata')
        self.symbols = self._load_symbols()
        self._dwarf = self._elf.get_dwarf_info()
        self._line_map = None      # lazy: built on first pc_to_source call
        self._die_cache = {}       # lazy: function DIEs indexed by address

    def _check_arch(self):
        machine = self._elf.header['e_machine']
        assert machine == 'EM_RISCV', f"Expected RISC-V ELF, got {machine}"
        self.is_64bit = self._elf.elfclass == 64

    def _load_segment(self, name: str) -> Segment | None: ...
    def _load_symbols(self) -> dict[str, Symbol]: ...
```

### 1.2 PC to Source Location

```python
@dataclass
class SourceLocation:
    file: str
    line: int
    column: int

def pc_to_source(self, pc: int) -> SourceLocation | None:
    """
    Use the DWARF line number program to map a PC to file:line:column.
    Builds the full line table on first call and caches it.
    """
    if self._line_map is None:
        self._line_map = self._build_line_map()
    return self._line_map.get(pc)

def _build_line_map(self) -> dict[int, SourceLocation]:
    result = {}
    for cu in self._dwarf.iter_CUs():
        lineprog = self._dwarf.line_program_for_CU(cu)
        for entry in lineprog.get_entries():
            if entry.state and not entry.state.end_sequence:
                result[entry.state.address] = SourceLocation(
                    file=lineprog['file_entry'][entry.state.file - 1].name
                             .decode('utf-8'),
                    line=entry.state.line,
                    column=entry.state.column
                )
    return result
```

### 1.3 Function Boundaries

```python
@dataclass
class FunctionInfo:
    name: str
    low_pc: int
    high_pc: int
    return_type: str
    parameters: list['VariableInfo']
    locals: list['VariableInfo']

def function_at(self, pc: int) -> FunctionInfo | None:
    """Return the function containing this PC, using DWARF DIEs."""

def function_by_name(self, name: str) -> FunctionInfo | None:
    """Find a function by name in the DWARF DIE tree."""

def function_bounds(self, name: str) -> tuple[int, int]:
    """Return (low_pc, high_pc) for the named function."""
    info = self.function_by_name(name)
    if info is None:
        # Fall back to symbol table
        sym = self.symbols.get(name)
        if sym:
            return (sym.address, sym.address + sym.size)
        raise KeyError(f"Function not found: {name}")
    return (info.low_pc, info.high_pc)
```

### 1.4 Live Variable Resolution

```python
@dataclass
class VariableInfo:
    name: str
    type_name: str
    byte_size: int
    location: 'DWARFLocation'   # abstract location descriptor

@dataclass
class DWARFLocation:
    """Describes where a variable lives at a given PC."""
    kind: str          # 'register', 'frame_offset', 'address', 'composite'
    register: int | None
    offset: int | None
    address: int | None

def live_variables_at(self, pc: int) -> list[VariableInfo]:
    """
    Find all variables in scope at this PC, with their locations.
    Uses DW_TAG_variable and DW_AT_location from the DIE tree,
    plus DW_LNE_set_address from location lists for variables
    whose location changes across their lifetime.
    """

def resolve_variable(self, var: VariableInfo,
                     registers: dict[int, int],
                     memory: dict[int, int]) -> int | None:
    """
    Evaluate a DWARF location expression against a concrete
    machine state to produce a concrete variable value.
    Implements the DWARF expression stack machine.
    """
```

**Tests:** Load several small RISC-V ELF binaries (compiled with `-g`),
verify PC→source, verify function bounds match `nm` output, verify variable
locations match GDB output at specific PCs.

---

## Phase 2: BTOR2 Node Layer

**Goal:** Represent the BTOR2 node DAG as Python objects, build the RISC-V
machine semantics in Python (mirroring C Rotor's logic), and emit valid BTOR2
text that C Rotor would also produce.

**Deliverable:** `rotor/btor2/nodes.py`, `builder.py`, `printer.py`

### 2.1 Node Objects

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Sort:
    nid: int
    kind: str          # 'bitvec' or 'array'
    width: int | None  # for bitvec
    index_sort: 'Sort | None'   # for array
    elem_sort: 'Sort | None'    # for array
    comment: str = ''

@dataclass
class Node:
    nid: int
    op: str
    sort: Sort
    args: list['Node'] = field(default_factory=list)
    params: list[int] = field(default_factory=list)   # for slice, ext
    symbol: str = ''                                   # for state, input
    comment: str = ''
    # Evaluation cache
    _concrete_value: int | None = None
    _is_symbolic: bool = False

class NodeDAG:
    """
    The global BTOR2 node DAG. Implements structural sharing:
    two nodes with identical structure get the same object
    (pointer equivalence implies structural equivalence,
    mirroring C Rotor's invariant).
    """
    def __init__(self):
        self._next_nid = 1
        self._nodes: dict[int, Node] = {}
        self._dedup: dict[tuple, Node] = {}

    def _key(self, op, sort, args, params, symbol) -> tuple:
        return (op, sort.nid,
                tuple(a.nid for a in args),
                tuple(params), symbol)

    def get_or_create(self, op, sort, args=(), params=(),
                      symbol='', comment='') -> Node:
        key = self._key(op, sort, args, params, symbol)
        if key in self._dedup:
            return self._dedup[key]
        node = Node(nid=self._next_nid, op=op, sort=sort,
                    args=list(args), params=list(params),
                    symbol=symbol, comment=comment)
        self._next_nid += 1
        self._nodes[node.nid] = node
        self._dedup[key] = node
        return node
```

### 2.2 Sort and Constant Constructors

```python
class BTOR2Builder:
    def __init__(self):
        self.dag = NodeDAG()
        self._sorts: dict[tuple, Sort] = {}

    def bitvec(self, width: int, comment='') -> Sort:
        key = ('bitvec', width)
        if key not in self._sorts:
            s = Sort(nid=self.dag._next_nid, kind='bitvec',
                     width=width, index_sort=None, elem_sort=None,
                     comment=comment)
            self.dag._next_nid += 1
            self._sorts[key] = s
        return self._sorts[key]

    def array(self, index_sort: Sort, elem_sort: Sort,
              comment='') -> Sort: ...

    def zero(self, sort: Sort, comment='') -> Node:
        return self.dag.get_or_create('zero', sort, comment=comment)

    def one(self, sort: Sort, comment='') -> Node:
        return self.dag.get_or_create('one', sort, comment=comment)

    def constd(self, sort: Sort, value: int, comment='') -> Node:
        return self.dag.get_or_create('constd', sort,
                                      params=[value], comment=comment)

    def consth(self, sort: Sort, value: int, comment='') -> Node:
        return self.dag.get_or_create('consth', sort,
                                      params=[value], comment=comment)
```

### 2.3 Operations

```python
    # Unary
    def not_(self, a: Node, comment='') -> Node: ...
    def inc(self, a: Node, comment='') -> Node: ...
    def dec(self, a: Node, comment='') -> Node: ...
    def neg(self, a: Node, comment='') -> Node: ...

    # Extension and slicing
    def sext(self, sort: Sort, a: Node, w: int, comment='') -> Node: ...
    def uext(self, sort: Sort, a: Node, w: int, comment='') -> Node: ...
    def slice(self, sort: Sort, a: Node, u: int, l: int,
              comment='') -> Node: ...

    # Binary arithmetic and logic
    def add(self, a: Node, b: Node, comment='') -> Node: ...
    def sub(self, a: Node, b: Node, comment='') -> Node: ...
    def mul(self, a: Node, b: Node, comment='') -> Node: ...
    def udiv(self, a: Node, b: Node, comment='') -> Node: ...
    def urem(self, a: Node, b: Node, comment='') -> Node: ...
    def sdiv(self, a: Node, b: Node, comment='') -> Node: ...
    def srem(self, a: Node, b: Node, comment='') -> Node: ...
    def and_(self, a: Node, b: Node, comment='') -> Node: ...
    def or_(self, a: Node, b: Node, comment='') -> Node: ...
    def xor(self, a: Node, b: Node, comment='') -> Node: ...
    def sll(self, a: Node, b: Node, comment='') -> Node: ...
    def srl(self, a: Node, b: Node, comment='') -> Node: ...
    def sra(self, a: Node, b: Node, comment='') -> Node: ...
    def concat(self, a: Node, b: Node, comment='') -> Node: ...

    # Comparison (result is bitvec 1)
    def eq(self, a: Node, b: Node, comment='') -> Node: ...
    def neq(self, a: Node, b: Node, comment='') -> Node: ...
    def ult(self, a: Node, b: Node, comment='') -> Node: ...
    def ugte(self, a: Node, b: Node, comment='') -> Node: ...
    def slt(self, a: Node, b: Node, comment='') -> Node: ...
    def sgte(self, a: Node, b: Node, comment='') -> Node: ...

    # Ternary
    def ite(self, c: Node, a: Node, b: Node, comment='') -> Node: ...

    # Array operations
    def read(self, array: Node, index: Node, comment='') -> Node: ...
    def write(self, array: Node, index: Node,
              value: Node, comment='') -> Node: ...

    # Sequential logic
    def state(self, sort: Sort, symbol: str, comment='') -> Node: ...
    def input(self, sort: Sort, symbol: str, comment='') -> Node: ...
    def init(self, sort: Sort, state: Node,
             value: Node, comment='') -> Node: ...
    def next(self, sort: Sort, state: Node,
             value: Node, comment='') -> Node: ...

    # Properties
    def bad(self, cond: Node, symbol: str, comment='') -> Node: ...
    def constraint(self, cond: Node, symbol: str,
                   comment='') -> Node: ...
```

### 2.4 Machine Builder

This is the core translation of C Rotor's machine semantics into Python.
Each subsystem corresponds to a function in `rotor.c`:

```python
class RISCVMachineBuilder(BTOR2Builder):
    """
    Builds the BTOR2 model of a RISC-V machine.
    Mirrors the structure of rotor.c:
      init_machine_interface()   → _build_sorts_and_constants()
      new_register_file_state()  → _build_register_file()
      new_memory_segments()      → _build_memory_segments()
      rotor_combinational()      → _build_fetch_decode_execute()
      kernel_combinational()     → _build_kernel_model()
      rotor_properties()         → _build_properties()
    """

    def __init__(self, config: 'ModelConfig'):
        super().__init__()
        self.config = config

    def build(self) -> 'MachineModel':
        self._build_sorts_and_constants()
        self._build_register_file()
        self._build_memory_segments()
        self._build_fetch_decode_execute()
        self._build_kernel_model()
        self._build_properties()
        return MachineModel(self.dag, self._state_nodes,
                            self._property_nodes)

    def _build_sorts_and_constants(self):
        # SID_BOOLEAN, SID_BYTE, SID_HALF_WORD, SID_SINGLE_WORD,
        # SID_DOUBLE_WORD, SID_MACHINE_WORD, SID_VIRTUAL_ADDRESS
        # NID_FALSE, NID_TRUE, NID_MACHINE_WORD_0..8, etc.
        self.SID_BOOLEAN = self.bitvec(1, "Boolean")
        self.NID_FALSE = self.constd(self.SID_BOOLEAN, 0, "false")
        self.NID_TRUE = self.constd(self.SID_BOOLEAN, 1, "true")
        w = 64 if self.config.is_64bit else 32
        self.SID_MACHINE_WORD = self.bitvec(w, f"{w}-bit machine word")
        # ... all constants from init_machine_interface() ...

    def _build_register_file(self):
        # SID_REGISTER_ADDRESS = bitvec(5)
        # SID_REGISTER_STATE = array(SID_REGISTER_ADDRESS, SID_MACHINE_WORD)
        # state_register_file = state(SID_REGISTER_STATE, ...)
        # init to zero, then write initial register values
        ...

    def _build_memory_segments(self):
        # Code, data, heap, stack segments as BTOR2 arrays
        # Initialized from binary or left symbolic for synthesis
        ...

    def _build_fetch_decode_execute(self):
        # The big one: mirrors rotor_combinational()
        # fetch_instruction → decode_instruction → data/control flow
        # Compressed instructions (RVC) if enabled
        ...

    def _build_kernel_model(self):
        # Syscall handling: exit, brk, openat, read, write
        # Program break, file descriptor, input buffer state
        ...

    def _build_properties(self):
        # Default: illegal instruction, segfault, division by zero
        # exit code checks
        # User-specified properties added via add_bad() / add_constraint()
        ...
```

**Strategy:** Rather than reimplementing all of C Rotor's instruction
semantics from scratch, the initial implementation of Phase 2 will use
**C Rotor as a subprocess** to generate BTOR2 text, then parse that text into
the Python node DAG. This validates the DAG representation and printer while
deferring the full Python semantics reimplementation.

```python
class CRotorBackend:
    """
    Calls the C Rotor binary to generate BTOR2, parses the output
    into a Python NodeDAG. Serves as the initial Phase 2 implementation
    and as a correctness oracle for the full Python reimplementation.
    """
    def __init__(self, rotor_binary: str = 'rotor'):
        self.binary = rotor_binary

    def build(self, elf_path: str, config: ModelConfig) -> NodeDAG:
        args = self._config_to_args(config)
        result = subprocess.run(
            [self.binary, '-'] + args + [elf_path, '-'],
            capture_output=True, text=True
        )
        return parse_btor2(result.stdout)
```

### 2.5 BTOR2 Printer

```python
class BTOR2Printer:
    """
    Emits a NodeDAG as BTOR2 text.
    Performs a topological traversal, assigning nids in order,
    exploiting structural sharing (each unique node emitted once).
    """
    def print(self, dag: NodeDAG,
              properties: list[Node],
              out: IO[str]) -> None: ...
```

**Tests:** Parse C Rotor output, round-trip through Python printer, diff
against original. The output should be semantically equivalent (same nid
assignment may differ but structure must match).

---

## Phase 3: RotorInstance and Solver Interface

**Goal:** A live Python object representing one BTOR2 model instance,
with the ability to add user-specified properties, emit to a solver,
and retrieve results.

**Deliverable:** `rotor/instance.py`, `rotor/solvers/`

### 3.1 Model Configuration

```python
@dataclass
class ModelConfig:
    # Target architecture
    is_64bit: bool = True
    word_size: int = 64

    # Code
    code_start: int = 0
    code_end: int = 0
    symbolic_code: bool = False       # synthesis mode
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
    enable_rvc: bool = True           # compressed instructions
    enable_rv64m: bool = True         # multiply/divide
    riscu_only: bool = False          # restrict to RISC-U subset

    # Solver
    solver: str = 'bitwuzla'          # 'bitwuzla', 'btormc', 'ic3'
    bound: int = 1000

    # Checks
    check_division_by_zero: bool = True
    check_seg_faults: bool = True
    check_invalid_addresses: bool = True
    check_bad_exit_code: bool = True
```

### 3.2 RotorInstance

```python
class RotorInstance:
    def __init__(self, binary: RISCVBinary, config: ModelConfig):
        self.binary = binary
        self.config = config
        self._builder = CRotorBackend()   # Phase 2 subprocess approach
        self._model: MachineModel | None = None
        self._extra_bads: list[tuple[Node, str]] = []
        self._extra_constraints: list[tuple[Node, str]] = []

    def build_machine(self) -> None:
        """Generate the BTOR2 machine model."""
        self._model = self._builder.build(self.binary._path, self.config)

    def add_bad(self, condition: Node, name: str) -> None:
        """Add a user-specified bad property."""
        self._extra_bads.append((condition, name))

    def add_constraint(self, condition: Node, name: str) -> None:
        """Add a user-specified constraint (good property)."""
        self._extra_constraints.append((condition, name))

    def pc_equals(self, target_pc: int) -> Node:
        """Return a BTOR2 node that is true when PC == target_pc."""
        pc_node = self._model.state_nodes['pc']
        return self._model.builder.eq(
            pc_node,
            self._model.builder.consth(
                self._model.builder.SID_MACHINE_WORD,
                target_pc, f"pc == 0x{target_pc:x}"
            )
        )

    def outputs_differ(self) -> Node:
        """
        For multicore models: return a node that is true when the
        return values of core 0 and core 1 diverge.
        Used for semantic equivalence checking.
        """
        assert self.config.cores >= 2
        a0_core0 = self._model.register_value(core=0, reg=10)  # a0
        a0_core1 = self._model.register_value(core=1, reg=10)
        return self._model.builder.neq(a0_core0, a0_core1,
                                       "output registers differ")

    def check(self, bound: int) -> 'CheckResult':
        """
        Emit the model to the configured solver and return the result.
        """
        solver = self._make_solver()
        btor2_text = self._emit_btor2()
        return solver.check(btor2_text, bound)

    def get_witness(self) -> list['MachineState']:
        """
        After a SAT result, retrieve the concrete execution trace
        (sequence of PC + register + memory states).
        """
        ...

    def _emit_btor2(self) -> str:
        printer = BTOR2Printer()
        buf = io.StringIO()
        all_bads = (self._model.property_nodes +
                    [b for b, _ in self._extra_bads])
        all_constraints = [c for c, _ in self._extra_constraints]
        printer.print(self._model.dag, all_bads, all_constraints, buf)
        return buf.getvalue()

    def _make_solver(self) -> 'SolverBackend':
        backends = {
            'bitwuzla': BitwuzlaSolver,
            'btormc':   BtorMCSolver,
            'ic3':      IC3Solver,
        }
        return backends[self.config.solver](self.config)
```

### 3.3 Solver Interface

```python
from abc import ABC, abstractmethod

@dataclass
class CheckResult:
    verdict: str           # 'sat', 'unsat', 'unknown'
    steps: int | None      # steps to bad state (sat only)
    witness: list[dict] | None   # raw solver witness (sat only)
    invariant: str | None  # inductive invariant (unsat via IC3 only)
    solver: str
    elapsed: float

class SolverBackend(ABC):
    @abstractmethod
    def check(self, btor2: str, bound: int) -> CheckResult: ...

class BitwuzlaSolver(SolverBackend):
    """
    Uses Bitwuzla's Python API directly (no subprocess).
    Translates the NodeDAG to Bitwuzla terms natively for
    incremental solving and model querying.
    """
    def check(self, btor2: str, bound: int) -> CheckResult:
        import bitwuzla
        tm = bitwuzla.TermManager()
        parser = bitwuzla.Parser(tm, bitwuzla.Options())
        parser.parse(btor2, format='btor2')
        # ... BMC unrolling and solving ...

class BtorMCSolver(SolverBackend):
    """
    Calls BtorMC as a subprocess. Supports both BMC and IC3 modes
    depending on the algorithm flag.
    """
    def check(self, btor2: str, bound: int) -> CheckResult:
        with tempfile.NamedTemporaryFile(suffix='.btor2', mode='w') as f:
            f.write(btor2)
            f.flush()
            result = subprocess.run(
                ['btormc', f'--bound={bound}', f.name],
                capture_output=True, text=True, timeout=300
            )
        return self._parse_result(result.stdout)

class IC3Solver(SolverBackend):
    """
    Calls rIC3, AVR, or ABC's PDR engine for unbounded IC3 reasoning.
    Returns either a SAT result with counterexample or an UNSAT result
    with the inductive invariant discovered.
    """
    def __init__(self, config: ModelConfig,
                 backend: str = 'ric3'):
        self.backend = backend   # 'ric3', 'avr', 'abc'
    ...
```

**Tests:** Check small hand-crafted BTOR2 models (counter, flag protocol),
verify SAT/UNSAT verdicts match expected, verify witness format.

---

## Phase 4: RotorEngine — Orchestration

**Goal:** High-level engine that creates and coordinates multiple
RotorInstance objects, implements compositional strategies, and runs
parallel portfolios.

**Deliverable:** `rotor/engine.py`, `rotor/solvers/portfolio.py`

### 4.1 Core Engine

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class RotorEngine:
    def __init__(self, binary_path: str,
                 default_solver: str = 'bitwuzla',
                 default_bound: int = 1000):
        self.binary = RISCVBinary(binary_path)
        self.default_solver = default_solver
        self.default_bound = default_bound

    def create_instance(self, function: str | None = None,
                        config: ModelConfig | None = None
                        ) -> RotorInstance:
        cfg = config or ModelConfig(solver=self.default_solver,
                                    bound=self.default_bound)
        if function:
            low, high = self.binary.function_bounds(function)
            cfg.code_start = low
            cfg.code_end = high
        inst = RotorInstance(self.binary, cfg)
        inst.build_machine()
        return inst

    def run_portfolio(self,
                      instances: list[RotorInstance],
                      timeout: float = 300.0
                      ) -> list[CheckResult]:
        """
        Run all instances in parallel. Returns all results collected
        within the timeout. Cancels remaining instances when the first
        conclusive (sat or unsat) result arrives.
        """
        results = []
        with ThreadPoolExecutor(max_workers=len(instances)) as ex:
            futures = {
                ex.submit(inst.check, inst.config.bound): inst
                for inst in instances
            }
            try:
                for future in as_completed(futures,
                                           timeout=timeout):
                    result = future.result()
                    results.append(result)
                    if result.verdict in ('sat', 'unsat'):
                        # Cancel remaining (best effort)
                        for f in futures:
                            f.cancel()
                        break
            except TimeoutError:
                pass
        return results
```

### 4.2 Compositional Verification

```python
    def verify_compositionally(self,
                               call_graph: dict[str, list[str]],
                               specs: dict[str, 'Specification'],
                               bound: int = 1000
                               ) -> dict[str, CheckResult]:
        """
        Verify each function independently, in callee-before-caller order.
        Verified callee specs become constraints when verifying callers,
        avoiding state space explosion from inlining.
        """
        results = {}
        for function in topological_sort(call_graph):
            callee_constraints = {
                f: specs[f].postcondition
                for f in call_graph[function]
                if f in results and results[f].verdict == 'unsat'
            }
            inst = self.create_instance(function)
            inst.add_constraint(specs[function].precondition,
                                "precondition")
            for callee, post in callee_constraints.items():
                inst.add_constraint(post,
                                    f"callee-{callee}-postcondition")
            inst.add_bad(specs[function].postcondition_negated,
                         "postcondition-violation")
            results[function] = inst.check(bound)
        return results
```

### 4.3 Concolic Exploration

```python
    def explore_paths(self,
                      function: str,
                      max_paths: int = 100,
                      bound: int = 1000
                      ) -> list['SourceTrace']:
        """
        Systematically explore execution paths using a concolic strategy:
        check one path condition, then negate branch decisions to fork
        into unexplored paths. Returns source-annotated traces.
        """
        paths = []
        worklist = [PathCondition.empty()]

        while worklist and len(paths) < max_paths:
            condition = worklist.pop()
            inst = self.create_instance(function)
            inst.add_constraint(condition.as_node(inst), "path-condition")
            inst.build_machine()
            result = inst.check(bound)

            if result.verdict == 'sat':
                trace = self._witness_to_source_trace(
                    inst.get_witness())
                paths.append(trace)
                for branch in result.branch_points:
                    worklist.append(condition.extend(
                        branch.negated()))
        return paths
```

**Tests:** Compositional verification of a two-function program where the
whole-program model is too large for direct BMC but the per-function models
are tractable.

---

## Phase 5: Source-Level Trace and DWARF Evaluation

**Goal:** Transform a solver witness (concrete machine state sequence) into a
human-readable, source-annotated trace using DWARF information.

**Deliverable:** `rotor/trace.py`, `rotor/dwarf.py`

### 5.1 Machine State

```python
@dataclass
class MachineState:
    step: int
    pc: int
    registers: dict[int, int]   # reg number → value
    memory: dict[int, int]      # address → byte value (sparse)
```

### 5.2 Source Step

```python
@dataclass
class SourceStep:
    step: int
    location: SourceLocation
    function_name: str
    instruction: str              # disassembled RISC-V instruction
    variables: dict[str, int]     # variable name → concrete value
    changed: set[str]             # variables that changed this step

class SourceTrace:
    def __init__(self, machine_states: list[MachineState],
                 binary: RISCVBinary):
        self.steps = []
        prev_vars: dict[str, int] = {}
        for state in machine_states:
            loc = binary.pc_to_source(state.pc)
            func = binary.function_at(state.pc)
            live = binary.live_variables_at(state.pc)
            variables = {}
            for var in live:
                value = binary.resolve_variable(
                    var, state.registers, state.memory)
                if value is not None:
                    variables[var.name] = value
            changed = {k for k, v in variables.items()
                       if prev_vars.get(k) != v}
            self.steps.append(SourceStep(
                step=state.step,
                location=loc,
                function_name=func.name if func else '?',
                instruction=binary.disassemble(state.pc),
                variables=variables,
                changed=changed
            ))
            prev_vars = variables

    def __str__(self) -> str: ...
    def as_sarif(self) -> dict: ...
    def as_gdb_script(self) -> str: ...
    def as_markdown(self) -> str: ...
```

### 5.3 DWARF Location Expression Evaluator

```python
class DWARFExprEvaluator:
    """
    Evaluates DWARF expression opcodes (DW_OP_*) against a concrete
    machine state to determine where a variable's value lives.
    Implements the DWARF 5 expression stack machine.
    """
    def __init__(self, registers: dict[int, int],
                 memory: dict[int, int],
                 frame_base: int):
        self.regs = registers
        self.mem = memory
        self.frame_base = frame_base
        self.stack: list[int] = []

    def evaluate(self, expr: bytes) -> int | None:
        """
        Interpret a DWARF location expression and return the value
        (for DW_OP_stack_value) or address (for memory locations).
        """
        ops = self._parse_opcodes(expr)
        for op, args in ops:
            self._execute_op(op, args)
        return self.stack[-1] if self.stack else None

    def _execute_op(self, op: int, args: list[int]) -> None:
        # DW_OP_lit0..DW_OP_lit31: push literal
        # DW_OP_reg0..DW_OP_reg31: push register value
        # DW_OP_breg0..DW_OP_breg31: push register + signed offset
        # DW_OP_fbreg: push frame_base + signed offset
        # DW_OP_addr: push absolute address
        # DW_OP_deref: pop address, push memory[address]
        # DW_OP_plus, DW_OP_minus, DW_OP_mul, DW_OP_and, DW_OP_or...
        # DW_OP_stack_value: value is top of stack (not an address)
        ...
```

**Tests:** Compile a small C program with `-g -O0`, step through with GDB,
record variable values at specific PCs, verify DWARFExprEvaluator produces
identical values given the same register/memory state.

---

## Phase 6: IC3 and Unbounded Reasoning

**Goal:** Integrate IC3/PDR backends to provide unbounded safety proofs,
and implement CEGAR for programs with larger state spaces.

**Deliverable:** `rotor/solvers/ic3.py`, `rotor/cegar.py`

### 6.1 IC3 Backend Selection

```python
class IC3Solver(SolverBackend):
    """
    Tries IC3 backends in order of performance on the given model.
    Uses the rIC3, AVR, and ABC/PDR backends.
    Returns PROVED with an inductive invariant, or SAT with a trace,
    or UNKNOWN if resource limits are hit.
    """
    BACKENDS = ['ric3', 'avr', 'abc']

    def check(self, btor2: str, bound: int) -> CheckResult:
        # Write to temp file, try backends in parallel
        ...

    def _parse_invariant(self, output: str) -> str | None:
        """
        Extract the inductive invariant from the solver output.
        Invariants from IC3 are clause sets (CNF over state variables).
        """
        ...
```

### 6.2 Deciding When to Use IC3

```python
class RotorEngine:
    ...

    def _select_solver(self, function: str,
                       property_type: str) -> str:
        """
        Heuristically choose BMC or IC3 based on the function's
        structure and the property type.
        """
        info = self.binary.function_at(
            self.binary.function_bounds(function)[0])
        state_estimate = self._estimate_state_space(function)

        if state_estimate < 10_000:
            return 'ic3'        # small enough for direct IC3
        elif property_type in ('reachability', 'synthesis'):
            return 'bitwuzla'   # BMC is better for finding witnesses
        else:
            return 'portfolio'  # let all backends compete
```

### 6.3 CEGAR Loop

```python
class CEGARVerifier:
    """
    Counterexample-Guided Abstraction Refinement.
    Iteratively abstracts the BTOR2 model, runs IC3 on the
    abstraction, and refines when the counterexample is spurious.
    Achieves unbounded verification for programs with large state
    spaces by working in a coarser abstract domain.
    """
    def __init__(self, engine: RotorEngine, max_iterations: int = 20):
        self.engine = engine
        self.max_iterations = max_iterations

    def verify(self, function: str,
               property_bad: Node,
               bound: int = 1000) -> CheckResult:

        abstract_inst = self.engine.create_instance(function)
        abstract_inst = self._abstract(abstract_inst)

        for iteration in range(self.max_iterations):
            result = abstract_inst.check(bound)

            if result.verdict == 'unsat':
                # IC3 found proof in abstract model
                # Validate that invariant lifts to concrete model
                if self._invariant_holds_concretely(
                        function, result.invariant):
                    return result   # genuine proof
                else:
                    # Spurious proof: refine abstraction
                    abstract_inst = self._refine_for_invariant(
                        abstract_inst, result.invariant)

            elif result.verdict == 'sat':
                # Check if counterexample is real
                concrete = self.engine.create_instance(function)
                concrete_result = self._check_trace_concrete(
                    concrete, result.witness, bound)

                if concrete_result.verdict == 'sat':
                    # Genuine bug
                    trace = self.engine.binary
                    return concrete_result
                else:
                    # Spurious: refine abstraction
                    abstract_inst = self._refine(
                        abstract_inst, result.witness)
            else:
                return result  # unknown, resource exhausted

        return CheckResult(verdict='unknown',
                           steps=None, witness=None,
                           invariant=None,
                           solver='cegar',
                           elapsed=0.0)

    def _abstract(self, inst: RotorInstance) -> RotorInstance: ...
    def _refine(self, inst: RotorInstance,
                spurious_trace: list) -> RotorInstance: ...
    def _invariant_holds_concretely(self, function: str,
                                     invariant: str) -> bool: ...
```

**Tests:** Small programs with bounded loops that IC3 can prove safe in
under 1 second; larger programs where CEGAR is needed; programs with genuine
bugs where IC3 finds the counterexample.

---

## Phase 7: High-Level Question API

**Goal:** A clean, high-level API that directly maps to the question
categories an LLM (or a human) would ask about a binary.

**Deliverable:** `rotor/api.py`

```python
class RotorAPI:
    """
    The primary interface for asking questions about a RISC-V binary.
    Each method accepts natural-language or structured specifications
    and returns source-annotated answers.
    """

    def __init__(self, binary_path: str):
        self.engine = RotorEngine(binary_path)

    # ── Reachability ──────────────────────────────────────────────────

    def can_reach(self,
                  function: str,
                  condition: str,
                  bound: int = 1000,
                  unbounded: bool = False) -> ReachResult:
        """
        Can execution of `function` ever reach a state where `condition`
        holds?

        condition: a string expression over variable names and register
        names, e.g. "retval < 0" or "pc == 0x1040" or "buf[i] == 0"

        Returns REACHABLE with a concrete source trace, or
                UNREACHABLE with a proof (if unbounded=True and IC3
                succeeds), or UNKNOWN.
        """

    # ── Input Synthesis ───────────────────────────────────────────────

    def find_input(self,
                   function: str,
                   output_condition: str,
                   bound: int = 1000) -> InputResult:
        """
        Find a concrete input to `function` that produces a state
        satisfying `output_condition`.

        Returns INPUT with concrete bytes and a source trace, or
                NO_SUCH_INPUT with a proof (up to bound).
        """

    # ── Equivalence ───────────────────────────────────────────────────

    def are_equivalent(self,
                       other_binary: str,
                       function: str,
                       bound: int = 1000,
                       unbounded: bool = False) -> EquivResult:
        """
        Do `function` in this binary and `function` in `other_binary`
        produce identical outputs for all inputs?

        Returns EQUIVALENT with a proof (if unbounded=True and IC3
                succeeds), or DIFFERS with the diverging input and
                both source traces.
        """

    # ── Property Verification ─────────────────────────────────────────

    def verify(self,
               function: str,
               invariant: str,
               bound: int = 1000,
               unbounded: bool = False) -> VerifyResult:
        """
        Does `invariant` hold at every step of `function`'s execution?

        Returns HOLDS with a proof (unbounded if IC3 succeeds, bounded
                otherwise), or VIOLATED with a counterexample trace.
        """

    # ── Causality ─────────────────────────────────────────────────────

    def find_responsible_inputs(self,
                                function: str,
                                condition: str,
                                known_input: bytes,
                                bound: int = 1000) -> CausalResult:
        """
        Given a known input that triggers `condition`, find the minimal
        subset of input bytes whose values determine whether `condition`
        holds. Uses binary search over input byte subsets.

        Returns a CausalResult identifying the critical bytes with an
        explanation in source terms.
        """

    # ── Synthesis ─────────────────────────────────────────────────────

    def synthesize_value(self,
                         function: str,
                         hole_location: str,
                         spec: str,
                         bound: int = 100) -> SynthResult:
        """
        Find a concrete value at `hole_location` (expressed as a
        source file:line reference or a symbol name) such that `spec`
        holds for all executions of `function`.

        Uses C Rotor's code synthesis mode: the instruction(s) at the
        hole location are made symbolic and the solver finds values that
        satisfy the spec.

        Returns a SynthResult with the concrete value and the
        disassembled instruction, or UNSATISFIABLE.
        """

    # ── Portfolio / Adaptive ──────────────────────────────────────────

    def _select_strategy(self,
                         function: str,
                         question_type: str,
                         unbounded: bool) -> list[tuple[str, dict]]:
        """
        Choose solver configurations based on function complexity
        and question type. Returns a list of (solver, options) tuples
        for portfolio execution.
        """
        info = self.engine.binary.function_at(
            self.engine.binary.function_bounds(function)[0])

        configs = []

        if question_type in ('reachability', 'input_synthesis'):
            configs.append(('bitwuzla', {'bound': 100}))
            configs.append(('bitwuzla', {'bound': 1000}))

        if unbounded or question_type == 'verification':
            configs.append(('ic3', {'backend': 'ric3'}))
            configs.append(('ic3', {'backend': 'avr'}))

        if question_type == 'equivalence':
            configs.append(('ic3', {'backend': 'ric3',
                                    'cores': 2}))

        return configs
```

**Tests:** End-to-end tests using real C programs compiled for RISC-V.
Each question type tested against programs with known bugs (verify bug found)
and known-safe programs (verify no false positives, and IC3 proof obtained
for small programs).

---

## Milestones and Timeline

| Phase | Deliverable | Estimated Effort |
|-------|-------------|-----------------|
| 1 | ELF + DWARF parsing (`binary.py`) | 2–3 weeks |
| 2 | BTOR2 node DAG + C Rotor subprocess backend | 2–3 weeks |
| 3 | RotorInstance + Bitwuzla + BtorMC solvers | 2–3 weeks |
| 4 | RotorEngine, portfolio, compositional | 2 weeks |
| 5 | Source trace, DWARF evaluator, output formats | 2–3 weeks |
| 6 | IC3 backends, CEGAR loop | 3–4 weeks |
| 7 | High-level question API | 1–2 weeks |

Total estimated effort: **4–5 months** for a single developer, less with
parallel work on independent phases.

---

## Testing Strategy

### Fixture Binaries

Maintain a set of small RISC-V ELF binaries in `tests/fixtures/`, each
compiled from a known C or Rust source with known properties:

- `counter.c`: Simple loop with a known overflow point
- `buffer.c`: Buffer with a known overflow input
- `sort.c`: Sorting function, provably correct for array length ≤ 8
- `protocol.c`: State machine with a known bad state
- `equivalent_a.c` / `equivalent_b.c`: Two provably equivalent implementations
- `inequivalent_a.c` / `inequivalent_b.c`: Two with a known diverging input

### Test Levels

```
Unit tests:     Each module in isolation with synthetic inputs
Integration:    Full pipeline on fixture binaries, expected verdicts
Regression:     C Rotor output vs. Python builder output (structural diff)
Performance:    Track solve times across commits, alert on regression
```

### CI Pipeline

```yaml
- Build C Rotor from selfie submodule
- Compile fixture binaries (riscv64-linux-gnu-gcc -g)
- Run pytest (unit + integration)
- Run regression diff against C Rotor output
- Report solver time table
```

---

## Relationship to C Rotor

Python Rotor depends on C Rotor in two ways:

1. **Phase 2 bootstrap**: C Rotor generates BTOR2 as a subprocess. Python
   parses the output into its node DAG. This gets the system working quickly
   without reimplementing all instruction semantics in Python.

2. **Long-term**: As the Python machine builder (`RISCVMachineBuilder`)
   matures, it becomes the primary model generator, with C Rotor serving as
   a correctness oracle. The two implementations must agree on the BTOR2
   output for all test binaries.

C Rotor is included as a Git submodule pointing to
`https://github.com/cksystemsgroup/selfie`.

```
git submodule add https://github.com/cksystemsgroup/selfie selfie
ln -s selfie/tools/rotor.c tools/rotor.c
```

The build system compiles `rotor.c` as part of the Python package
installation, making the C binary available at `rotor/_bin/rotor`.