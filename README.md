# Rotor

**Rotor** is a reasoning engine for any code that compiles to RISC-V. It uses
bit-precise formal verification — bounded model checking, IC3/PDR, and
counterexample-guided abstraction refinement — to answer questions about
compiled binaries with mathematical certainty, and maps all results back to
the original source code via DWARF debug information.

This repository contains a **Python reimplementation and orchestration layer**
built on top of the original C implementation of Rotor, which is part of the
[Selfie Project](https://github.com/cksystemsgroup/selfie) at the
Computational Systems Group, University of Salzburg.

---

## Quick Start by Example

Install from the repository root:

```bash
pip install -e .[bitwuzla]      # optional extras: bitwuzla, z3, sarif, dev
```

### Five-minute CLI walkthrough

Everything below uses `tests/fixtures/add2.elf` — a two-argument sum
function. Swap in your own `.elf` (compiled with `-g`) to try it on real
code.

```console
$ rotor info tests/fixtures/add2.elf --functions
path      : tests/fixtures/add2.elf
is_64bit  : True
entry     : 0x1118c
code      : 0x11158..0x11190 (56 bytes)
functions:
  0x00011158 +52   add2
  0x0001118c +36   _start

$ rotor disasm tests/fixtures/add2.elf --function add2
  0x00011158: addi sp, sp, -32       ; add2.c:4:0
  0x0001115c: sd ra, 24(sp)          ; add2.c:4:0
  ...
  0x00011178: addw a0, a0, a1        ; add2.c:5:14
  0x00011188: ret                    ; add2.c:5:5

$ rotor analyze tests/fixtures/add2.elf --function add2
range    : 0x11158..0x1118c (13 instrs)
status   : clean

$ rotor reach tests/fixtures/add2.elf --function add2 \
      --condition "pc == 0x11178" --bound 10
verdict  : reachable
elapsed  : 0.31s
| step | location    | function | pc       | instruction      | ...

$ rotor verify tests/fixtures/add2.elf --function add2 \
      --invariant "0 == 0" --unbounded --bound 4
verdict  : holds
unbounded: True
elapsed  : 0.13s
inductive invariant:
  1-inductive invariant: for all reachable states, none of the bad
  properties holds: 'invariant-violated:'
```

The CLI subcommands map directly to the Python API: `reach` →
`RotorAPI.can_reach`, `find-input` → `RotorAPI.find_input`, `verify` →
`RotorAPI.verify`, `equivalent` → `RotorAPI.are_equivalent`. Exit codes
follow a predictable convention: `0` for "safe / proved / equivalent",
`1` for "something found (bug / reachable / diverges)", `2` for
"unknown or error".

### 1. Load a binary and inspect it

```python
from rotor import RISCVBinary

with RISCVBinary("path/to/program.elf") as binary:
    print(binary)                         # RISCVBinary('...', rv64, entry=0x10078)
    print(binary.function_bounds("main")) # (0x101a0, 0x101e4)
    print(binary.pc_to_source(0x101b4))   # main.c:12:5
```

Equivalent from the command line:

```bash
rotor info path/to/program.elf
rotor btor2 path/to/program.elf -o program.btor2
```

### 2. Ask a reachability question

`RotorAPI` is the one-line entry point. Conditions are written in a compact
bitvector expression language over registers (`x0..x31`, ABI names like `a0`,
`sp`, `ra`), `pc`, and integer literals:

```python
from rotor import RotorAPI

api = RotorAPI("path/to/program.elf", default_solver="bitwuzla",
               default_bound=200)

result = api.can_reach(function="check_input",
                       condition="a0 < 0",
                       bound=200)

if result.verdict == "reachable":
    print(result.trace.as_markdown())    # source-annotated counterexample
elif result.verdict == "unreachable":
    print("proved safe up to bound")
```

### 3. Synthesize an input

```python
result = api.find_input(function="parse",
                        output_condition="a0 == 0xdeadbeef",
                        bound=500)

if result.verdict == "found":
    print(f"input bytes: {result.input_bytes!r}")
    print(result.trace.as_markdown())
```

### 4. Prove an invariant with IC3

```python
result = api.verify(function="counter",
                    invariant="x10 <= 100",
                    bound=1000,
                    unbounded=True)

if result.verdict == "holds" and result.unbounded:
    print("inductive invariant:")
    print(result.proof)
elif result.verdict == "violated":
    print(result.counterexample.as_markdown())
```

### 5. Check semantic equivalence of two binaries

```python
result = api.are_equivalent(other_binary="path/to/refactored.elf",
                            function="sort",
                            bound=500,
                            unbounded=False)

print(result.verdict)                    # 'equivalent' | 'differs' | 'unknown'
```

### 6. Drop into lower levels when needed

```python
from rotor import RotorEngine, ModelConfig

engine = RotorEngine("path/to/program.elf")
inst = engine.create_instance(
    function="foo",
    config=ModelConfig(solver="btormc", bound=100, cores=1),
)
inst.add_bad(inst.pc_equals(0x10240), "reached-error-handler")
print(inst.check().verdict)
```

Further runnable scripts live in
[`examples/`](examples): `buffer_overflow.py`, `input_synthesis.py`,
`equivalence_check.py`, `ic3_proof.py`.

---

## Motivation

Compilers for C, C++, Rust, Go, Ada, Zig, and many other languages target
RISC-V. Rather than reasoning about source code — which requires trusting the
compiler's semantic preservation — Rotor reasons about the actual binary that
runs on the machine. It then uses DWARF debug information to present results
in source terms: file names, line numbers, variable names, and concrete values.

The result is a tool that is simultaneously more trustworthy than source-level
analyzers and more usable than raw binary analysis tools.

---

## Foundation: C Rotor

The core RISC-V semantic encoding is provided by **Rotor in C**, available at:

> **[github.com/cksystemsgroup/selfie](https://github.com/cksystemsgroup/selfie)**
> (`tools/rotor.c`)

C Rotor translates a RISC-V ELF binary — or an uninitialized machine for code
synthesis — into a **BTOR2** model in linear time and space in the size of the
binary. BTOR2 is a word-level model checking format over bitvectors and arrays
of bitvectors, supported by state-of-the-art hardware model checkers including
Bitwuzla, BtorMC, AVR, rIC3, and ABC.

The key semantic reduction C Rotor establishes is:

> *For all k > 0, the SMT formula unrolled from the BTOR2 model k times is
> satisfiable if and only if there exists machine input such that a bad state
> is reached after executing up to k+1 machine instructions.*

C Rotor supports full RV64I/RV32I with multiplication and division (RV64M,
RV32M) and compressed instructions (RVC), generating models in linear time in
the size of the binary. It also supports multiple cores for semantic
equivalence checking of two RISC-V binaries.

---

## Python Rotor: Architecture

Python Rotor transforms C Rotor from a one-shot model generator into a
**multi-instance reasoning engine**:

```
Binary (ELF + DWARF)
        │
        ▼
┌───────────────────┐
│   RISCVBinary     │  ← DWARF: PC → source, variables, types
│   (pyelftools)    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   RotorEngine     │  ← orchestration, strategy, portfolio
│   (Python)        │
└────────┬──────────┘
         │  spawns many
         ▼
┌───────────────────┐
│  RotorInstance    │  ← one BTOR2 model per question/hypothesis
│  (C Rotor core)   │
└────────┬──────────┘
         │
         ▼
┌────────────────────────────────────┐
│  Solver Backend (choose per query) │
│  BMC: Bitwuzla / Z3                │
│  IC3: rIC3 / AVR / ABC             │
│  CEGAR: CPAchecker / SeaHorn       │
└────────────────────────────────────┘
         │
         ▼
┌───────────────────┐
│   Source Trace    │  ← DWARF: machine state → file:line:variable=value
└───────────────────┘
```

Multiple instances run in parallel, each targeting a different question,
function, bound, or solver. Results are collected and merged at the Python
level into source-annotated answers.

---

## Reasoning Modes

### Bounded Model Checking (BMC)

For a given bound k, BMC unrolls the BTOR2 machine model k times and asks an
SMT solver whether any bad state is reachable. This finds concrete bugs fast
but cannot prove safety — only that no bug exists within k steps.

**Answer quality:** BUG (with concrete input and source trace) or
SAFE-UP-TO-k (incomplete).

### IC3 / Property-Directed Reachability (PDR)

IC3 searches for an **inductive invariant** that separates initial states from
bad states, without bounding the number of steps. If found, this is an
unbounded safety proof: the property holds for all executions of any length.

**Answer quality:** BUG (with concrete counterexample) or PROVED (invariant
verified for all executions, with no bound). For finite-state components of a
RISC-V execution — bounded loops, fixed-size buffers, protocol state machines
— IC3 typically terminates quickly.

### CEGAR (Counterexample-Guided Abstraction Refinement)

For programs with larger state spaces, CEGAR iteratively abstracts the model,
runs IC3 on the abstraction, and refines when the IC3 counterexample is
spurious (not reproducible in the concrete machine). This extends unbounded
reasoning to programs that are too large for direct IC3.

### Portfolio

Multiple solver configurations run in parallel. The first conclusive result
wins. The Python layer can apply learned algorithm selection — choosing the
configuration most likely to succeed based on features of the BTOR2 model —
or simply run all configurations and race them.

---

## Questions Python Rotor Can Answer

### Reachability
- Can this code ever reach this state (assertion, error path, crash site)?
- Is this branch dead code?
- Can this null pointer dereference actually happen?

### Input Synthesis
- What input triggers this buffer overflow?
- Is there an input that causes this function to return a negative value?
- What sequence of calls puts this data structure in an invalid state?

### Property Verification
- Does this sort function always produce sorted output?
- Is the stack pointer always aligned when this function is called?
- Does this counter ever overflow?
- Is the mutex always held when this shared variable is accessed?

### Equivalence
- Does this refactored version have identical behavior to the original?
- Did this security patch change any behavior beyond the intended fix?
- Are these two implementations of the same algorithm semantically equivalent?

### Causality
- Which input bytes are responsible for this crash?
- What is the minimal input change that avoids this error?
- Which function call corrupts this state?

### Synthesis
- What constant makes this bounds check correct?
- What instruction sequence satisfies this behavioral specification?
- What initial value makes this test pass?

For **reachability, property verification, and equivalence**, IC3 mode can
give unbounded answers — not "safe within k steps" but "safe for all possible
executions." For **input synthesis and causality**, BMC gives concrete
witnesses. For **synthesis**, Rotor's code synthesis mode (uninitialized
code segment) finds concrete instruction sequences satisfying a BTOR2
specification.

---

## Source Connection via DWARF

Every answer Python Rotor produces is lifted back to the source via DWARF
debug information embedded in the ELF binary. The mapping covers:

- **PC → source location**: file name, line number, column (`.debug_line`)
- **PC → live variables**: which variables are in scope and where their
  values live — register, stack offset, or computed address (`.debug_loc`,
  `.debug_info`)
- **PC → function**: name, return type, parameter types (`.debug_info`)
- **Function → address range**: code bounds for focused instance creation,
  including non-contiguous ranges after optimization (`.debug_ranges`)

A counterexample trace from the solver — a sequence of PC values and register
states — becomes a source-level trace with variable names and concrete values
at each step. This trace can be rendered as a human-readable explanation,
exported as SARIF for IDE integration, or converted to a GDB replay script.

---

## Intended Use

Python Rotor is designed as a **backend reasoning oracle** for:

- **LLM coding assistants**: The LLM proposes a property or asks a question
  about code behavior; Rotor answers with a proof or a concrete
  counterexample. The LLM's statistical pattern-matching and Rotor's
  bit-precise exhaustive reasoning are complementary — each contributes what
  the other cannot.

- **CI/CD verification pipelines**: Automated property checking on each
  commit, with unbounded proofs for critical invariants and bounded checking
  for regression detection.

- **Security analysis**: Systematic search for exploitable inputs, with
  machine-verified certificates that entire vulnerability classes do not exist.

- **Compiler and toolchain validation**: Semantic equivalence checking between
  different compiler versions, optimization levels, or code generators,
  verified at the binary level.

---

## Status

This repository is in early planning and design. Contributions, discussion,
and feedback are welcome.

The C Rotor implementation in
[selfie](https://github.com/cksystemsgroup/selfie) is the stable reference
and is actively maintained.

---

## References

- [Selfie Project](https://selfie.cs.uni-salzburg.at) —
  the educational systems project from which C Rotor originates
- [BTOR2 format](https://github.com/btor2tools/btor2tools) —
  the word-level model checking format used as the internal representation
- [Bitwuzla](https://bitwuzla.github.io) —
  SMT solver with strong bitvector support, primary BMC backend
- [IC3/PDR](https://dl.acm.org/doi/10.1145/1987389.1987415) —
  Bradley 2011, the algorithm enabling unbounded safety proofs
- [Hardware Model Checking Competition](https://hwmcc.github.io) —
  annual competition driving advances in BTOR2 model checker performance
- [DWARF standard](https://dwarfstd.org) —
  the debug information format used for source-level result presentation