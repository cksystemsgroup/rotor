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