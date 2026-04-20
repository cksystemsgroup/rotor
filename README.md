# Rotor

**Rotor is a BTOR2 compiler for RISC-V questions.**

Given a RISC-V ELF binary and a question about its behavior — "can this
buffer overflow happen?", "does this refactor preserve semantics?", "what
input triggers this crash?" — rotor compiles the question into a **BTOR2
proof obligation** that encodes it bit-precisely, hands that obligation to
an external model checker, and lifts the answer back to the source via
DWARF.

Everything else in rotor is optimization. The proof obligations are why
you can trust the answer; internal IRs are how rotor produces smaller
obligations faster.

This repository is the **Python Rotor** orchestration layer. The
BTOR2-level RISC-V semantics originate in [C Rotor](https://github.com/cksystemsgroup/selfie)
(`tools/rotor.c`) from the Selfie Project at the Computational Systems
Group, University of Salzburg.

---

## Quick start

Install from the repository root:

```bash
pip install -e .
```

Everything below uses `tests/fixtures/add2.elf` — a two-function test
binary (`add2` and `sign`). Swap in your own RV64I ELF compiled with
`-g` and `-march=rv64im` (no compressed instructions yet) to try it on
real code.

```console
$ rotor info tests/fixtures/add2.elf --functions
path      : tests/fixtures/add2.elf
is_64bit  : True
entry     : 0x100b0
code      : 0x100b0..0x100d0 (32 bytes)
functions :
  0x000100b0 +8    add2
  0x000100b8 +24   sign

$ rotor disasm tests/fixtures/add2.elf --function add2
  0x000100b0: addw a0, a0, a1         ; add2.c:11
  0x000100b4: ret                     ; add2.c:11

$ rotor reach tests/fixtures/add2.elf --function add2 --target 0x100b4 --bound 2
verdict  : reachable
bound    : 2
step     : 1
backend  : z3-bmc
```

Exit codes follow the convention `0` = safe / proved / equivalent,
`1` = reached / found / differs, `2` = unknown / error. On a reachable
verdict, a source-annotated counterexample trace is written to stderr
(or to a file via `--trace PATH`).

The CLI subcommands map 1-to-1 to the Python API:

```python
from rotor import RotorAPI

with RotorAPI("tests/fixtures/add2.elf", default_bound=2) as api:
    add2 = api.binary.function("add2")
    result = api.can_reach(function="add2", target_pc=add2.start + 4)
    if result.verdict == "reachable":
        print(result.trace.to_markdown())
```

Currently shipped verbs (L0 today): `info`, `disasm`, `reach` /
`can_reach`. Additional verbs (`verify`, `find-input`, `equivalent`)
land in later milestones under the same external contract — see
[`PLAN.md`](PLAN.md).

---

## Debugging and benchmarking with BTOR2 text

The same BTOR2 that rotor emits to solvers can also be fed back in. Two
subcommands expose this seam:

```console
$ rotor btor2-roundtrip model.btor2      # parse then re-emit (stdout)
$ rotor solve-btor2 model.btor2 \        # race Z3BMC across several bounds
    --bound 10 --bound 40 --bound 160
verdict  : reachable
bound    : 160
step     : 100
elapsed  : 42.1ms
backend  : z3-bmc
```

`btor2-roundtrip` normalizes a model into rotor's canonical form (dense
ids, `constd`-only constants, no trailing symbols) and surfaces parser
diagnostics on stderr — useful for delta-debugging failing emissions
and for ingesting HWMCC-style benchmarks whose surface syntax differs
from what rotor emits.

`solve-btor2` parses the file, hands the resulting `Model` to a
`Portfolio` of `Z3BMC` entries (one per `--bound`), and races them. The
race short-circuits on the first `reachable` and otherwise returns the
deepest `unreachable`.

This is a debugging and benchmarking seam, not a second input language:
rotor still has no source-lift story for arbitrary BTOR2, because DWARF
only exists for the binary. See PLAN.md's Non-goals section.

---

## The architectural principle

Rotor is organized around a single invariant:

> **Every layer of rotor emits the same BTOR2 proof obligations at its
> external boundary.** The internal representation is a performance story,
> never a semantic one.

The simplest rotor — a BTOR2 compiler plus a solver dispatcher plus a
DWARF lift — is the **reference implementation**. Every optimization
layered underneath is validated by producing the same answer on the same
question. New IRs do not change what rotor means; they change the size
and shape of the obligation it hands to the solver.

This is the compiler-design pattern applied to reasoning: the IR is
internal, the emitted artifact is the contract, and the contract is
checkable against a trivially-correct baseline.

---

## Layered architecture

```
                     ┌─────────────────────────┐
question  ─────────► │      RotorEngine        │ ─── selects layer, races portfolio
(reach/verify/...)   └────────────┬────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┬──────────────────┐
            ▼                     ▼                     ▼                  ▼
     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    ┌──────────────┐
     │ L0  identity │     │ L1  BV DAG   │     │ L2  SSA-BV   │    │ L3  BVDD     │
     │              │     │              │     │              │    │              │
     │ C rotor →    │     │ hash-cons +  │     │ φ-nodes,     │    │ set repr.,   │
     │ raw BTOR2    │     │ const fold + │     │ dominators,  │    │ image /      │
     │              │     │ CSE + DCE    │     │ slicing      │    │ project /    │
     │              │     │              │     │              │    │ subsumption  │
     └──────┬───────┘     └──────┬───────┘     └──────┬───────┘    └──────┬───────┘
            │                    │                    │                   │
            └────────────────────┴────────────────────┴───────────────────┘
                                          │
                                          ▼
                             ╔══════════════════════╗
                             ║   BTOR2 emitter      ║  ◄── the universal seam
                             ║   (one interface,    ║
                             ║    many impls)       ║
                             ╚══════════┬═══════════╝
                                        ▼
                        Bitwuzla / rIC3 / AVR / ABC / BtorMC / Z3
                                        │
                                        ▼
                           model (SAT) / invariant (UNSAT) / unknown
                                        │
                                        ▼
                              DWARF lift → source answer
```

### L0 — identity

The reference layer. Rotor generates a full BTOR2 machine model by walking
the binary (via C Rotor, as a subprocess or linked library) and appending
question-specific init/bad states. The model is linear in the size of the
binary. External solvers consume the BTOR2 and return SAT with a witness,
UNSAT with an optional inductive invariant, or UNKNOWN.

L0 has no symbolic executor, no partial evaluator, no IR beyond the
BTOR2 itself. It is deliberately the simplest thing that answers questions
correctly end-to-end, and it is the correctness oracle for every other
layer.

### L1 — BV expression DAG

A hash-consed DAG of bitvector expressions built during symbolic
execution of the binary. Constants fold, dead branches drop, common
subexpressions share. When a question arrives, the DAG is linearized to
BTOR2 — smaller than L0's because folding has already happened. Operations
rotor needs without a solver (simplify, substitute, emit) are all
structural. This is the classic symbolic-executor IR (angr/Claripy, KLEE).

### L2 — SSA-BV

The DAG, lifted to SSA over bitvector values: one definition per
instruction, φ-nodes at CFG joins, dominator information, def-use chains.
Unlocks goal-directed BTOR2 slicing (only emit obligations reachable from
the property), higher-quality IC3 transition relations, and cleaner DWARF
lifting because SSA defs align naturally with source variables.

### L3 — BVDD

Bit-Vector Decision Diagrams as the internal representation for
**reachable-state sets**. When rotor maintains an IC3 frame, a CEGAR
abstraction, or an equivalence product across two cores, BVDDs answer
set-theoretic questions — is-empty, subsumes, image, project — structurally,
without invoking the solver. Starts as a pure-Python prototype; graduates
to a Rust backend via PyO3 when Python's performance ceiling binds.

BVDDs are developed independently in [bitr](https://github.com/cksystemsgroup/bitr).
Rotor uses the data structure; the external solver continues to do the
SAT/SMT search on residual obligations.

---

## L0-equivalence: rotor's correctness story

Every IR layer earns its place by passing the **L0-equivalence harness**:
for a corpus of questions, the IR's BTOR2 emission must yield the same
verdict as L0's BTOR2 emission. This is the regression oracle and
shipping gate. It runs in CI on every change that touches an IR.

The harness is what lets rotor add aggressive optimization without
compromising trust: if L1's constant folding is unsound, or L3's frame
algebra subtly loses a case, L0 catches it on a concrete example.

---

## Reasoning modes

Reasoning modes are properties of the solver stack, independent of which
IR layer produced the BTOR2. Any layer can target any mode.

### Bounded Model Checking (BMC)

Unroll the BTOR2 machine model `k` times; ask an SMT solver whether any
bad state is reachable. Finds concrete bugs fast; cannot prove safety
beyond `k` steps. **Answer:** BUG (with input and source trace) or
SAFE-UP-TO-k.

### IC3 / Property-Directed Reachability

Search for an inductive invariant separating initial from bad states,
without bounding steps. **Answer:** BUG (concrete counterexample) or
PROVED (invariant verified for all executions of any length). For
finite-state components — bounded loops, fixed-size buffers, protocol
state machines — IC3 typically terminates quickly.

### CEGAR (Counterexample-Guided Abstraction Refinement)

Abstract the model, run IC3 on the abstraction, refine on spurious
counterexamples. Extends unbounded reasoning to programs too large for
direct IC3.

### Portfolio

Race multiple solver configurations in parallel. First conclusive result
wins. Can be combined with learned algorithm selection based on features
of the BTOR2 model.

---

## Questions rotor can answer

### Reachability
- Can this code ever reach this state (assertion, error path, crash site)?
- Is this branch dead code?
- Can this null pointer dereference actually happen?

### Input synthesis
- What input triggers this buffer overflow?
- Is there an input that causes this function to return a negative value?

### Property verification
- Does this sort function always produce sorted output?
- Is the stack pointer always aligned when this function is called?
- Does this counter ever overflow?

### Equivalence
- Does this refactored version have identical behavior to the original?
- Did this security patch change any behavior beyond the intended fix?

### Causality
- Which input bytes are responsible for this crash?
- What is the minimal input change that avoids this error?

### Synthesis
- What constant makes this bounds check correct?
- What instruction sequence satisfies this behavioral specification?

For reachability, property verification, and equivalence, IC3 mode gives
**unbounded** answers — not "safe within k steps" but "safe for all
possible executions." For input synthesis and causality, BMC gives
concrete witnesses.

---

## Source connection via DWARF

Every answer rotor produces is lifted back to source via DWARF debug
information embedded in the ELF binary:

- **PC → source location** (file, line, column) via `.debug_line`
- **PC → live variables** with their register / stack / memory locations
  via `.debug_loc` and `.debug_info`
- **PC → function** (name, return type, parameters) via `.debug_info`
- **Function → address range(s)** via `.debug_ranges`, including
  non-contiguous ranges after optimization

A solver witness — a sequence of PC values and machine states — becomes a
source-level trace with variable names and concrete values at each step.
The trace can be rendered as human-readable markdown, exported as SARIF
for IDE integration, or converted to a GDB replay script.

---

## Relationship to C Rotor and bitr

- **[C Rotor](https://github.com/cksystemsgroup/selfie)** (`tools/rotor.c`)
  is the reference BTOR2 encoder for RV32I / RV64I plus M-extensions and
  compressed instructions. It produces a full machine model in linear
  time. Python Rotor's **L0 emitter** delegates to C Rotor; L1–L3 emit
  BTOR2 that is semantically equivalent but constant-folded, sliced, or
  projected.

- **[bitr](https://github.com/cksystemsgroup/bitr)** develops BVDDs —
  Bit-Vector Decision Diagrams with 256-bit bitmask edge labels that let
  set-theoretic operations on bitvector state run structurally. Rotor's
  **L3** consumes bitr's BVDD data structure (via PyO3 once available in
  Rust) while continuing to use external solvers for residual
  obligations.

Both projects share BTOR2 as their external contract, which is what makes
them composable in either direction.

---

## Intended use

Python Rotor is a **backend reasoning oracle** for:

- **LLM coding assistants.** The LLM proposes a property or asks a
  question about code behavior; rotor answers with a proof or a concrete
  counterexample. Statistical pattern-matching and bit-precise exhaustive
  reasoning are complementary.
- **CI/CD verification.** Automated property checking per commit, with
  unbounded proofs for critical invariants and bounded checking for
  regression detection.
- **Security analysis.** Systematic search for exploitable inputs plus
  machine-verified certificates that entire vulnerability classes cannot
  occur.
- **Compiler and toolchain validation.** Semantic equivalence checking
  between compiler versions, optimization levels, or code generators —
  verified at the binary level.

---

## Status

**This repository is a fresh start.** The design articulated above
supersedes earlier scaffolding. See [`PLAN.md`](PLAN.md) for the phased
implementation roadmap, starting from an empty working tree and ending at
a four-layer architecture with an L0-equivalence correctness harness.

L0 is the shipping gate. L1, L2, L3 are optional and additive.

---

## References

- [Selfie Project](https://selfie.cs.uni-salzburg.at) — the educational
  systems project from which C Rotor originates.
- [BTOR2 format](https://github.com/Boolector/btor2tools) — the word-level
  model-checking format used as rotor's external contract.
- [Bitwuzla](https://bitwuzla.github.io) — SMT solver with strong
  bitvector support.
- [rIC3](https://github.com/gipsyh/rIC3), [AVR](https://github.com/aman-goel/avr),
  [ABC](https://github.com/berkeley-abc/abc) — IC3/PDR engines consuming
  BTOR2.
- [IC3/PDR](https://dl.acm.org/doi/10.1145/1987389.1987415) — Bradley 2011,
  the algorithm enabling unbounded safety proofs.
- [Hardware Model Checking Competition](https://hwmcc.github.io) — annual
  driver of BTOR2 model-checker performance.
- [DWARF standard](https://dwarfstd.org) — debug-information format used
  for source-level answer presentation.
- [bitr](https://github.com/cksystemsgroup/bitr) — BVDD data structure
  and BTOR2 model checker using BVDDs.
