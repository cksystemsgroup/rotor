# Python Rotor: Implementation Plan

This plan turns the architecture in [`README.md`](README.md) into an
executable roadmap. It is organized by **layer**, not by feature. L0 is
the shipping gate and the correctness oracle; L1, L2, L3 are additive
optimizations under the same external contract.

---

## Guiding principle

> **Every layer emits the same BTOR2 proof obligations at its external
> boundary.** The IR is internal; the obligation is the contract.

Consequences this plan takes seriously:

1. L0 must be buildable and correct *before* any IR is written.
2. Every IR ships only after passing the **L0-equivalence harness** on a
   declared corpus.
3. The `BTOR2Emitter` interface is the single seam that every layer
   crosses. It is the first thing we define and the last thing we change.
4. Reasoning modes (BMC / IC3 / CEGAR / portfolio) belong to the solver
   stack, not to IRs. Any layer can target any mode.

---

## Target project structure

```
rotor/
├── README.md
├── PLAN.md
├── LICENSE
├── pyproject.toml
├── rotor/
│   ├── __init__.py
│   ├── binary.py                ← L0 Phase 1: ELF + DWARF parsing
│   ├── dwarf.py                 ← L0 Phase 1: DWARF evaluation
│   ├── btor2/                   ← L0 Phase 2: BTOR2 generation (native Python)
│   │   ├── __init__.py
│   │   ├── nodes.py
│   │   ├── builder.py
│   │   ├── printer.py
│   │   ├── parser.py
│   │   └── riscv/               ← L0 Phase 8: RV64I decoder + ISA lowering
│   │       ├── __init__.py
│   │       ├── decoder.py
│   │       └── isa.py
│   ├── riscv/                   ← non-BTOR2 RISC-V utilities (disasm, etc.)
│   │   ├── __init__.py
│   │   └── disasm.py
│   ├── solvers/                 ← L0 Phase 3: solver backends
│   │   ├── base.py
│   │   ├── z3bv.py              ← M1–M7: Z3 BMC backend (shipping)
│   │   ├── portfolio.py
│   │   ├── bitwuzla.py          ← deferred: additional BMC backend
│   │   ├── btormc.py            ← deferred: native BTOR2 model checker
│   │   ├── kind.py              ← deferred: shared k-induction driver
│   │   └── ic3.py               ← deferred: IC3/PDR subprocess bridges
│   ├── ir/                      ← L1–L3: internal representations
│   │   ├── __init__.py
│   │   ├── protocols.py         ← capability protocols
│   │   ├── emitter.py           ← BTOR2Emitter Protocol + IdentityEmitter
│   │   ├── dag.py               ← L1: BV expression DAG
│   │   ├── ssa.py               ← L2: SSA-BV
│   │   └── bvdd.py              ← L3: BVDD (pure Python; Rust via PyO3 later)
│   ├── instance.py              ← L0: RotorInstance (one obligation)
│   ├── engine.py                ← L0: RotorEngine (orchestration + portfolio)
│   ├── api.py                   ← L0: high-level question API
│   ├── cli.py                   ← L0: command-line interface
│   ├── trace.py                 ← L0: source-level trace rendering
│   └── witness.py               ← L0: solver model → machine trace
├── tests/
│   ├── fixtures/                ← small RV ELF binaries
│   ├── unit/                    ← per-module tests
│   ├── integration/             ← end-to-end L0 tests
│   └── equivalence/             ← L0-equivalence harness (runs per IR)
└── examples/
    ├── buffer_overflow.py
    ├── input_synthesis.py
    ├── equivalence_check.py
    └── ic3_proof.py
```

---

## Layer map

| Layer | Role | Ships when | Correctness gate |
|---|---|---|---|
| **L0** | BTOR2 compiler + solver dispatcher + DWARF lift | All API verbs answer end-to-end on the corpus | Integration tests on fixture binaries |
| **L1** | BV expression DAG: const fold, CSE, DCE | Passes L0-equivalence on full corpus | L0-equivalence harness |
| **L2** | SSA-BV: φ, dominators, slicing | Passes L0-equivalence + slicing metrics | L0-equivalence harness |
| **L3** | BVDD: set-theoretic ops for IC3 frames / equivalence products | Passes L0-equivalence + frame algebra tests | L0-equivalence harness |

Each layer is independently benchmarked against L0 on a fixed corpus.
"Fast but wrong" never ships.

---

## L0: the BTOR2 compiler (phases 1–9)

L0 is the reference implementation. It can answer every API question
without any IR. L1–L3 optimize this path; they never replace it.

### Phase 1 — ELF and DWARF parsing

**Deliverable:** `rotor/binary.py`, `rotor/dwarf.py`

- `RISCVBinary(path)` loads an ELF via `pyelftools`.
- Extracts: segments, symbols, entry point, function bounds (including
  non-contiguous ranges), instruction bytes.
- Resolves `PC → (file, line, col)`, `PC → function`, `PC → live
  variables` with location expressions (`DW_OP_*`).
- Reverse mapping `function-name → PC range` for focused instance
  creation.

**Tests:** `tests/unit/test_binary.py`, `tests/unit/test_dwarf.py` on
fixture binaries with handcrafted DWARF.

### Phase 2 — BTOR2 generation

**Deliverable:** `rotor/btor2/`

- `nodes.py`: typed BTOR2 node DAG (sorts, states, inputs, nexts, bads,
  constraints).
- `builder.py`: assembles a machine model from a `RISCVBinary` plus a
  question spec. First pass delegates to C Rotor as a subprocess and
  parses its output; later, optional in-process construction.
- `printer.py` / `parser.py`: bidirectional text BTOR2.

**Tests:** structural tests on small machines; round-trip through
printer/parser.

### Phase 3 — RotorInstance and solver interface

**Deliverable:** `rotor/instance.py`, `rotor/solvers/base.py`,
`rotor/solvers/z3bv.py`.

- `RotorInstance` holds one BTOR2 model plus added init/bad/constraint
  nodes; provides `check(mode, bound, timeout)`.
- `SolverBackend` Protocol: `check(btor2_bytes, mode, bound, timeout)
  -> SolverResult(verdict, model?, invariant?)`.
- M1 scope ships a single concrete backend: `Z3BMC` (Z3 as BMC engine
  over the BTOR2 model). Additional backends — Bitwuzla (BMC via
  SMT-LIB), BtorMC (BMC direct), a shared k-induction driver, and
  subprocess bridges for external engines — are deferred; each lands
  under the existing `SolverBackend` Protocol without further
  architectural change.

**Tests:** unit + integration against fixture BTOR2.

### Phase 4 — RotorEngine and portfolio

**Deliverable:** `rotor/engine.py`, `rotor/solvers/portfolio.py`.

- `RotorEngine(binary, config)` spawns `RotorInstance`s per question.
- `portfolio.py` races multiple solver configs in parallel; first
  conclusive verdict wins; others cancel.
- Strategy hooks: per-question bound selection, solver preference,
  timeout escalation.

**Tests:** race semantics, cancellation, deterministic replay.

### Phase 5 — Source-level trace

**Deliverable:** `rotor/trace.py`, `rotor/witness.py`.

- `witness.py`: parse solver model / BTOR2 witness into
  `(step, pc, regs, memory_ops, inputs)` sequence.
- `trace.py`: lift each step through DWARF into source location + live
  variable values; render as markdown, JSON, SARIF, or GDB script.

**Tests:** round-trip witness → source on fixture programs.

### Phase 6 — IC3 and CEGAR wiring

**Deliverable:** `rotor/solvers/z3spacer.py`, `rotor/cegar.py`, plus
CLI / API surfaces (`--unbounded`, `--cegar`, `api.can_reach(unbounded=
True)`, `api.cegar_reach`).

- **IC3 via Z3 Spacer (in-process).** `Z3Spacer` translates a BTOR2
  Model into a Constrained Horn Clause system over a single `Inv`
  relation and queries `z3.Fixedpoint` with engine=spacer. Returns
  `proved` with an invariant certificate, `reachable` (no step —
  Spacer's Python API doesn't expose witness traces), or `unknown`.
  Bound is accepted for Protocol compatibility but ignored.
- **Subprocess bridges to rIC3 / AVR / ABC: deferred.** Added only
  when an external engine beats Spacer on a workload rotor ships
  against. Same `SolverBackend` Protocol; no architectural change.
- **CEGAR loop.** `rotor/cegar.py` wraps Spacer in a refinement
  loop: start with every register havoc'd (`build_reach(havoc_regs=
  set(range(1,32)))`), run Spacer, validate any `reachable` verdict
  against rotor's concrete witness simulator, and unhavoc registers
  the real path read on spurious CEX. The refinement strategy is
  register-localization — a simple, terminating heuristic; smarter
  predicate-abstraction variants are future work (see Open Questions).
- **Entry-state assumption** in `build_reach`: constraint `(ra & ~1)
  ∉ [fn.start, fn.end)` so free `ra` cannot collapse `can_reach`
  via an adversarial `ret`. Skipped when `ra` is havoc'd. First
  facet of a future `EntryAssumptions` object that will grow to
  model a real bootloader / call-site on non-leaf functions.
- **Memory-model skip** in `build_reach`: if no instruction in the
  function is a load or store, the SMT array state + ELF write-chain
  init are omitted. Cuts the Model by ~7× on pure-arithmetic
  functions and is what makes Spacer viable on them — array theory
  in PDR is otherwise a scaling cliff.

**Tests:** `tests/integration/test_ic3_counter.py` exercises the
BMC/IC3 contrast on `counter.elf::tiny_mask` — BMC answers
`unreachable` at every bound, Spacer answers `proved` with an
invariant, and the portfolio prefers the globally-conclusive result.
`tests/integration/test_cegar.py` covers CEGAR's `proved` (2
iterations, 27/31 regs still havoc'd), `reachable` (concrete replay
confirms a real CEX), and `unknown` (iteration-budget exhaustion)
paths. `tests/integration/test_api_unbounded.py` covers the API
kwargs and the regression that default `can_reach` stays bounded BMC.
`counter.elf::bounded_counter` — the loop-carried dead-branch case —
exceeds Spacer under the current L0 encoding; the fix is M8's
goal-directed slicing, which will collapse the PDR state space to
just the registers in the property's fanin.

### Phase 7 — High-level API and CLI

**Deliverable:** `rotor/api.py`, `rotor/cli.py`.

- `RotorAPI(path, default_solver, default_bound)` with:
  - `can_reach(function, condition, bound) -> ReachResult`
  - `find_input(function, output_condition, bound) -> SynthResult`
  - `verify(function, invariant, bound, unbounded) -> VerifyResult`
  - `are_equivalent(other_binary, function, bound, unbounded) -> EqResult`
- CLI subcommands map 1:1 to API verbs. Exit codes: `0` safe/proved,
  `1` found/reachable, `2` unknown/error.

**Tests:** CLI integration; API contract tests.

### L0 shipping gate

L0 ships when:

- Every fixture binary in `tests/fixtures/` answers every API verb
  end-to-end.
- `rotor info | disasm | reach | find-input | verify | equivalent` all
  work from the CLI.
- Integration tests pass in CI.
- README's Quick Start walk-through runs as written.

### Phase 8 — Full RV64I

**Deliverable:** expanded `rotor/btor2/riscv/decoder.py`,
`rotor/btor2/riscv/isa.py`, mirrored in `rotor/witness.py`.

M1's L0 supports six instructions (addi, addw, sub, sltu, blt, jalr) —
enough to model leaf functions GCC compiles from `add2.c`. Phase 8
brings L0 to full RV64I-base coverage so any leaf function compiled
with `-march=rv64i` decodes and runs:

- I-type arith complete: slti, sltiu, xori, ori, andi, slli, srli,
  srai, addiw, slliw, srliw, sraiw.
- R-type arith complete: add, sll, slt, xor, srl, sra, or, and,
  subw, sllw, srlw, sraw.
- Branches complete: beq, bne, bge, bltu, bgeu (blt already done).
- U-type: lui, auipc.
- J-type: jal.
- Misc-mem: fence (modeled as no-op — safe approximation for
  single-core reasoning).

System instructions (ecall, ebreak) are modeled in Phase F (syscalls),
not here. Loads/stores are Phase 9 (memory).

**Architectural cleanup:** the decoder and ISA lowering refactor from
flat if/elif into opcode-dispatched tables. With ~30 instructions a
linear chain stops being readable.

**Tests:** decoder coverage of every new opcode; witness simulator
mirrors the BTOR2 semantics for every instruction; new fixture
`branches.elf` exercises all six branch types; L0-equivalence corpus
gains entries that take the new control-flow paths.

### Phase 9 — Memory model

**Deliverable:** array-sort BTOR2 + memory-aware builder + interpreter.

Adds the SMT array of bytes that lets rotor model loads, stores, and
non-leaf functions (stack save/restore, .rodata access, anything
touching memory).

- BTOR2 nodes: array sort `bv64 -> bv8`, ops `read` / `write` /
  `init_array` (constant array), serialized by `printer.py`.
- Z3 backend: `z3.Array`, `z3.Select`, `z3.Store`.
- Memory state: byte-addressed, single state variable
  `mem: bv64 -> bv8`. Initial value: ELF `.text`/`.rodata`/`.data`/
  `.bss` segments stored at their virtual addresses; everything else
  free.
- Loads (lb / lh / lw / ld + lbu / lhu / lwu) compose multiple byte
  reads with concat plus sign- or zero-extension.
- Stores (sb / sh / sw / sd) decompose the value with slice plus
  multiple writes.
- Witness simulator gains a concrete `dict[int, int]` memory mirror.

**Tests:** non-leaf function fixture (with stack save/restore around
ra), .rodata-reading fixture, alignment-relevant load/store
combinations. Equivalence corpus extended.

---

## The BTOR2 emitter seam

Once L0 ships, introduce `rotor/ir/emitter.py` **before** any IR:

```python
# rotor/ir/emitter.py
from typing import Protocol
from rotor.btor2 import ModelSpec

class BTOR2Emitter(Protocol):
    """Emits a BTOR2 proof obligation for one question on one binary."""
    def emit(self, spec: ModelSpec) -> bytes: ...

class IdentityEmitter:
    """L0: delegate to C Rotor; no IR transformation."""
    def emit(self, spec: ModelSpec) -> bytes:
        return _c_rotor_emit(spec)
```

`RotorEngine` accepts an `emitter_factory` parameter (a callable
`RISCVBinary -> BTOR2Emitter`); the default is `IdentityEmitter`. The
engine constructs the emitter once per binary and reuses it across
`RotorInstance`s it spawns. Every IR implements `BTOR2Emitter`. This is
the single change that prepares the codebase for L1–L3 without adding
behavior.

---

## IR capability protocols

IRs differ in what they answer **without** invoking a solver. Declare
this as opt-in capabilities in `rotor/ir/protocols.py`, mirroring
`collections.abc`:

```python
class Builder(Protocol):
    def const(self, value: int, width: int) -> Term: ...
    def fresh(self, name: str, width: int) -> Term: ...
    def bvadd(self, a: Term, b: Term) -> Term: ...
    def bvand(self, a: Term, b: Term) -> Term: ...
    def bvslt(self, a: Term, b: Term) -> Term: ...
    def extract(self, t: Term, hi: int, lo: int) -> Term: ...
    def ite(self, c: Term, t: Term, e: Term) -> Term: ...
    def select(self, mem: Term, addr: Term) -> Term: ...
    def store(self, mem: Term, addr: Term, val: Term) -> Term: ...

class State(Protocol):
    regs: Mapping[str, Term]
    mem:  Term
    pc:   Term
    guard: Term
    def fork(self, cond: Term) -> tuple[State, State]: ...
    def merge(self, other: State) -> State: ...

class SolverBridge(Protocol):
    def emit_btor2(self, spec: ModelSpec) -> bytes: ...
    def emit_smtlib(self, spec: ModelSpec) -> str: ...
    def lift_model(self, raw: SolverModel) -> dict[Term, int]: ...

# opt-in capabilities
class Simplify(Protocol):
    def simplify(self, t: Term) -> Term: ...

class Canonical(Protocol):
    def syntactic_eq(self, a: Term, b: Term) -> bool: ...
    def term_hash(self, t: Term) -> int: ...

class SetOps(Protocol):              # BVDD territory
    def is_empty(self, s: State) -> bool: ...
    def subsumes(self, a: State, b: State) -> bool | None: ...
    def image(self, s: State, trans: Transition) -> State: ...
    def project(self, s: State, vars: Sequence[str]) -> State: ...

class SSA(Protocol):                 # SSA-BV exposes def-use
    def def_of(self, t: Term) -> Definition: ...
    def uses_of(self, t: Term) -> Iterable[Definition]: ...
```

Rotor code uses the core (`Builder`/`State`/`SolverBridge`) everywhere
and checks `isinstance(backend, SetOps)` for fast paths. Consumers must
not assume capabilities they did not require.

---

## L0-equivalence harness

**Deliverable:** `tests/equivalence/`.

For a declared corpus of `(binary, question, expected_verdict)` triples,
run each IR under test and assert its verdict matches L0's. Spot-checks
also compare:

- BTOR2 size (expect monotonic non-increase across IRs).
- Solve time (expect monotonic non-increase on hard cases).
- Witness round-trip through DWARF (expect identical source trace modulo
  internal var naming).

The harness is a pytest fixture parameterized by
`emitter ∈ {IdentityEmitter, DagEmitter, SsaEmitter, BvddEmitter}`.
New IRs extend the matrix; they do not get to skip it.

**Shipping rule:** an IR may enter the portfolio only after it passes
the harness on the full corpus. CI enforces.

---

## L1: BV expression DAG

**Deliverable:** `rotor/ir/dag.py`.

- Hash-consed BV expression nodes (pythonic `__slots__`, `__hash__`).
- Structural simplification: constant folding, `x+0 → x`, `x & -1 → x`,
  ITE with constant guard, etc. Limited, auditable rewrites.
- Symbolic executor in Python stepping RISC-V instructions over DAG
  values; memory as SMT array term initially.
- Emitter: linearize reachable DAG rooted at goal into BTOR2; declares
  states/inputs as needed.

**Exit criterion:** L0-equivalence on full corpus; measurable BTOR2-size
reduction on at least 50% of corpus cases.

---

## L2: SSA-BV

**Deliverable:** `rotor/ir/ssa.py`.

- SSA IR layered on L1's DAG: per-instruction def, φ at joins, dominator
  tree, def-use chains.
- Goal-directed slicing: walk def-use backward from the property; emit
  only reachable defs into BTOR2.
- DWARF provenance on SSA defs (attach location info at definition,
  propagate through φ).
- IC3-friendly transition relation: declare per-register next-state
  functions explicitly, avoid unnecessary quantification.

**Exit criterion:** L0-equivalence; IC3 solve-time improvement on frame-
heavy benchmarks.

---

## L3: BVDD

**Deliverable:** `rotor/ir/bvdd.py`.

**Phase 3a — pure-Python prototype.**

- Dict-based hash-consed nodes; tuple-keyed apply cache.
- 256-bit edge bitmasks as `int` (Python bigints).
- Core ops: `apply(op, a, b)`, `image(trans, s)`, `project(vars, s)`,
  `is_empty`, `subsumes`.
- `SetOps` capability satisfied.

**Phase 3b — Rust backend via PyO3.**

- Introduced only when the Python prototype hits a measurable ceiling.
- Bulk API (one call ⇒ many internal BVDD ops) to keep FFI cheap.
- Uses the BVDD data structure from
  [bitr](https://github.com/cksystemsgroup/bitr) once available as a
  crate.

**Use cases that exercise L3:**

- IC3 frame representation (per-frame BVDDs; frame-push = BVDD union;
  subsumption via `subsumes`).
- CEGAR abstraction/refinement as BVDD projection/intersection.
- Multi-core equivalence product for `are_equivalent`.

**Exit criterion:** L0-equivalence; measurable speedup on IC3 frame-
heavy and equivalence workloads. Pure-Python prototype must pass the
harness even if slow — correctness first.

---

## Portfolio integration

Each IR, once it passes L0-equivalence, can enter `rotor/solvers/portfolio.py`
as an additional racer. The portfolio config declares
`(emitter, solver, mode, bound, timeout)` tuples; rotor spawns and
races them. No IR is mandatory; L0 + Bitwuzla is always in the race.

---

## Testing strategy

| Layer | Tests |
|---|---|
| L0 | unit (`tests/unit/`), integration (`tests/integration/`), CLI |
| BTOR2 seam | `tests/unit/test_emitter.py` — IdentityEmitter contract |
| IRs (L1–L3) | `tests/equivalence/` — L0-equivalence harness, per IR |
| Portfolio | race / cancel / determinism tests |
| Examples | every `examples/*.py` runs in CI as a smoke test |

CI matrix:

- Python 3.11, 3.12.
- With / without optional solvers (Bitwuzla, rIC3).
- Corpus: `tests/fixtures/*.elf` — small but structurally diverse
  (straight-line, branching, loops, syscalls, memory).

---

## Milestones

| # | Scope | Gate |
|---|---|---|
| **M1** | Phases 1–3: ELF + BTOR2 + one BMC backend answers `can_reach` | Integration test on `add2.elf` passes |
| **M2** | Phases 4–5: portfolio + source trace | Trace markdown lands for BMC counterexamples |
| **M3** | Phases 6–7: IC3/CEGAR + API/CLI parity | README Quick Start runs verbatim |
| **M4** | BTOR2 emitter seam introduced (no behavior change) | `IdentityEmitter` used end-to-end; tests green |
| **M5** | Phase 8: full RV64I in the native L0 builder | branches.elf fixture passes; decoder + witness cover every RV64I instruction |
| **M6** | Phase 9: memory model — SMT array of bytes; loads / stores; ELF segment loader | non-leaf and .rodata-reading fixtures pass |
| **M7** | L1 (BV DAG) ships | L0-equivalence harness passes |
| **M7.5** | BTOR2 parser (5 phases): inverse of printer, round-trip + diagnostics tests, HWMCC benchmark coverage, `btor2-roundtrip` + `solve-btor2` CLI subcommands, README section | Round-trip on rotor-emitted models and HWMCC corpus |
| **M8** | L2 (SSA-BV) ships | L0-equivalence + slicing metrics |
| **M9** | L3 (BVDD) pure-Python prototype | L0-equivalence on set-heavy workloads |
| **M10** | L3 Rust backend (optional) | Measurable improvement over Python prototype |

M1–M3 are the L0 shipping gate. M4 is the seam. M5–M6 expand L0's
RISC-V coverage so the corpus can grow before any IR work begins.
M7–M10 are the IR layers, each independently pursuable once M4 lands.
M7.5 is a parallel track that landed alongside M7: it makes BTOR2 a
first-class ingestion format for debugging and benchmarking without
extending rotor's source-lift contract (see Non-goals).

Future small follow-ups beyond M10 (each ~1 day): RV64M (mul/div/rem),
RVC (compressed instructions), syscall modeling matching selfie, and
two-core composition for the `are_equivalent` verb.

---

## Relationship to C Rotor and bitr

- **C Rotor** is the reference BTOR2 encoder. L0's `IdentityEmitter`
  delegates to it. Higher layers emit BTOR2 that is semantically
  equivalent but typically smaller.
- **bitr** develops the BVDD data structure; L3 consumes it (initially
  as a pure-Python reimplementation, later as a Rust crate via PyO3).
  The BTOR2 seam means bitr's BVDD can plug into rotor without either
  project owning the other's solver loop.

Both projects share BTOR2 as the external contract. That is what lets
the architecture compose in either direction.

---

## Non-goals

- Replacing mature SMT/model-checker engines. Rotor delegates the hard
  search to Bitwuzla, rIC3, AVR, ABC, BtorMC. IRs exist to feed those
  engines better inputs, not to replace them.
- A new input language. Rotor's input is the compiled binary plus a
  question; its output is a source-level answer. BTOR2 is the internal
  currency. A BTOR2 parser ships alongside the emitter (`rotor/btor2/
  parser.py`) and is wired into two CLI verbs — `rotor btor2-roundtrip`
  for delta-debugging the emitter/parser pair, and `rotor solve-btor2`
  for running rotor's portfolio against external BTOR2 benchmarks (e.g.
  HWMCC). This is a debugging and benchmarking seam, not a second input
  language: the binary + question pipeline remains the sole path to a
  source-level answer, and the parser deliberately has no DWARF story.
- Source-level reasoning. Rotor reasons about the binary and maps back
  via DWARF. Source-level tools are a consumer of rotor, not rotor
  itself.

---

## Open questions (for future iteration)

- Where does the memory model live in L1/L2? Full SMT array theory
  always, or paged concrete + symbolic overlay when the symbolic
  executor can prove aliasing bounds?
- Should L3 BVDDs represent joint `(pc, regs)` state or factor by
  register? Factored is smaller in common cases; joint is simpler for
  image computation.
- Does rotor need its own incremental BTOR2 protocol, or is per-query
  regeneration (with IR-side caching) good enough?
- How does CEGAR choose abstraction predicates beyond register-
  localization? Phase 6 ships a simple "unhavoc every register read
  by the concrete replay path" heuristic — terminating and sound but
  coarse. Houdini-style conjunction candidates, learned predicates,
  or interpolant-driven refinement are obvious next steps; each
  slots under the same `cegar_reach` entry point.

These are explicitly out of scope for the initial L0 ship and are
revisited after M4.
