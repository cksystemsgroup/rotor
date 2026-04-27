# Rotor: Implementation Plan

This is the working plan for implementing rotor from scratch as a
**minimal LLM-programmable platform for reasoning about RISC-V code**.
It is self-contained: a fresh Claude Code session should be able to
pick up from here without consulting prior context.

---

## What rotor is

Rotor is a deterministic translator from `(question_spec, RISC-V
binary)` to `(BTOR2 model + structured annotation)`, plus a thin
dispatcher to external solvers and a lifter that maps solver outputs
back through the annotation to source-level information.

**Rotor does no reasoning.** It does not:

- decide what to verify (the LLM constructs the spec)
- choose solvers, bounds, or timeouts (the spec specifies)
- refine abstractions or run CEGAR loops (the LLM iterates by
  re-specifying)
- compose or transfer invariants across questions (the LLM curates)
- interpret verdicts beyond mechanical lifting through the annotation

Rotor's invariant is: for any LLM session, the LLM could in principle
predict the BTOR2 output by reading the **schema**, the **spec**, and
the **binary**, with no other information. Same `(spec, binary)` →
byte-identical BTOR2. No internal state, no learned heuristics, no
adaptivity in compilation.

The reasoning lives in the LLM. Rotor is the substrate; the schema
plus annotation is the contract; the LLM is the agent.

## The split between rotor and the LLM

| Concern | Owner |
|---|---|
| What question to ask | LLM |
| How to express it as a `QuestionSpec` | LLM |
| How to compile a spec to BTOR2 | rotor (schema-deterministic) |
| Which solver to invoke and with what budget | LLM (via spec) |
| How to dispatch the solver and capture its output | rotor |
| How to interpret the verdict | LLM |
| How to lift solver state to source-level facts | rotor (mechanical) |
| Whether/how to refine and re-ask | LLM |
| Whether an invariant from question A applies to question B | LLM |

Anywhere rotor would otherwise make a heuristic choice, that choice
becomes either (a) schema-documented and fixed forever, or (b) a
spec parameter the LLM specifies. There is no third option.

## Architectural commitments

### 1. The schema is the contract

A versioned document spelling out every translation rule from RISC-V
semantics to BTOR2. Every state variable convention, every
instruction lowering, every memory-model decision, every entry-state
assumption rotor adds by default. The schema is authoritative; rotor
implements it; the LLM consults it.

When a translation choice is not obvious (e.g. how to encode signed
division's overflow case), the schema picks one and documents why.
Subsequent rotor changes that affect the schema bump its version.
Cached BTOR2 outputs are tagged with the schema version they were
produced under.

### 2. Hierarchical BTOR2 layers

A monolithic BTOR2 file conflates layers that have very different
stability. Rotor splits its output into the following layers, each a
separate file with its own content hash:

| Layer | Stability | Contents |
|---|---|---|
| **header** | universal | sort declarations (BV1, BV8, ..., BV128, mem array sort) |
| **machine** | per ISA variant + core count | state variable declarations: x1..x31, pc, mem (per core if multi-core) |
| **library** | per ISA variant | per-instruction lowering definitions (universal, named) |
| **dispatch** | per analyzed function set | the PC-keyed ITE chain selecting which library lowering applies at each PC |
| **init** | per question | initial-state clauses (binary memory image, register inits, entry assumptions) |
| **constraint** | per question, accumulates | invariants and assumptions; carries provenance per clause |
| **bad** | per question | the property under investigation |
| **binding** | per question | wires `next` clauses connecting states to dispatch |
| **havoc** | per question (overlay) | optional overrides replacing specific `next` clauses with fresh per-cycle inputs |

Cross-layer references go through symbolic names (`(export name nid)`
and `(import name)` directives in `;@`-comment headers). A linker
resolves names, renumbers nids, and emits a flat BTOR2 file for the
solver. The hierarchy is rotor-internal; what the solver sees is
standard BTOR2.

This factoring is essential for the LLM use case: when an LLM asks
many questions about one binary, layers 1–4 are reused across
questions, and the diff between questions is exactly which init,
constraint, and bad files are linked. This makes incremental analysis
cheap and the *semantic* relationship between questions explicit.

### 3. Annotation sidecar

Alongside each BTOR2 layer, rotor emits a structured annotation file
recording, for every emitted node:

- **role** (state, library-instruction, dispatch-arm, init-clause,
  entry-assumption, learned-invariant, bad-expression, …)
- **source mapping** (for dispatch arms: which RISC-V instruction at
  which PC; for init memory bytes: which ELF segment + virtual
  address; etc.)
- **provenance** (rotor library version, user-specified parameter,
  invariant from prior question N validated against …)
- **DWARF lift** (when available: file/line/column, function, live
  variables)

The annotation is rotor's mechanism for "conveying meaning." The LLM
introspects it via a tool call rather than parsing BTOR2. It is
*emitted alongside compilation*, not reverse-engineered after.

### 4. No reasoning policies in rotor

There is no portfolio. There is no CEGAR loop. There is no liveness
slicer that runs by default. The LLM dispatches one solver at a time
with explicit parameters; if it wants to race, it dispatches in
parallel itself; if it wants CEGAR, it loops over rotor calls with
progressively-refined specs.

Spec parameters do allow the LLM to *request* things rotor knows how
to do (e.g. "havoc registers `{r1, r5, r7}`"). Rotor encodes the
request mechanically. It does not decide *which* registers to havoc.

## Repository layout

```
rotor/
├── README.md                     ← what rotor is, with pointer to PLAN.md
├── PLAN.md                       ← this file
├── SCHEMA.md                     ← the schema document (the contract)
├── LICENSE
├── pyproject.toml
├── rotor/
│   ├── __init__.py
│   ├── binary/                   ← ELF and DWARF parsing (no policy)
│   │   ├── __init__.py
│   │   ├── elf.py
│   │   └── dwarf.py
│   ├── btor2/                    ← in-memory BTOR2 + printer + parser
│   │   ├── __init__.py
│   │   ├── nodes.py
│   │   ├── printer.py
│   │   └── parser.py
│   ├── riscv/                    ← decoder + per-instruction lowering library
│   │   ├── __init__.py
│   │   ├── decoder.py            ← RV64I+M+C decoder (variable-length)
│   │   ├── library.py            ← per-instruction BTOR2 lowering (the schema in code)
│   │   └── disasm.py             ← for human-readable annotation lift
│   ├── spec/                     ← the QuestionSpec language (LLM-facing)
│   │   ├── __init__.py
│   │   ├── observable.py         ← register/memory/pc observables
│   │   ├── assumption.py         ← entry / invariant assumptions
│   │   ├── property.py           ← bad / goal expressions
│   │   ├── analysis.py           ← solver + budget directives
│   │   └── spec.py               ← the top-level QuestionSpec dataclass
│   ├── compile/                  ← schema-deterministic translation
│   │   ├── __init__.py
│   │   ├── header.py             ← universal sorts
│   │   ├── machine.py            ← per-ISA state declarations
│   │   ├── dispatch.py           ← per-function PC-keyed ITE chain
│   │   ├── init.py               ← from binary + spec.assumptions
│   │   ├── constraint.py         ← from spec.assumptions + spec.learned
│   │   ├── bad.py                ← from spec.property
│   │   ├── binding.py            ← next-clause wiring
│   │   ├── havoc.py              ← optional state-override overlay
│   │   ├── linker.py             ← resolve (export/import), renumber, flatten
│   │   └── annotation.py         ← emit sidecar metadata
│   ├── dispatch/                 ← thin solver wrappers (no policy)
│   │   ├── __init__.py
│   │   ├── base.py               ← SolverBackend Protocol
│   │   ├── z3bmc.py
│   │   ├── z3spacer.py
│   │   ├── bitwuzla.py           ← optional, gated on import
│   │   ├── cvc5.py               ← optional, gated on import
│   │   └── pono.py               ← optional, gated on PATH
│   ├── lift/                     ← solver output → source-level facts
│   │   ├── __init__.py
│   │   ├── witness.py            ← reachable: trace replay through library
│   │   ├── invariant.py          ← proved: stringify Spacer's invariant
│   │   └── trace.py              ← markdown / JSON rendering
│   ├── llm/                      ← LLM-facing tool surface
│   │   ├── __init__.py
│   │   ├── tools.py              ← describe / compile / dispatch / lift / introspect
│   │   └── mcp.py                ← optional MCP server adapter
│   └── cli.py                    ← thin CLI mirroring tool surface
├── tests/
│   ├── fixtures/                 ← small RV ELFs + .c + build.sh
│   ├── unit/                     ← per-module
│   ├── golden/                   ← compile-output regression
│   └── integration/              ← end-to-end against fixtures
└── examples/                     ← short Python scripts using rotor as a library
```

What's deliberately absent from current-rotor's layout:

- no `cegar.py` — CEGAR is an LLM pattern, not a rotor module
- no `solvers/portfolio.py` — racing is the LLM's responsibility
- no `ir/dag.py`, `ir/ssa.py`, `ir/liveness.py` — slicing/optimization
  is requested via spec, not auto-applied
- no `engine.py` — there is no orchestration layer; `compile` and
  `dispatch` are independent functions

## The QuestionSpec language

The single data structure rotor consumes from the LLM. Open enough
to express any sensible question; structured enough that ill-formed
specs are rejected before compilation.

```python
@dataclass(frozen=True)
class QuestionSpec:
    binary: BinaryRef                 # path + content hash
    scope: AnalysisScope              # entry function + included callees
    entry: EntryAssumptions           # ra/sp/argument constraints (see below)
    observables: tuple[Observable, ...]   # what to expose for property/witness
    assumptions: tuple[Assumption, ...]   # constraints to add at every cycle
    learned: tuple[LearnedFact, ...]      # invariants from prior questions
    property: Property                # bad / goal expression over observables
    analysis: AnalysisDirective       # solver + bound + timeout
```

### Observables

```python
class Observable:                     # base
    name: str                         # symbolic name for use in property

@dataclass(frozen=True)
class RegisterAt(Observable):
    register: int                     # 0..31
    pc: int                           # observe at this PC

@dataclass(frozen=True)
class MemoryAt(Observable):
    address: Expression               # may reference other observables
    width: int                        # bytes
    pc: int

@dataclass(frozen=True)
class PCAtStep(Observable):
    step: int

@dataclass(frozen=True)
class Executed(Observable):
    pc: int                           # was this PC visited within bound?
```

Observables are Booleans / bitvectors that can appear in `property`
expressions. Rotor lowers each observable into BTOR2 nodes during
compile, with a name that the property layer references.

### Assumptions

```python
@dataclass(frozen=True)
class RegisterInit(Assumption):
    register: int
    op: Comparison                    # eq | neq | slt | ...
    value: int

@dataclass(frozen=True)
class MemoryInit(Assumption):
    address: int
    width: int
    op: Comparison
    value: int

@dataclass(frozen=True)
class CycleInvariant(Assumption):
    expression: Expression            # over state vars
    provenance: Provenance            # see "LearnedFact" below
```

### LearnedFact

A first-class type for invariants the LLM is bringing forward from
prior questions. Distinct from `CycleInvariant` to make the
provenance explicit:

```python
@dataclass(frozen=True)
class LearnedFact:
    expression: Expression
    source_question_hash: str         # hash of the spec that produced it
    source_engine: str                # 'z3-spacer', 'pono-ic3ia', ...
    validated: bool                   # was it re-validated on this model?
```

Rotor adds learned facts as `constraint` clauses, but the annotation
records their provenance so the LLM (and downstream consumers) know
to treat them as imported assumptions rather than as facts established
in this question.

### EntryAssumptions

A first-class object growing over time; only the `excluded_pc_ranges`
field ships in v1. Future fields (sp range, callee-saved equality
between entry and exit, argument constraints) slot in without
touching the spec language.

```python
@dataclass(frozen=True)
class EntryAssumptions:
    excluded_pc_ranges: tuple[tuple[int, int], ...]  # ra outside these
```

### AnalysisDirective

```python
@dataclass(frozen=True)
class AnalysisDirective:
    engine: str                       # 'z3-bmc' | 'z3-spacer' | 'bitwuzla' | …
    bound: int | None                 # for bounded engines
    timeout: float | None
    havoc_registers: frozenset[int]   # explicit; rotor does not pick
    extra_options: Mapping[str, str]  # engine-specific
```

If the LLM wants a portfolio, it submits N specs and dispatches them
itself. If it wants CEGAR, it loops with progressively-shrinking
`havoc_registers`. Rotor sees one spec, runs one solver, returns one
artifact.

## The LLM-facing tool surface

Five tools, mechanical semantics:

### `describe(topic: str) -> SchemaEntry`

Returns the schema entry for a topic: an instruction's lowering, a
state-variable convention, an entry-assumption default, a verdict
type. The LLM consults this on demand instead of being burdened with
the entire schema upfront.

### `compile(spec: QuestionSpec) -> CompiledArtifact`

Deterministic. Returns:

```python
@dataclass(frozen=True)
class CompiledArtifact:
    layers: Mapping[str, BTOR2Layer]      # header, machine, ..., bad, binding
    annotation: AnnotationSidecar         # per-node metadata
    flattened: bytes                      # linker output, ready for solvers
    schema_version: str
    spec_hash: str                        # reproducibility key
```

### `dispatch(artifact: CompiledArtifact, directive: AnalysisDirective) -> RawSolverResult`

Wraps the chosen external solver. Returns:

```python
@dataclass(frozen=True)
class RawSolverResult:
    verdict: str                          # 'reachable' | 'unreachable' | 'proved' | 'unknown'
    elapsed: float
    engine: str
    raw_invariant: str | None             # solver-emitted, not interpreted
    raw_witness: SolverWitness | None     # initial state + step trace
    reason: str | None                    # for unknown
```

### `lift(artifact: CompiledArtifact, raw: RawSolverResult) -> LiftedResult`

Mechanical translation through the annotation: maps witness PCs to
source locations, maps register names to source variables (where
DWARF allows), maps the raw invariant string to its referenced state
variables. Does not interpret meaning beyond mapping.

### `introspect(artifact: CompiledArtifact, query: IntrospectQuery) -> IntrospectResult`

Read-only query against the annotation. "What does node 1247
represent?" "Which BTOR2 nodes correspond to the addw at PC 0x10b8?"
"Which constraints are user-supplied vs. learned?" Pure lookup.

That's the whole LLM surface. Anything richer — proposing follow-up
questions, reasoning about transferability, comparing question
variants — is implemented in the LLM's own logic, composed from
these primitives.

## The schema document (SCHEMA.md)

A peer to PLAN.md, written before any code beyond the most trivial
scaffolding. Contents, in order:

1. **Versioning.** Schema version + how it bumps + how cached
   artifacts are tagged.
2. **Sorts.** Fixed bitvector widths and the array sort.
3. **State variables.** Register and PC conventions, memory model.
4. **ELF loading.** PT_LOAD bytes → memory init; uninitialized
   regions → free.
5. **Instruction lowering.** One subsection per RV64I+M+C mnemonic,
   spelling out the BTOR2 fragment it emits, with explicit attention
   to: signedness, overflow, divide-by-zero, shift-amount masking,
   byte ordering, sign- vs zero-extension. The schema *fixes* every
   choice the RISC-V spec leaves implementation-dependent or that
   could be encoded multiple ways.
6. **Dispatch.** PC-keyed ITE; arm ordering (ascending by PC); how
   PCs outside the analyzed set self-loop.
7. **Entry assumptions.** Default `ra` outside analyzed set; `sp`
   left free unless specified.
8. **Constraint and bad encoding.** Polarity conventions; how multi-
   clause constraints aggregate.
9. **Havoc semantics.** What it means to havoc a register; how
   memory havoc would be encoded if added later.
10. **Verdict semantics.** What `reachable`, `unreachable`, `proved`,
    `unknown` mean from each engine; how `bound` is interpreted by
    each.

Schema-as-document is intentional: it should be readable by an LLM
in one sitting and quoted from in `describe()` outputs.

## Phased implementation

Each phase ships a contract-tested artifact. No phase relies on
implementation details of a later phase. CI gates each phase.

### Phase 0 — Repository scaffolding (½ day)

- `README.md`, `PLAN.md`, `SCHEMA.md` (skeleton — sections present,
  contents stubbed where appropriate), `LICENSE` (MIT), `pyproject.toml`
- Empty package skeleton matching the layout above
- `tests/` directory with `conftest.py` and a single smoke test
  (import every module)
- CI configuration: pytest on Python 3.11, 3.12

**Exit:** `pip install -e .` succeeds; `pytest -q` passes the smoke
test; `pre-commit` (or equivalent) is wired up.

### Phase 1 — BTOR2 in-memory model + text I/O (1 day)

- `rotor/btor2/nodes.py`: `Sort`, `ArraySort`, `Node`, `Model`.
  No domain knowledge — pure BTOR2.
- `rotor/btor2/printer.py`: `to_text(model) -> str`.
- `rotor/btor2/parser.py`: `from_text(str) -> ParseResult`,
  `from_path(...)`. Diagnostics-collecting (no exceptions on bad
  input). Accept the HWMCC superset (`zero`/`one`/`ones` shorthand,
  `const`/`consth`, trailing symbols) and normalize to canonical
  output.

**Tests:**

- `tests/unit/test_btor2.py`: every node kind, every sort, hand-built
  models round-trip through `to_text` then `from_text` and back.
- HWMCC golden corpus: a handful of public BTOR2 files parse and
  re-emit without diagnostics.

**Exit:** any rotor-emitted Model round-trips byte-for-byte through
`to_text → from_text → to_text`.

### Phase 2 — ELF and DWARF (1 day)

- `rotor/binary/elf.py`: `RISCVBinary(path)` exposing functions,
  instructions (variable-length, RVC-aware), loadable bytes.
- `rotor/binary/dwarf.py`: PC → SourceLocation lookup.

**Tests:** small fixture binary; assert function ranges, instruction
words, byte map, line lookups.

**Exit:** can load `tests/fixtures/add2.elf` and report `add2`'s PC
range, instruction stream, and source-line map.

### Phase 3 — RISC-V decoder (1.5 days)

- `rotor/riscv/decoder.py`: full RV64I + RV64M + RVC. Variable-length
  scan: low two bits decide 16 vs 32. RVC expands to its 32-bit
  equivalent before lowering.
- `rotor/riscv/disasm.py`: `Decoded → str` with the common
  pseudo-instruction shorthand.

**Tests:** every supported mnemonic has at least one (word, expected
Decoded) sample; reserved encodings return None; RVC expansions
match the spec table.

**Exit:** decoder + disasm cleanly handle the entire fixture corpus,
including a `gcc -O2 -march=rv64imc` binary.

### Phase 4 — Schema document (1 day)

Write SCHEMA.md fully. Every choice rotor will make in compilation
is recorded here before the corresponding code is written. This is
the contract; deviations from it in code are bugs.

**Exit:** SCHEMA.md is reviewable as a self-contained specification
of the rotor encoding, sufficient for an LLM (or human) to predict
rotor's BTOR2 output for any given RV64I+M+C function.

### Phase 5 — Per-instruction library (2 days)

- `rotor/riscv/library.py`: one function per supported mnemonic,
  taking `(decoded, pc, model, regs, mem) -> (writes, next_pc, next_mem)`.
  Strictly implements SCHEMA.md. No fallbacks, no defaults outside
  the schema.
- A *concrete witness simulator* mirroring each library lowering
  exactly: `rotor/lift/witness.py::simulate`. The two implementations
  must produce identical observable behavior on every concrete trace
  — this is asserted by tests and is the soundness story for witness
  replay.

**Tests:**

- Per-mnemonic unit: lower into a tiny synthetic model, run BMC at
  bound 1 with concrete inputs, assert the resulting state matches
  hand-computed expectations.
- Cross-check: run library lowering inside BMC against simulator on
  concrete inputs over a corpus of small instruction sequences;
  assert agreement.

**Exit:** every supported instruction lowers correctly; library and
simulator agree on every fixture instruction sequence; SCHEMA.md
sections 5 and the corresponding library code are 1-to-1.

### Phase 6 — QuestionSpec language (1 day)

- `rotor/spec/`: dataclasses for every spec element listed above.
  Pure data; no compilation logic.
- Validation: `validate(spec) -> list[Diagnostic]` checks structural
  consistency (registers in 0..31, comparison ops are valid, scope
  refers to functions present in the binary, etc.) without compiling.

**Tests:** valid specs validate clean; malformed specs surface
specific diagnostics.

**Exit:** the LLM-facing data type for questions exists, with no
compilation dependency.

### Phase 7 — Compile pipeline (3 days)

The largest single phase. Implements the hierarchical layering.

- `rotor/compile/header.py`: emits universal sorts.
- `rotor/compile/machine.py`: emits state declarations parameterized
  on core count and memory sharing.
- `rotor/compile/dispatch.py`: emits the PC-keyed ITE for the
  analyzed function set. Per SCHEMA.md, ascending PC order.
- `rotor/compile/init.py`: from `binary + spec.entry +
  spec.assumptions[RegisterInit | MemoryInit]`, emits init clauses
  + ELF segment writes.
- `rotor/compile/constraint.py`: from `spec.assumptions[CycleInvariant]
  + spec.learned + spec.entry`, emits constraint clauses.
- `rotor/compile/bad.py`: from `spec.observables + spec.property`,
  emits bad expression(s).
- `rotor/compile/binding.py`: emits `next` clauses wiring states
  through dispatch.
- `rotor/compile/havoc.py`: optional overlay replacing specific
  state `next` clauses with per-cycle inputs.
- `rotor/compile/linker.py`: parses `;@export` / `;@import`
  directives across layer files, resolves names, renumbers nids
  with offsets, emits flat BTOR2.
- `rotor/compile/annotation.py`: per-node metadata emitted alongside
  every layer; structured JSON.

**Tests:**

- Golden tests: a small corpus of `(spec, binary)` → expected
  `(layers, annotation, flattened)`. Byte-identical reproduction
  on re-compile.
- Layer reuse: changing only `spec.property` produces a new bad
  layer but identical header/machine/library/dispatch layers.
- Linker: cross-layer references resolve; nid collisions don't
  occur; flattened output parses through `btor2/parser.py`.
- L0-equivalence (in spirit): for each fixture in the corpus,
  asserting the flattened output's verdict matches a hand-checked
  expected verdict via `Z3BMC`.

**Exit:** rotor can compile a `QuestionSpec` to BTOR2 + annotation;
the BTOR2 dispatches successfully against external solvers; the
annotation's structure matches what `introspect` will need.

### Phase 8 — Solver dispatch (1.5 days)

- `rotor/dispatch/base.py`: `SolverBackend` Protocol — `dispatch(
  flattened: bytes, directive: AnalysisDirective) -> RawSolverResult`.
- `rotor/dispatch/z3bmc.py`: in-process via z3-solver.
- `rotor/dispatch/z3spacer.py`: in-process via z3.Fixedpoint.
- `rotor/dispatch/bitwuzla.py`, `cvc5.py`: optional, `ImportError`-
  guarded.
- `rotor/dispatch/pono.py`: subprocess, `shutil.which` guarded.

Important: dispatch returns the *raw* output. No verdict massaging,
no portfolio, no fallback. Concurrent invocation by the LLM is the
LLM's responsibility (each solver is constructed fresh per call so
contexts don't leak).

**Tests:** synthetic BTOR2 counter models; assert each backend
produces correct raw verdicts on simple cases.

**Exit:** the LLM (or a test harness) can hand a flattened BTOR2 +
directive to dispatch and get back a structured raw result.

### Phase 9 — Lift (1 day)

- `rotor/lift/witness.py`: replay a `RawSolverResult.raw_witness`
  through the concrete simulator, producing source-tagged steps
  using DWARF and the annotation.
- `rotor/lift/invariant.py`: parse a Spacer invariant string and
  re-name its references through the annotation (e.g. `pre_x10` →
  "register a0 at the current cycle").
- `rotor/lift/trace.py`: render `LiftedResult` as markdown / JSON.

**Tests:** for a known reachable fixture, lift produces a trace
whose source mapping matches expectations.

**Exit:** raw solver outputs become source-grounded structured
artifacts the LLM can introspect.

### Phase 10 — LLM tool surface (1 day)

- `rotor/llm/tools.py`: the five tool functions (`describe`,
  `compile`, `dispatch`, `lift`, `introspect`) with explicit
  serializable input/output schemas.
- `rotor/llm/mcp.py`: optional MCP server adapter exposing the
  five tools.

**Tests:** end-to-end simulated session: call `describe('addw')`,
`compile(spec)`, `dispatch(artifact, directive)`, `lift(artifact,
raw)`, `introspect(artifact, query)`. Assert outputs are
serializable and round-trip through JSON.

**Exit:** rotor is usable from an LLM session via the five tools.

### Phase 11 — CLI mirroring tools (½ day)

`rotor/cli.py`: subcommands `describe`, `compile`, `dispatch`,
`lift`, `introspect`. Useful for human debugging and for
non-LLM scripting.

**Exit:** CLI mirrors the tool surface; `rotor compile spec.json
--out artifact/` produces files that other rotor commands consume.

### Phase 12 — Examples and documentation (1 day)

- `examples/`: 3–5 short Python scripts demonstrating common
  question shapes. Each runs in CI as a smoke test.
- README rewritten to point at PLAN.md and SCHEMA.md as the
  primary references; quick start uses one tiny example.

**Exit:** a new user — human or LLM — can read README → PLAN →
SCHEMA in order and successfully run a small analysis.

## Total estimated effort: ~14 working days

Phases are largely sequential, with one parallelizable strand:
phases 4 (schema) and 5 (library) interleave (writing schema for
mnemonic X, then implementing it, then moving to mnemonic Y),
which compresses them somewhat.

## What is *not* in this plan

- **Multi-core / concurrency.** The state-declaration layer is
  designed to accommodate this (machine layer is parameterized on
  core count) but the v1 implementation is single-core only. Add
  this later when an LLM workflow demands it.
- **CEGAR-as-a-service.** Rotor never offers this. CEGAR is an LLM
  pattern composed from rotor calls. An *example* showing the LLM
  pattern can ship in `examples/`, but rotor itself does not ship
  CEGAR.
- **Equivalence checking.** Same — the LLM constructs a product
  spec; rotor compiles. Add a spec primitive for "two-binary product
  observation" if a real LLM workflow needs it; otherwise keep the
  spec language minimal.
- **A graded-canonicalization layer.** Future work. The artifact
  format is structured to accept it (annotations carry provenance,
  layers are individually addressable, learned facts are first-class)
  but rotor v1 does not implement canonicity grading. Adding it
  later is additive.
- **Synthesis-mode primitives** (find_input as a verb). The LLM
  expresses synthesis as an unsatisfiability question on a negated
  property and reads `reachable` as "synthesis succeeded." If real
  workflows show this is awkward, add explicit `goal` vs `bad`
  polarity in `Property`.

## Working notes for a fresh Claude Code session

When picking this up cold:

1. **Read in order:** README → PLAN.md (this file) → SCHEMA.md.
2. **Check `git log`** — phases land as separate commits with
   commit messages of the form `phase N: <one-line summary>`.
3. **Find the current phase** by reading the most recent commit
   and checking which phase exit criteria are met.
4. **Within a phase**, write the schema/spec language *before* the
   code that implements it. The contract precedes the implementation.
5. **Run `pytest -q`** before *and* after every change; if the
   smoke test stops importing, you've broken the package skeleton.
6. **Resist the urge to add reasoning into rotor.** If you find
   yourself writing "if the function has loops, do X automatically,"
   stop. Either codify the choice in SCHEMA.md (if it's truly
   universal) or add a spec parameter (if the LLM should choose).
   The architectural test: would removing this code make rotor
   simpler without changing what an LLM could in principle do?
   If yes, the code shouldn't be there.
7. **The schema is authoritative.** If code and schema disagree,
   the code is wrong; fix the code, not the schema. (If the schema
   is wrong, that's a versioned change with downstream cache
   invalidation.)
8. **The L0-equivalence harness lives in spirit.** There is no
   "L0 emitter" anymore — there's only the schema-deterministic
   compile pipeline. But the *idea* of golden tests asserting
   compilation reproducibility carries over: every change to
   compile must keep the corpus of `(spec, binary) → flattened
   BTOR2` byte-identical except where the schema version bumped.
9. **When in doubt, ask: would the LLM be able to predict this?**
   Rotor's invariant is determinism from `(spec, binary, schema)`.
   Anything that violates that — adaptivity, internal state,
   heuristics — is a bug in the design.

## A worked example to keep in mind

A canonical LLM workflow rotor must support cleanly:

1. LLM wants to know whether `bubble_sort` ever returns with `a0 != 0`
   on small arrays.
2. LLM calls `describe('verify')` — but rotor has no such verb.
   `describe` returns nothing, with a hint pointing at relevant
   schema sections (observables, properties, bad encoding).
3. LLM constructs a `QuestionSpec` with `RegisterAt(register=10,
   pc=<ret pc>)` as observable, `Property(bad = neq(observable, 0))`
   as property, `AnalysisDirective(engine='z3-bmc', bound=50)`.
4. LLM calls `compile(spec)`, gets a `CompiledArtifact` back.
5. LLM calls `dispatch(artifact, spec.analysis)`, gets a raw verdict.
6. If `unreachable`: LLM may construct a follow-up spec with
   `bound=200` or `engine='z3-spacer'`; rotor doesn't decide.
7. If `proved` (via Spacer): the raw invariant is in the result. The
   LLM may call `introspect` to see how the invariant references
   state variables, then construct a *new* question on a related
   sorting routine that includes the invariant as a `LearnedFact`
   in `assumptions`. Rotor compiles; the LLM dispatches; the new
   question may be much easier.
8. If `reachable`: LLM calls `lift(artifact, raw)` to get a
   source-grounded trace. The LLM interprets the trace and decides
   what to ask next.

At every step, rotor does mechanical work (translation, dispatch,
lift); the LLM does all the reasoning (what to ask, what to
believe, what to do next). That's the architectural commitment.