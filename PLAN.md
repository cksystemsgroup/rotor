# Hurdy-Gurdy: Implementation Plan

This is the working plan for implementing hurdy-gurdy from scratch as a
**framework for building (source, reasoning) translation pairs that
LLMs use to reason about programs through external solvers**.

The project ships as the `hurdy-gurdy` PyPI package, exposes the
`gurdy` CLI command, and is imported in Python as `gurdy`. Pair
identifiers are kebab-case strings: `riscv-btor2`, `python-smtlib`,
etc.; their Python sub-packages use the corresponding underscore form
(`gurdy.pairs.riscv_btor2`).

It is self-contained: a fresh Claude Code session should be able to
pick up from here without consulting prior context.

---

## What hurdy-gurdy is

Hurdy-gurdy compiles `(QuestionSpec, source program)` to a reasoning
artifact plus a structured annotation, dispatches the artifact to
external solvers, and lifts solver outputs back to source-level facts.

A *pair* is a fixed combination of a source language (e.g. RISC-V,
Python) and a reasoning language (e.g. BTOR2, SMT-LIB), with a
documented translation between them. Each pair is independent. The
first pair built is `riscv-btor2`; `python-smtlib` is the planned
second.

**The framework itself does no reasoning.** It does not:

- decide what to verify (the LLM constructs the spec)
- choose solvers, bounds, or timeouts (the spec specifies)
- refine abstractions or run CEGAR loops (the LLM iterates by
  re-specifying)
- compose or transfer invariants across questions (the LLM curates)
- interpret verdicts beyond mechanical lifting through the annotation

The invariant is: for any LLM session, given a pair, the LLM could in
principle predict the reasoning artifact by reading the pair's
**schema**, the **spec**, and the **source**, with no other information.
Same `(spec, source)` → byte-identical reasoning artifact. No internal
state, no learned heuristics, no adaptivity in compilation.

The reasoning lives in the LLM. Hurdy-gurdy is the substrate; the
schema plus annotation is the contract; the LLM is the agent.

## The instrument analogy in one paragraph

The project's name is not arbitrary. A hurdy-gurdy is an instrument
where the player turns a wheel and presses keys; the mechanism
deterministically produces sound on paired drone+melody strings. In
this project, a pair (source+reasoning) is the unit producing
meaningful output, the schema is the keyboard (deterministic
key-to-pitch mapping), compilation is the wheel (the mechanical step
that actually produces the artifact), the framework is the instrument
body (hosts the strings, supports the resonance, doesn't decide what's
played), and the LLM is the player (chooses what to play; the mechanism
handles production). README has a fuller version of this analogy.

## The split between the framework and the LLM

| Concern | Owner |
|---|---|
| What question to ask | LLM |
| How to express it as a `QuestionSpec` | LLM |
| How to compile a spec to a reasoning artifact | framework (schema-deterministic) |
| Which solver to invoke and with what budget | LLM (via spec) |
| How to dispatch the solver and capture its output | framework |
| How to interpret the verdict | LLM |
| How to lift solver state to source-level facts | framework (mechanical) |
| Whether/how to refine and re-ask | LLM |
| Whether an invariant from question A applies to question B | LLM |

Anywhere the framework would otherwise make a heuristic choice, that
choice becomes either schema-documented and fixed forever, or a spec
parameter the LLM specifies. There is no third option.

## Architectural commitments

### 1. The schema is the contract

A versioned document, **per pair**, spelling out every translation rule
from source semantics to the reasoning language. Every state-variable
convention, every construct lowering, every memory or heap model
decision, every entry-state assumption added by default. The schema is
authoritative; the pair implements it; the LLM consults it.

When a translation choice is not obvious (e.g. how to encode signed
division's overflow case in BTOR2, or how to model Python integer
arithmetic in SMT-LIB), the schema picks one and documents why.
Subsequent changes that affect the schema bump its version. Cached
compilation outputs are tagged with the schema version they were
produced under.

### 2. Pair model: no intermediate representation

Each pair is a direct translation from one source language to one
reasoning language. There is deliberately no shared intermediate
representation between source and reasoning. The reasons:

- **An IR is a forecasting exercise.** It would have to anticipate,
  before any specific pair is built, what semantic features the union
  of all future source languages will need and what the intersection of
  all future backends will accept. Get the forecast wrong and the IR
  fits poorly.
- **An IR is a third schema.** Adding an IR adds a source-to-IR schema
  and an IR-to-reasoning schema, plus the IR specification. That's a
  tripling of documentation surface, and each must be kept in sync.
  Three-stage debugging is much harder than one-stage.
- **The `m + n` promise is illusory.** Most cross-products of source
  and reasoning languages aren't actually wanted. RISC-V wants
  bitvector reasoning. Python wants theory-rich reasoning. The natural
  pairings are few; the IR's combinatorial argument doesn't apply.
- **It forces premature abstraction.** Without empirical data from
  multiple pairs, designing an IR is guessing. The pair model defers
  generalization until evidence justifies it.

What *is* shared is mechanical infrastructure: annotation, caching,
layer linking, dispatch, provenance, tool surface. These are
language-agnostic and live in the framework.

If, after several pairs exist, common semantic structure emerges that's
worth abstracting, it can be added with empirical grounding. Until
then, no IR.

### 3. Hierarchical layered artifacts

A pair's reasoning artifact is split into named *layers*, each with its
own stability profile and content hash. The `riscv-btor2` layers are:

| Layer | Stability | Contents |
|---|---|---|
| **header** | universal | sort declarations |
| **machine** | per ISA + core count | state variable declarations |
| **library** | per ISA | per-instruction lowering definitions |
| **dispatch** | per analyzed function set | PC-keyed ITE selecting which library lowering applies |
| **init** | per question | initial-state clauses |
| **constraint** | per question, accumulates | invariants and assumptions; carries provenance per clause |
| **bad** | per question | property under investigation |
| **binding** | per question | wires `next` clauses connecting states to dispatch |
| **havoc** | per question (overlay) | optional overrides replacing specific `next` clauses with fresh inputs |

Cross-layer references go through symbolic names (`(export name nid)`
and `(import name)` directives in `;@`-comment headers, for BTOR2;
analogous mechanisms for other reasoning languages). A linker resolves
names, renumbers IDs, and flattens to standard reasoning-language
syntax for the solver. The hierarchy is internal; the solver sees
standard output.

Other pairs declare their own layer sets via the framework's
`LayerSpec` mechanism. The linker is generic; the layer declarations
are pair-specific.

This factoring is essential for the LLM use case: when an LLM asks
many questions about one source program, the stable bottom layers are
reused across questions, and the diff between questions is exactly the
volatile top layers. Incremental analysis is cheap; the *semantic*
relationship between questions is explicit.

### 4. Annotation sidecar

Alongside each layer, hurdy-gurdy emits a structured annotation file
recording, for every emitted node:

- **role** (state, transition, init-clause, entry-assumption,
  learned-invariant, bad-expression, …)
- **source mapping** (per-pair: PC + DWARF for RISC-V; AST node + line
  for Python; …)
- **provenance** (library version, user-specified parameter, invariant
  from prior question N validated against …)

The annotation framework is shared (data types, persistence, lookup,
the `introspect` query engine); the *vocabulary* of source mappings is
pair-specific.

### 5. No reasoning policies in the framework

There is no portfolio. There is no CEGAR loop. There is no automatic
slicing. The LLM dispatches one solver at a time with explicit
parameters; if it wants to race, it dispatches in parallel itself; if
it wants CEGAR, it loops over compile/dispatch calls with
progressively-refined specs.

Spec parameters allow the LLM to *request* things the framework knows
how to do (e.g. "havoc registers `{r1, r5, r7}`"). The pair encodes
the request mechanically. It does not decide which registers to havoc.

### 6. Framework / pair separation

Hurdy-gurdy is split into a language-agnostic *core* (the framework)
and one or more *pairs* (the language-specific plugins).

The core owns: tool surface and CLI, spec validation framework,
annotation sidecar machinery, layer declaration and linking, solver
dispatch wrappers, content-addressed caching, structured diagnostics,
schema-document indexer.

A pair owns: source loader, `SCHEMA.md`, spec vocabulary, translator,
lifter, solver wrappers.

The interface between them is a single `Pair` protocol object.

## Repository layout

```
hurdy-gurdy/
├── README.md                       ← what hurdy-gurdy is
├── PLAN.md                         ← this file
├── LICENSE                         ← MIT
├── pyproject.toml                  ← name = "hurdy-gurdy"
├── gurdy/
│   ├── core/                       ← framework, language-agnostic
│   │   ├── __init__.py
│   │   ├── pair.py                 ← Pair Protocol + registry
│   │   ├── spec/
│   │   │   ├── base.py             ← BaseSpec, BaseObservable, ...
│   │   │   └── validate.py         ← validation harness
│   │   ├── annotation/
│   │   │   ├── types.py            ← Annotation, NodeProvenance, ...
│   │   │   ├── sidecar.py          ← persistence
│   │   │   └── lookup.py           ← introspect query engine
│   │   ├── layers/
│   │   │   ├── declaration.py      ← LayerSpec + dependency resolution
│   │   │   └── linker.py           ← cross-layer name resolution
│   │   ├── dispatch/
│   │   │   ├── backend.py          ← SolverBackend Protocol
│   │   │   ├── timeout.py
│   │   │   └── result.py           ← RawResult types
│   │   ├── cache/
│   │   │   └── content_addressed.py
│   │   ├── diagnostics.py
│   │   ├── schema/
│   │   │   └── indexer.py          ← SCHEMA.md → describe() index
│   │   ├── tools/                  ← LLM-facing tool surface
│   │   │   ├── describe.py
│   │   │   ├── compile.py
│   │   │   ├── dispatch.py
│   │   │   ├── lift.py
│   │   │   └── introspect.py
│   │   └── cli.py                  ← `gurdy` command entry point
│   └── pairs/
│       ├── riscv_btor2/            ← pair identifier "riscv-btor2"
│       │   ├── __init__.py         ← registers the pair
│       │   ├── SCHEMA.md           ← the contract
│       │   ├── source/             ← ELF, DWARF, decoder
│       │   ├── btor2/              ← BTOR2 in-memory model + I/O
│       │   ├── spec.py             ← RegisterAt, MemoryAt, ...
│       │   ├── translation/        ← schema-implementing emission
│       │   ├── lift.py
│       │   └── solvers/            ← z3bmc, z3spacer, bitwuzla, pono
│       └── python_smtlib/          ← pair identifier "python-smtlib"
│           └── ...                 ← (built post-v1)
├── tests/
│   ├── core/                       ← framework tests, no pair dependency
│   ├── pairs/
│   │   └── riscv_btor2/
│   │       ├── unit/
│   │       ├── golden/             ← compile-output regression
│   │       └── integration/
│   └── fixtures/                   ← small RV ELFs + .c + build.sh
└── examples/                       ← short scripts using gurdy as a library
```

The directory `pairs/riscv_btor2/` (Python identifier; underscores)
hosts the pair whose registered string identifier is `riscv-btor2`
(kebab-case, matching CLI conventions). This split is routine: Python
imports want underscores, user-facing strings want hyphens.

What's deliberately absent:

- no `cegar.py` — CEGAR is an LLM pattern, not a hurdy-gurdy module
- no `solvers/portfolio.py` — racing is the LLM's responsibility
- no `ir/` — no intermediate representation
- no auto-slicing or auto-optimization

## The Pair protocol

A pair is a single object exposed via the framework's registry:

```python
@dataclass(frozen=True)
class Pair:
    identifier: str                       # 'riscv-btor2', 'python-smtlib', ...
    schema_version: str                   # contract version for caching
    
    # Loading source
    source_loader: SourceLoader           # bytes -> Source
    
    # Spec language
    spec_class: type[BaseSpec]            # the pair's QuestionSpec subclass
    spec_validator: SpecValidator         # spec + source -> diagnostics
    
    # Translation
    layer_specs: tuple[LayerSpec, ...]    # what layers the pair emits
    translator: Translator                # spec + source -> layered + annotation
    
    # Output interpretation
    lifter: Lifter                        # raw_result + artifact -> lifted
    
    # Solvers
    solvers: Mapping[str, type[SolverBackend]]
    
    # Documentation
    schema_path: Path
```

Each callable above is a small Protocol. For instance:

```python
class Translator(Protocol):
    def translate(
        self,
        spec: BaseSpec,
        source: Source,
        annotation_emitter: AnnotationEmitter,
    ) -> CompiledArtifact:
        ...
```

The translator receives an `annotation_emitter` from the framework and
records provenance during translation. Layer machinery, annotation
format, and emission infrastructure are framework concerns; the
translator just uses them.

A new pair registers itself in its `__init__.py`:

```python
from gurdy.core.pair import Pair, register_pair

PAIR = Pair(
    identifier='riscv-btor2',
    schema_version='1.0.0',
    source_loader=load_riscv_binary,
    spec_class=RiscvBtor2Spec,
    spec_validator=validate_riscv_btor2_spec,
    layer_specs=RISCV_BTOR2_LAYERS,
    translator=translate,
    lifter=lift,
    solvers={'z3-bmc': Z3BMCSolver, 'z3-spacer': Z3SpacerSolver, ...},
    schema_path=Path(__file__).parent / 'SCHEMA.md',
)

register_pair(PAIR)
```

That is the *entire* surface a pair exposes to the framework.

## What the framework provides (and pairs inherit)

- **Tool surface** (`describe`, `compile`, `dispatch`, `lift`,
  `introspect`) and CLI (`gurdy`) mirroring it. Pairs don't write tool
  wrappers.
- **Spec validation framework** with `BaseSpec` scaffold, structural
  validation, serialization, hashing for caching. Pairs subclass and
  add their vocabulary.
- **Annotation sidecar** with shared data types, persistence to JSON,
  lookup engine. Pairs emit annotations using framework types.
- **Layered artifact format** with generic linker resolving cross-layer
  names. Pairs declare their layer sets.
- **Dispatch wrappers** with `SolverBackend` Protocol, timeout
  enforcement, structured `RawResult`. Pairs implement shallow
  solver-specific subclasses.
- **Content-addressed caching** keyed on
  `(spec_hash, source_hash, schema_version)`. Pairs benefit
  automatically since compilation is deterministic.
- **Diagnostics framework** with shared `Diagnostic` type and uniform
  rendering.
- **Schema-document indexer** that parses each pair's `SCHEMA.md` and
  serves `describe(topic, pair)`.
- **Provenance threading** through every artifact, automatically.
- **LearnedFact infrastructure** with shared data type and constraint
  injection; pairs only translate their learned-fact expressions.

A new pair after the first is roughly: source loader, schema document,
spec vocabulary, translation function, lifter, solver wrappers. Five
pieces of work. Everything else is framework.

## What stays pair-specific (irreducibly)

- The source loader (intrinsically format-specific)
- The `SCHEMA.md` document (the largest pair deliverable)
- The translation function (no abstraction makes "this RISC-V
  instruction becomes this BTOR2 fragment" shareable across pairs)
- The observable, assumption, and property vocabulary
- The lift from raw solver output to source-level facts
- Solver wrappers compatible with the pair's reasoning language

These are pair-specific by design. The framework provides scaffolding
(base classes, protocols, helpers) but does not constrain the content.

## The QuestionSpec language (`riscv-btor2` pair)

Each pair has its own QuestionSpec subclass. Below is the `riscv-btor2`
pair's spec; other pairs follow the same structural pattern with their
own vocabulary.

```python
@dataclass(frozen=True)
class RiscvBtor2Spec(BaseSpec):
    binary: BinaryRef                         # path + content hash
    scope: AnalysisScope                      # entry function + included callees
    entry: EntryAssumptions                   # ra/sp/argument constraints
    observables: tuple[Observable, ...]       # what to expose
    assumptions: tuple[Assumption, ...]       # constraints to add at every cycle
    learned: tuple[LearnedFact, ...]          # invariants from prior questions
    property: Property                        # bad / goal expression
    analysis: AnalysisDirective               # solver + bound + timeout
```

### Observables (`riscv-btor2`)

```python
@dataclass(frozen=True)
class RegisterAt(Observable):
    register: int                             # 0..31
    pc: int                                   # observe at this PC

@dataclass(frozen=True)
class MemoryAt(Observable):
    address: Expression
    width: int                                # bytes
    pc: int

@dataclass(frozen=True)
class PCAtStep(Observable):
    step: int

@dataclass(frozen=True)
class Executed(Observable):
    pc: int                                   # was this PC visited within bound?
```

### Assumptions, LearnedFacts, EntryAssumptions, AnalysisDirective

```python
@dataclass(frozen=True)
class RegisterInit(Assumption):
    register: int
    op: Comparison
    value: int

@dataclass(frozen=True)
class MemoryInit(Assumption):
    address: int
    width: int
    op: Comparison
    value: int

@dataclass(frozen=True)
class CycleInvariant(Assumption):
    expression: Expression
    provenance: Provenance

@dataclass(frozen=True)
class LearnedFact:
    expression: Expression
    source_question_hash: str
    source_engine: str
    validated: bool

@dataclass(frozen=True)
class EntryAssumptions:
    excluded_pc_ranges: tuple[tuple[int, int], ...]

@dataclass(frozen=True)
class AnalysisDirective:
    engine: str                               # 'z3-bmc' | 'z3-spacer' | 'bitwuzla' | …
    bound: int | None
    timeout: float | None
    havoc_registers: frozenset[int]
    extra_options: Mapping[str, str]
```

If the LLM wants a portfolio, it submits N specs and dispatches them
itself. If it wants CEGAR, it loops with progressively-shrinking
`havoc_registers`. The framework sees one spec, runs one solver,
returns one artifact.

## The LLM-facing tool surface

Five tools, mechanical semantics, the same across all pairs:

### `describe(topic: str, pair: str) -> SchemaEntry`

Returns the schema entry for a topic from the named pair: an
instruction's lowering, a state-variable convention, an entry-assumption
default, a verdict type. The LLM consults this on demand.

### `compile(spec: BaseSpec) -> CompiledArtifact`

Deterministic. The spec's pair identifier routes to the right
translator. Returns:

```python
@dataclass(frozen=True)
class CompiledArtifact:
    pair: str                                 # which pair produced this
    layers: Mapping[str, Layer]               # per-pair layer set
    annotation: AnnotationSidecar             # per-node metadata
    flattened: bytes                          # linker output, ready for solvers
    schema_version: str
    spec_hash: str                            # reproducibility key
```

### `dispatch(artifact: CompiledArtifact, directive: AnalysisDirective) -> RawSolverResult`

Wraps the chosen external solver (must be one the pair declares).
Returns:

```python
@dataclass(frozen=True)
class RawSolverResult:
    verdict: str                              # 'reachable' | 'unreachable' | 'proved' | 'unknown'
    elapsed: float
    engine: str
    payload: Any                              # pair-specific (witness, invariant, ...)
    reason: str | None                        # for unknown
```

### `lift(artifact: CompiledArtifact, raw: RawSolverResult) -> LiftedResult`

Mechanical translation through the annotation. Calls into the pair's
lifter for source-language-specific interpretation; the framework
formats the result.

### `introspect(artifact: CompiledArtifact, query: IntrospectQuery) -> IntrospectResult`

Read-only query against the annotation. Pure lookup.

That's the entire LLM surface. Anything richer is composed from these
primitives in the LLM's own logic.

## Phased implementation

Each phase ships a contract-tested artifact. No phase relies on
implementation details of a later phase. CI gates each phase. Phases
land as separate commits with messages of the form
`phase N: <one-line summary>`.

Phases are organized as **framework first, then `riscv-btor2` pair,
then v1 ship, then second pair**. The framework phases have no pair
dependency; the pair phases use the framework but don't extend it.

### Phase 0 — Repository scaffolding (½ day)

- `README.md`, `PLAN.md`, `LICENSE` (MIT), `pyproject.toml` (name =
  `hurdy-gurdy`, console script `gurdy = gurdy.core.cli:main`)
- Empty package skeleton matching the layout above
- `tests/` with `conftest.py` and a single smoke test
- CI configuration: pytest on Python 3.11, 3.12

**Exit:** `pip install -e .` succeeds; `pytest -q` passes the smoke
test; `gurdy --help` runs and prints a stub message; `pre-commit` (or
equivalent) is wired up.

### Phase 1 — Core framework: data types and protocols (1 day)

Framework, no pair dependency.

- `gurdy/core/pair.py`: `Pair` dataclass, `register_pair`, `get_pair`,
  registry singleton, all related Protocols (`SourceLoader`,
  `SpecValidator`, `Translator`, `Lifter`, `SolverBackend`).
- `gurdy/core/spec/base.py`: `BaseSpec`, `BaseObservable`,
  `BaseAssumption`, `BaseProperty`, `BaseAnalysisDirective`.
- `gurdy/core/diagnostics.py`: `Diagnostic`, severity levels,
  rendering.

**Tests:** types instantiate; Protocols accept structurally-correct
mocks; registry roundtrips registration and lookup.

**Exit:** the framework's contract surface exists and is type-checked.

### Phase 2 — Core framework: annotation and layers (1 day)

- `gurdy/core/annotation/types.py`: `Annotation`, `NodeProvenance`,
  `Role`, `LearnedFactProvenance`, `SourceMapping` (base class for
  pair-specific extensions).
- `gurdy/core/annotation/sidecar.py`: serialize/deserialize to JSON;
  content addressing.
- `gurdy/core/annotation/lookup.py`: query engine for `introspect`.
- `gurdy/core/layers/declaration.py`: `LayerSpec` with name, stability,
  dependencies.
- `gurdy/core/layers/linker.py`: cross-layer name resolution. The
  linker is parameterized over a pair-supplied parser/printer for
  cross-layer reference syntax, so it doesn't bake in BTOR2's
  `;@export`/`;@import` convention.

**Tests:** annotation round-trips through serialization; linker
resolves a hand-built layer graph with synthetic syntax.

**Exit:** framework can persist annotations and link layers without
knowing about any specific pair.

### Phase 3 — Core framework: caching, dispatch, schema indexer (1 day)

- `gurdy/core/cache/content_addressed.py`: keyed on
  `(spec_hash, source_hash, schema_version)`; pluggable
  `cache_key_extras` hook for pairs that need supplementary keying.
- `gurdy/core/dispatch/backend.py`: `SolverBackend` Protocol with
  `dispatch(artifact_bytes, directive) -> RawResult`; subprocess
  helpers; in-process helpers; timeout enforcement.
- `gurdy/core/dispatch/result.py`: `RawSolverResult` with typed
  pair-specific `payload`.
- `gurdy/core/schema/indexer.py`: parses `SCHEMA.md` files in
  registered pairs by H2/H3 headings, serves entries by topic.

**Tests:** cache hits and misses; subprocess solver wrapper with a mock
binary; schema indexer parses a sample SCHEMA.md and serves entries.

**Exit:** framework can cache, dispatch (synthetically), and serve
schema entries without any pair installed.

### Phase 4 — Core framework: tool surface and CLI (1 day)

- `gurdy/core/tools/{describe,compile,dispatch,lift,introspect}.py`:
  the five tools, routing by pair identifier.
- `gurdy/core/cli.py`: subcommands `gurdy describe`, `gurdy compile`,
  `gurdy dispatch`, `gurdy lift`, `gurdy introspect`, mirroring the
  tools.

**Tests:** end-to-end with a synthetic minimal pair (registered in
test fixtures): full tool surface produces expected outputs through
both Python API and `gurdy` CLI.

**Exit:** the framework is a working empty platform — pair-less but
exercisable end-to-end with a synthetic pair.

### Phase 5 — `riscv-btor2` pair: BTOR2 in-memory + text I/O (1 day)

This and subsequent phases build the first pair, under
`gurdy/pairs/riscv_btor2/`.

- `pairs/riscv_btor2/btor2/nodes.py`: `Sort`, `ArraySort`, `Node`,
  `Model`. No domain knowledge.
- `pairs/riscv_btor2/btor2/printer.py`: `to_text(model) -> str`.
- `pairs/riscv_btor2/btor2/parser.py`: `from_text(str) -> ParseResult`
  with diagnostic collection. Accept the HWMCC superset and normalize
  to canonical output.

**Tests:** every node kind round-trips through `to_text → from_text`;
HWMCC golden corpus parses and re-emits without diagnostics.

**Exit:** any emitted Model round-trips byte-for-byte.

### Phase 6 — `riscv-btor2` pair: ELF and DWARF (1 day)

- `pairs/riscv_btor2/source/elf.py`: `RISCVBinary(path)` exposing
  functions, instructions (variable-length, RVC-aware), loadable bytes.
- `pairs/riscv_btor2/source/dwarf.py`: PC → SourceLocation lookup.
- `pairs/riscv_btor2/source/loader.py`: implements `SourceLoader`,
  bridging the above into the framework.

**Tests:** small fixture binary; assert function ranges, instruction
words, byte map, line lookups.

**Exit:** can load `tests/fixtures/add2.elf` and report `add2`'s PC
range, instruction stream, source-line map; the framework recognizes
the loader.

### Phase 7 — `riscv-btor2` pair: decoder (1.5 days)

- `pairs/riscv_btor2/source/decoder.py`: full RV64I + RV64M + RVC.
  Variable-length scan; RVC expands to its 32-bit equivalent before
  lowering.
- `pairs/riscv_btor2/source/disasm.py`: `Decoded → str` with common
  pseudo-instruction shorthand.

**Tests:** every supported mnemonic has at least one (word, expected
Decoded) sample; reserved encodings return None; RVC expansions match
the spec table.

**Exit:** decoder + disasm cleanly handle the entire fixture corpus,
including a `gcc -O2 -march=rv64imc` binary.

### Phase 8 — `riscv-btor2` pair: SCHEMA.md (1 day)

Write `pairs/riscv_btor2/SCHEMA.md` fully. Every choice the translation
will make is recorded here before the corresponding code is written.
This is the contract; deviations from it in code are bugs.

Sections:

1. **Versioning.**
2. **Sorts.** Fixed bitvector widths and the array sort.
3. **State variables.** Register and PC conventions, memory model.
4. **ELF loading.** PT_LOAD bytes → memory init; uninitialized regions
   → free.
5. **Instruction lowering.** One subsection per RV64I+M+C mnemonic,
   spelling out the BTOR2 fragment, with explicit attention to:
   signedness, overflow, divide-by-zero, shift-amount masking, byte
   ordering, sign- vs zero-extension.
6. **Dispatch.** PC-keyed ITE; arm ordering (ascending by PC); how PCs
   outside the analyzed set self-loop.
7. **Entry assumptions.** Default `ra` outside analyzed set; `sp` left
   free unless specified.
8. **Constraint and bad encoding.** Polarity conventions; how
   multi-clause constraints aggregate.
9. **Havoc semantics.** What it means to havoc a register; how memory
   havoc would be encoded if added later.
10. **Verdict semantics.** What `reachable`, `unreachable`, `proved`,
    `unknown` mean from each engine; how `bound` is interpreted.

**Exit:** SCHEMA.md is reviewable as a self-contained specification of
the `riscv-btor2` encoding, sufficient for an LLM (or human) to predict
the BTOR2 output for any given RV64I+M+C function.

### Phase 9 — `riscv-btor2` pair: per-instruction library (2 days)

- `pairs/riscv_btor2/translation/library.py`: one function per
  supported mnemonic, taking
  `(decoded, pc, model, regs, mem) -> (writes, next_pc, next_mem)`.
  Strictly implements SCHEMA.md.
- A *concrete witness simulator* mirroring each library lowering
  exactly: `pairs/riscv_btor2/lift/simulator.py::simulate`. The two
  implementations must produce identical observable behavior on every
  concrete trace — this is the soundness story for witness replay.

**Tests:**

- Per-mnemonic unit: lower into a tiny synthetic model, run BMC at
  bound 1 with concrete inputs, assert the resulting state matches
  hand-computed expectations.
- Cross-check: run library lowering inside BMC against simulator on
  concrete inputs over a corpus of small instruction sequences; assert
  agreement.

**Exit:** every supported instruction lowers correctly; library and
simulator agree on every fixture instruction sequence; SCHEMA.md
section 5 and the corresponding library code are 1-to-1.

### Phase 10 — `riscv-btor2` pair: spec language (½ day)

- `pairs/riscv_btor2/spec.py`: `RiscvBtor2Spec` (subclass of
  `BaseSpec`) and the pair-specific observable/assumption/property
  types listed earlier.
- Validator: structural consistency without compiling.

**Tests:** valid specs validate clean; malformed specs surface
specific diagnostics through the framework's diagnostics channel.

**Exit:** the LLM-facing data type for `riscv-btor2` questions exists
and integrates with the framework's spec validation.

### Phase 11 — `riscv-btor2` pair: translation pipeline (3 days)

The largest single phase. Implements the layered translation.

- `pairs/riscv_btor2/translation/header.py`: emits universal sorts.
- `pairs/riscv_btor2/translation/machine.py`: emits state declarations
  (parameterized on core count for future multi-core).
- `pairs/riscv_btor2/translation/dispatch.py`: emits the PC-keyed ITE
  for the analyzed function set; ascending PC order per SCHEMA.md.
- `pairs/riscv_btor2/translation/init.py`: from
  `binary + spec.entry + spec.assumptions[RegisterInit | MemoryInit]`,
  emits init clauses + ELF segment writes.
- `pairs/riscv_btor2/translation/constraint.py`: from
  `spec.assumptions[CycleInvariant] + spec.learned + spec.entry`,
  emits constraint clauses.
- `pairs/riscv_btor2/translation/bad.py`: from
  `spec.observables + spec.property`, emits bad expression(s).
- `pairs/riscv_btor2/translation/binding.py`: emits `next` clauses
  wiring states through dispatch.
- `pairs/riscv_btor2/translation/havoc.py`: optional overlay replacing
  specific state `next` clauses with per-cycle inputs.
- `pairs/riscv_btor2/translation/translate.py`: top-level translator
  implementing the framework's `Translator` protocol; orchestrates the
  above and uses the framework's annotation emitter throughout.

The framework's linker handles cross-layer name resolution and
flattening; the pair supplies the BTOR2-specific syntax for export/
import directives via a small parser/printer registered with the
linker.

**Tests:**

- Golden tests: a small corpus of `(spec, binary)` → expected
  `(layers, annotation, flattened)`. Byte-identical reproduction on
  re-compile.
- Layer reuse: changing only `spec.property` produces a new bad layer
  but identical header/machine/library/dispatch layers.
- Linker: cross-layer references resolve; nid collisions don't occur;
  flattened output parses through the BTOR2 parser.
- Verdict cross-check: for each fixture, the flattened output's
  verdict (via Z3BMC) matches a hand-checked expected verdict.

**Exit:** the `riscv-btor2` pair compiles a `RiscvBtor2Spec` to BTOR2 +
annotation; the BTOR2 dispatches successfully against external
solvers; the annotation's structure matches what `introspect` needs.

### Phase 12 — `riscv-btor2` pair: solver wrappers (1 day)

- `pairs/riscv_btor2/solvers/z3bmc.py`: in-process via z3-solver.
- `pairs/riscv_btor2/solvers/z3spacer.py`: in-process via
  z3.Fixedpoint.
- `pairs/riscv_btor2/solvers/bitwuzla.py`, `cvc5.py`: optional,
  `ImportError`-guarded.
- `pairs/riscv_btor2/solvers/pono.py`: subprocess, `shutil.which`
  guarded.

Each is a shallow subclass of the framework's `SolverBackend`,
returning the framework's `RawSolverResult` with a BTOR2-pair-specific
payload (witness in BTOR2 model format, invariant in SMT-LIB syntax,
…).

**Tests:** synthetic BTOR2 counter models; assert each backend produces
correct raw verdicts on simple cases.

**Exit:** the LLM can hand a flattened BTOR2 + directive to dispatch
and get back a structured raw result.

### Phase 13 — `riscv-btor2` pair: lift (1 day)

- `pairs/riscv_btor2/lift/witness.py`: replay
  `RawSolverResult.payload`'s witness through the simulator, producing
  source-tagged steps using DWARF and the annotation.
- `pairs/riscv_btor2/lift/invariant.py`: parse a Spacer invariant
  string and re-name its references through the annotation.
- `pairs/riscv_btor2/lift/lift.py`: top-level lifter implementing the
  framework's `Lifter` protocol.

**Tests:** for a known reachable fixture, lift produces a trace whose
source mapping matches expectations.

**Exit:** raw solver outputs become source-grounded structured artifacts
through the framework's `lift` tool.

### Phase 14 — `riscv-btor2` pair: registration and v1 ship (½ day)

- `pairs/riscv_btor2/__init__.py`: assembles the `Pair` object and
  calls `register_pair`.
- End-to-end smoke test: from a fresh import of the package, the LLM
  tool surface works against the `riscv-btor2` pair.

**Exit:** v1 ships. `pip install hurdy-gurdy` provides a working
`gurdy` CLI and Python API with the `riscv-btor2` pair built in.

### Phase 15 — Examples and documentation (1 day)

- `examples/`: 3–5 short Python scripts demonstrating common question
  shapes against the `riscv-btor2` pair. Each runs in CI as a smoke
  test.
- README finalized; ensure SCHEMA.md is referenced correctly; framework
  README documents the Pair protocol for future pair authors.

**Exit:** a new user — human or LLM — can read README → PLAN →
`riscv-btor2` SCHEMA in order and successfully run a small analysis.

### Phase 16 (deferred, post-v1) — `python-smtlib` pair

Built after v1 ships. Validates that the framework is genuinely thin.
Estimated at 10–13 days for a defined Python subset.

Deliverables: `pairs/python_smtlib/SCHEMA.md`, source loader (Python
AST + restricted type model), spec vocabulary, translation function,
lift, solver wrappers (Z3, cvc5).

If, after this pair lands, common semantic structure between the two
pairs surfaces that's worth abstracting, that's the point at which we
revisit the no-IR commitment. Until then, the two pairs share the
framework and nothing else.

## Total estimated effort

- Framework (phases 0–4): ~4.5 days
- `riscv-btor2` pair (phases 5–14): ~12 days
- v1 ship + docs (phase 15): 1 day
- **v1 total: ~17 working days**
- `python-smtlib` pair (phase 16, post-v1): ~10–13 days

The framework cost is largely fixed; subsequent pairs incur only the
language-specific work.

## What is *not* in this plan

- **Multi-core / concurrency.** The state-declaration layer is
  parameterized on core count but v1 is single-core only.
- **CEGAR-as-a-service.** Hurdy-gurdy never offers this. CEGAR is an
  LLM pattern composed from compile/dispatch calls. An *example*
  showing the LLM pattern can ship in `examples/`.
- **Equivalence checking.** Same — the LLM constructs a product spec;
  the framework compiles. Add a spec primitive for "two-source product
  observation" if a real LLM workflow needs it.
- **A graded-canonicalization layer.** Future work. The annotation
  format and `LearnedFact` provenance are structured to accept it; v1
  doesn't implement it.
- **Synthesis-mode primitives** (find_input as a verb). The LLM
  expresses synthesis as unsatisfiability of a negated property. If
  real workflows show this is awkward, add explicit `goal` vs `bad`
  polarity in `Property`.
- **An intermediate representation.** See "Architectural commitments,
  point 2." Not in v1; revisited only if multi-pair empirical
  evidence justifies it.

## Working notes for a fresh Claude Code session

When picking this up cold:

1. **Read in order:** README → PLAN.md (this file) → after Phase 8,
   `gurdy/pairs/riscv_btor2/SCHEMA.md`.
2. **Check `git log`** — phases land as separate commits with messages
   `phase N: <one-line summary>`.
3. **Find the current phase** by reading the most recent commit and
   checking which phase exit criteria are met.
4. **Within a phase**, write the schema/spec language *before* the
   code that implements it. The contract precedes the implementation.
5. **Run `pytest -q`** before *and* after every change.
6. **Resist the urge to add reasoning into the framework or pairs.**
   If you find yourself writing "if the function has loops, do X
   automatically," stop. Either codify the choice in SCHEMA.md (if
   truly universal for this pair) or add a spec parameter (if the LLM
   should choose). The architectural test: would removing this code
   make hurdy-gurdy simpler without changing what an LLM could in
   principle do? If yes, the code shouldn't be there.
7. **Maintain framework/pair separation.** Code in `gurdy/core/` must
   not mention BTOR2, RISC-V, SMT-LIB, or any specific pair by name.
   If you're tempted to special-case framework behavior for a pair's
   needs, the pair's protocol is wrong — fix the protocol, not the
   framework.
8. **The schema is authoritative.** If code and schema disagree, the
   code is wrong; fix the code, not the schema. (If the schema is
   wrong, that's a versioned change with downstream cache invalidation.)
9. **No IR.** If you find yourself wanting to introduce an
   intermediate representation between source and reasoning, stop and
   re-read "Architectural commitments, point 2." The pair model is the
   commitment; an IR is explicitly out of scope until empirical
   evidence from multiple pairs justifies it.
10. **Names are settled.** Project: `hurdy-gurdy` (PyPI). Package and
    CLI: `gurdy`. Pair identifiers: `riscv-btor2`, `python-smtlib`
    (kebab-case strings); their Python sub-packages use the underscore
    form (`riscv_btor2`, `python_smtlib`). Don't drift.
11. **When in doubt, ask: would the LLM be able to predict this?**
    The invariant is determinism from `(spec, source, schema)`.
    Anything that violates that — adaptivity, internal state, heuristics
    — is a bug in the design.

## A worked example to keep in mind

A canonical LLM workflow hurdy-gurdy must support cleanly with the
`riscv-btor2` pair:

1. LLM wants to know whether `bubble_sort` ever returns with `a0 != 0`
   on small arrays.
2. LLM calls `describe('verify', pair='riscv-btor2')` — but there is
   no such verb. `describe` returns nothing, with a hint pointing at
   relevant schema sections (observables, properties, bad encoding).
3. LLM constructs a `RiscvBtor2Spec` with `RegisterAt(register=10,
   pc=<ret pc>)` as observable, `Property(bad=neq(observable, 0))` as
   property, `AnalysisDirective(engine='z3-bmc', bound=50)`.
4. LLM calls `compile(spec)`, gets a `CompiledArtifact` back.
5. LLM calls `dispatch(artifact, spec.analysis)`, gets a raw verdict.
6. If `unreachable`: LLM may construct a follow-up spec with
   `bound=200` or `engine='z3-spacer'`; the framework doesn't decide.
7. If `proved` (via Spacer): the raw invariant is in the result. The
   LLM may call `introspect` to see how the invariant references state
   variables, then construct a *new* question on a related sorting
   routine that includes the invariant as a `LearnedFact` in
   `assumptions`. The framework compiles; the LLM dispatches; the new
   question may be much easier.
8. If `reachable`: LLM calls `lift(artifact, raw)` to get a
   source-grounded trace. The LLM interprets the trace and decides
   what to ask next.

At every step, hurdy-gurdy does mechanical work (translation, dispatch,
lift); the LLM does all the reasoning (what to ask, what to believe,
what to do next). That's the architectural commitment.