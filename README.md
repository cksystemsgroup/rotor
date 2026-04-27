# rotor

A minimal, LLM-programmable platform for reasoning about RISC-V code.

Rotor compiles `(QuestionSpec, RISC-V binary)` into BTOR2 plus a
structured semantic annotation, dispatches the result to external
SMT and model-checking solvers, and lifts solver outputs back to
source-level facts. That is the entire scope.

Rotor itself does no reasoning. It does not decide what to verify,
choose solvers, refine abstractions, run CEGAR loops, or compose
invariants across questions. Those are the LLM's job. Rotor's job
is to translate, dispatch, and lift — mechanically and
deterministically — and to convey to the LLM exactly what each
piece of its output means.

## The architectural commitment

Same `(spec, binary)` → byte-identical BTOR2. No internal state,
no learned heuristics, no adaptivity in compilation. The translation
rules are spelled out in [`SCHEMA.md`](./SCHEMA.md), which is the
contract between rotor and any consumer of its output. An LLM (or
human, or future tool) reading the schema, the spec, and the binary
can in principle predict rotor's BTOR2 output exactly.

Anywhere rotor would otherwise make a heuristic choice, that choice
becomes either schema-documented and fixed, or a parameter the spec
specifies. There is no third option.

## What the LLM does

The LLM constructs questions as `QuestionSpec` values, decides which
solver to invoke and with what budget, interprets verdicts,
transfers learned facts across related questions, and runs whatever
refinement loops are appropriate. CEGAR is a pattern the LLM
implements by re-specifying; portfolios are something the LLM
constructs by dispatching in parallel; abstraction is something the
LLM requests via spec parameters.

This places real load on the LLM, but it is reasoning load — exactly
what the LLM is positioned to do. Rotor's role is to give the LLM a
substrate that is fully transparent, fully predictable, and richly
self-describing.

## How it conveys meaning

Rotor splits its BTOR2 output into hierarchical layers — header,
machine, library, dispatch, init, constraint, bad, binding, havoc —
each with its own stability profile and content hash. Layers are
linked by symbolic name and flattened to standard BTOR2 for
solvers, so the hierarchy is rotor-internal and the solver sees a
normal model.

Alongside compilation, rotor emits a structured annotation sidecar
recording, for every BTOR2 node it produces: the role (state,
instruction, init clause, learned invariant, …), the source mapping
(which RISC-V instruction at which PC, which ELF segment byte, which
DWARF line), and the provenance (rotor library, user spec, learned
from prior question N). The annotation is rotor's mechanism for
"telling the LLM what the model means" without rotor having to
interpret on the LLM's behalf.

## The LLM-facing surface

Five tools, mechanical semantics:

- `describe(topic)` — schema-on-demand
- `compile(spec)` — `(spec, binary)` → layered BTOR2 + annotation
- `dispatch(artifact, directive)` — run a single solver, return
  raw verdict
- `lift(artifact, raw)` — map solver output to source-grounded facts
- `introspect(artifact, query)` — read-only annotation lookup

Anything richer is composed from these primitives in the LLM's own
logic.

## What rotor supports

- RV64I + M + C instruction set
- ELF binaries with optional DWARF
- BMC and PDR/IC3 backends: Z3 (BMC and Spacer) by default;
  Bitwuzla, cvc5, Pono optionally
- Single-core analysis (state declarations are parameterized on core
  count for future multi-core)
- Reachability, safety properties, and synthesis (the latter
  expressed as unsatisfiability of a negated property)

## What rotor does not do

- Decide what to verify
- Choose solvers, bounds, or timeouts
- Run CEGAR or other refinement loops automatically
- Race solvers in a portfolio
- Slice for liveness or perform any optimization not explicitly
  requested via spec
- Validate that an invariant from one question applies to another
- Propose follow-up questions

All of these are LLM responsibilities and are excluded from rotor's
codebase as a matter of architectural principle.

## Status

Pre-implementation. The plan is in [`PLAN.md`](./PLAN.md), structured
into 12 phases (~14 working days). The schema document
[`SCHEMA.md`](./SCHEMA.md) is the contract and is written in Phase 4,
before the corresponding code lands.

## Reading order

1. This file — what rotor is
2. [`PLAN.md`](./PLAN.md) — how it gets built, phase by phase, with
   tests and exit criteria
3. [`SCHEMA.md`](./SCHEMA.md) — the translation contract; the
   authoritative reference once rotor is running

## License

MIT.