# hurdy-gurdy

A platform for building deterministic translations from source languages
to reasoning languages, so that LLMs can reason about programs through
external solvers.

A *pair* in hurdy-gurdy is a fixed combination of a source language and
a reasoning language — for example, RISC-V and BTOR2 — with a documented
translation between them. The first pair is `riscv-btor2`; `python-smtlib`
is planned as the second. Each pair is independent; pairs share the
framework but not their semantics.

The project ships as the `hurdy-gurdy` package on PyPI and exposes a
`gurdy` command and a `gurdy` Python module for daily use.

## What hurdy-gurdy does

Hurdy-gurdy compiles `(QuestionSpec, source program)` to a reasoning
artifact plus a structured semantic annotation, dispatches the artifact
to external solvers, and lifts solver outputs back to source-level
facts. That is the entire scope.

Hurdy-gurdy itself does no reasoning. It does not decide what to verify,
choose solvers, refine abstractions, run CEGAR loops, or compose
invariants across questions. Those are the LLM's job. Hurdy-gurdy's job
is to translate, dispatch, and lift — mechanically and deterministically —
and to convey to the LLM exactly what each piece of its output means.

## About the name

A hurdy-gurdy is a string instrument with a mechanical wheel the player
cranks; the wheel sounds multiple strings — paired as drone and melody —
and a keyboard of tangents presses against the melody strings to
determine pitch. The player chooses what to play and which keys to
press; the mechanism deterministically translates those inputs into
sound.

The project's architecture maps the instrument closely:

- A *pair* — drone+melody on the instrument, source+reasoning here — is
  the unit that produces meaningful output.
- The *schema* is the keyboard: a fixed, deterministic mapping from
  inputs to outputs. Same key, same pitch.
- *Compilation* is the wheel: the mechanical step that actually produces
  the artifact.
- The *framework* is the instrument's body: it hosts the strings and
  supports the resonance, but doesn't decide what's played.
- The *LLM* is the player: it thinks musically (what to verify, in what
  sequence, with what learned facts) while the instrument handles the
  mechanics.

A hurdy-gurdy can have several drone+melody pairings tuned together;
the player switches between them. Hurdy-gurdy (the project) hosts
several language pairs the same way: `riscv-btor2` first, `python-smtlib`
next, with the framework body unchanged.

## The architectural commitment

Same `(spec, source)` → byte-identical reasoning artifact, for every
pair. No internal state, no learned heuristics, no adaptivity in
compilation. Each pair's translation rules are spelled out in its own
`SCHEMA.md`, the contract between hurdy-gurdy and any consumer of that
pair's output. An LLM (or human) reading the schema, the spec, and the
source can in principle predict the output exactly.

Anywhere hurdy-gurdy would otherwise make a heuristic choice, that
choice becomes either schema-documented and fixed, or a parameter the
spec specifies. There is no third option.

## Pairs, not an intermediate representation

Hurdy-gurdy deliberately has no intermediate representation. Each pair
is a direct translation from one source language to one reasoning
language, governed by one schema. An IR would add a second schema,
complicate auditability, and force a forecast about future pairs'
expressive needs that we don't have evidence to make. Most cross-products
of source and reasoning languages aren't actually wanted — RISC-V wants
bitvector reasoning, Python wants theory-rich reasoning, and the
natural pairings are few.

What *is* shared across pairs is the framework: annotation, caching,
layer linking, dispatch, provenance, the LLM-facing tool surface. These
are mechanical, language-agnostic, and identical in structure across
pairs. Pairs plug into the framework; they don't share semantics with
each other.

If, after several pairs exist, common semantic structure emerges that
would benefit from abstraction, it can be added with empirical grounding.
We don't pre-commit.

## How hurdy-gurdy conveys meaning

For each pair, hurdy-gurdy splits its output into hierarchical *layers* —
named, individually-addressable pieces of the reasoning artifact, each
with its own stability profile and content hash. The `riscv-btor2`
pair's layers are header, machine, library, dispatch, init, constraint,
bad, binding, havoc; another pair declares its own. Layers are linked
by symbolic name and flattened to standard reasoning-language syntax
for solvers, so the hierarchy is internal and the solver sees a normal
artifact.

Alongside compilation, hurdy-gurdy emits a structured *annotation
sidecar* recording, for every node in the artifact: the role (state,
transition, init clause, learned invariant, …), the source mapping
(which source construct it was translated from), and the provenance
(which schema version, which spec, whether it came from a learned fact
carried forward from a prior question). The annotation is the mechanism
for telling the LLM what the artifact means without the framework
having to interpret on the LLM's behalf.

## What the LLM does

The LLM constructs questions as `QuestionSpec` values, decides which
solver to invoke and with what budget, interprets verdicts, transfers
learned facts across related questions, and runs whatever refinement
loops are appropriate. CEGAR is a pattern the LLM implements by
re-specifying. Portfolios are something the LLM constructs by
dispatching in parallel. Abstraction is something the LLM requests via
spec parameters.

This places real load on the LLM, but it is reasoning load — exactly
what the LLM is positioned to do. Hurdy-gurdy's role is to give the LLM
a substrate that is fully transparent, fully predictable, and richly
self-describing.

## The LLM-facing surface

Five tools, mechanical semantics, the same across all pairs:

- `describe(topic, pair)` — schema-on-demand
- `compile(spec)` — `(spec, source)` → layered artifact + annotation
- `dispatch(artifact, directive)` — run a single solver, return raw verdict
- `lift(artifact, raw)` — map solver output to source-grounded facts
- `introspect(artifact, query)` — read-only annotation lookup

Anything richer is composed from these primitives in the LLM's own
logic.

## Framework and pairs

The code is split between a language-agnostic *core* (the framework)
and one or more *pairs* (the language-specific plugins).

The core handles: the LLM tool surface and CLI, the spec validation
framework, the annotation sidecar machinery, layer declaration and
linking, solver dispatch wrappers, content-addressed caching of
compilation artifacts, structured diagnostics, and the schema-document
indexer that powers `describe`.

A pair contributes: a source loader, a `SCHEMA.md` documenting the
translation, a spec vocabulary (the source-language-specific observable,
assumption, and property types), a translator implementing the schema,
a lifter mapping solver output back to source-level facts, and solver
wrappers for solvers compatible with the pair's reasoning language.

The interface between core and pair is a single `Pair` protocol object.
Adding a new pair is mostly the work of writing the schema and the
translation; the rest is inherited from the framework.

## Pairs

### `riscv-btor2` (first pair)

- RV64I + M + C
- ELF binaries with optional DWARF
- BMC and PDR backends: Z3 (BMC and Spacer) by default; Bitwuzla, cvc5,
  Pono optionally
- Single-core analysis (state declarations parameterized on core count
  for future multi-core)
- Reachability, safety properties, and synthesis (the latter expressed
  as unsatisfiability of a negated property)

### `python-smtlib` (planned second pair)

A defined Python subset compiled to SMT-LIB. The subset and the
reasoning style are TBD — this pair exists in the plan as the second
instance that validates the framework's thinness.

## What hurdy-gurdy does not do

- Decide what to verify
- Choose solvers, bounds, or timeouts
- Run CEGAR or other refinement loops automatically
- Race solvers in a portfolio
- Slice or optimize artifacts beyond what the spec requests
- Validate that an invariant from one question applies to another
- Propose follow-up questions
- Maintain an intermediate representation between source and reasoning

All of these are LLM responsibilities (or, in the IR case, deliberate
non-features) and are excluded from the codebase as a matter of
architectural principle.

## Status

Pre-implementation. The plan is in [`PLAN.md`](./PLAN.md), structured
as framework phases followed by `riscv-btor2` pair phases. Each pair's
schema is written in its own phase, before the corresponding translation
code lands.

## Reading order

1. This file — what hurdy-gurdy is
2. [`PLAN.md`](./PLAN.md) — how it gets built, framework before pairs
3. `gurdy/pairs/riscv_btor2/SCHEMA.md` — the first pair's translation
   contract (written in its phase)

## Lineage

Hurdy-gurdy descends from rotor, originally developed as part of selfie
([`github.com/cksystemsteaching/selfie/tools/rotor.c`](https://github.com/cksystemsteaching/selfie/blob/main/tools/rotor.c)).
The RISC-V-to-BTOR2 encoding choices in the `riscv-btor2` pair draw on
rotor's design, and specific schema decisions may cite rotor as
historical context.

Hurdy-gurdy is not a port of rotor; it generalizes the architecture in
several ways that are deliberate departures: the framework/pair
separation hosts multiple language pairs rather than one fixed
translation; reasoning lives entirely in the LLM rather than in
built-in policies (no CEGAR loop, no portfolio dispatch, no automatic
slicing); the per-pair schema is the authoritative contract rather
than the C source. Implementation guidance flows from the schema and
the framework protocols, not from rotor's source code.

## License

MIT.