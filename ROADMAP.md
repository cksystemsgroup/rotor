# Roadmap: making rotor competitive before M9/M10

This document sequences the work needed to make rotor useful on
real binaries before diving into BVDDs (M9) and a Rust backend
(M10). The BVDD track optimizes a workload shape rotor cannot yet
produce on real compiled code; the tracks below close that gap
first.

`PLAN.md` remains the architectural roadmap. This file is the
execution plan layered on top of it.

---

## Framing: three axes of "competitive"

1. **Reach** — what real-world compiled binaries does rotor decode
   and lower to BTOR2 without bailing?
2. **Solver stack** — do the underlying engines close proofs on
   realistic workloads in realistic time?
3. **Verb coverage** — does rotor answer the questions users
   actually ask, or just `can_reach`?

Today, rotor is weak on all three. Closing the gap requires the
five tracks below.

---

## Track A — Solver stack diversity

**Blocker.** Z3 Spacer alone times out on any loop over bitvectors,
even after M8's register slicing. `bounded_counter` still fails to
close. This limits unbounded claims to trivial fixtures.

**Deliverable.** Three subprocess bridges under the existing
`SolverBackend` Protocol:

1. **Bitwuzla (BMC via SMT-LIB).** Typically 5–50× faster than Z3
   on pure-BV workloads. Reference engine for BV competitions.
   Plumbing: BTOR2 → SMT-LIB conversion, subprocess, sat/unsat
   parsing, model extraction for the `reachable` case.
2. **BtorMC (native BTOR2 BMC + k-induction).** Consumes rotor's
   BTOR2 emitter output directly, no format conversion. Part of
   the Boolector/Bitwuzla family.
3. **rIC3 (Rust IC3).** Consistently wins BV tracks at HWMCC.
   This is the engine most likely to close `bounded_counter`.

**Exit criterion.** Portfolio with all four engines closes every
`counter.elf` fixture within 30s, including `bounded_counter`.

**Effort estimate.** ~2 weeks. rIC3 requires `cargo install`;
others may need package installs.

---

## Track B — Full RV64I+M+C ISA

**Blocker.** rotor decodes RV64I base only. GCC/clang with default
flags emit RVC (compressed) and RV64M (mul/div); any real binary
fails on `decode() is None`.

**Deliverable.**

1. **RV64M** — `mul`, `mulh`, `mulhsu`, `mulhu`, `div`, `divu`,
   `rem`, `remu`, plus the `-w` 32-bit variants. Decoder + ISA
   lowering + witness + per-instruction tests. About a day.
2. **RVC** — variable-length encoding (16-bit / 32-bit mixed)
   requires reshaping the instruction-stream scanner. ~30
   compressed mnemonics expand to RV64I equivalents. About a
   week — decoder refactor is the hard part; semantics reuse
   existing RV64I lowerings.
3. **ecall/ebreak stubs** — halt the machine cleanly. Full
   syscall modeling is later.

**Exit criterion.** Rotor decodes `gcc -O2` of a small CLI program
end-to-end. Fixture: a non-trivial binary with a known
reachability property.

**Effort estimate.** ~1.5 weeks.

---

## Track C — Non-leaf function support

**Blocker.** Every interesting function calls others. Rotor's
current encoding treats the PC range as closed; `jal` outside the
function is modeled as "exits the function" with pc stuck.
Properties that span calls are unreachable from the analyzer.

**Deliverable.**

1. **`EntryAssumptions` object** generalizes Phase 6's
   `ra`-constraint. Models a realistic call frame: `sp` in a
   stack region, `ra` outside the analyzed set, `a0..a7` as
   arguments, callee-saved registers preserved per ABI.
2. **Call-graph scoping** — two modes: inline (expand callee's
   model into caller) and abstract (callee as havoc on
   callee-clobbered regs plus its memory footprint).
3. **Multi-function `build_reach`** — compile over a set of
   functions, not a single one.
4. **Scope-aware `ra`-outside-fn constraint** — `ra` points
   outside the *analyzed set*, not just the current function.

**Exit criterion.** A fixture with a non-leaf function whose
property depends on the callee's return value is `can_reach`-able,
and `cegar_reach` still works end-to-end.

**Effort estimate.** ~2 weeks. Design-heavy; many architectural
seams touched.

---

## Track D — Complete verb set (M3b)

**Blocker.** Rotor advertises four verbs; one ships. Users who
want anything beyond reachability hit a wall.

**Deliverable.**

1. **`verify(function, predicate)`** — user supplies a Python
   expression over live variables
   (`lambda regs, mem: regs['a0'] >= 0`); rotor compiles to
   `bad = !predicate`. Unbounded mode routes to Spacer, returns
   invariant or CEX.
2. **`find_input(function, output_condition)`** — `bad =
   output_condition` under BMC, return witness as SARIF / JSON /
   trace.
3. **`are_equivalent(other_binary, function)`** — product
   construction: two copies of the machine sharing inputs,
   `bad = output_differs`. Closes the binary-level
   refactoring-validation story.

**Exit criterion.** Each verb has a fixture, CLI subcommand, API
method, and integration test. README Quick Start gets one example
per verb.

**Effort estimate.** ~3 weeks total, parallelizable. `verify`
reuses `can_reach(unbounded=True)`; the other two need new
`QuestionSpec` types.

---

## Track E — HWMCC benchmark corpus + positioning

**Blocker.** Rotor's performance is self-reported on three
fixtures. No external comparison exists.

**Deliverable.**

1. **HWMCC BV benchmark ingestion.** The BTOR2 parser (M7.5)
   already handles the format. Add a corpus directory of HWMCC
   single-property BV benchmarks (~100–200 entries).
2. **Portfolio shootout.** Run Z3BMC / Z3Spacer / Bitwuzla /
   BtorMC / rIC3 each alone, and the portfolio of all. Record
   time-to-verdict per benchmark.
3. **`BENCHMARKS.md`.** Summary table: median speedup per engine,
   PAR-2 scores, which benchmarks each engine uniquely solves.
   Establishes rotor's solver-stack value proposition.

**Exit criterion.** Rotor's portfolio solves strictly more
benchmarks than any single engine. That is the positioning claim.

**Effort estimate.** ~2 weeks.

---

## Suggested sequencing

| Order | Track | Weeks | Unblocks |
|-------|-------|-------|----------|
| 1 | **A** solver stack | 2 | `bounded_counter` + loop invariants generally |
| 2 | **B** ISA coverage | 1.5 | real GCC/clang binaries |
| 3 | **D** verb set | 3 (parallel) | advertised functionality |
| 4 | **C** non-leaf | 2 | everything interesting in real code |
| 5 | **E** benchmarks | 2 | external credibility |

**Total: ~10 focused weeks** before M9/M10 becomes productive.

**Rationale for order.** A first: highest multiplier on everything
else — every other track benefits from a stronger solver stack.
B next: cheapest gate on real code. D and C can interleave
depending on whether "more verbs on toy code" or "reach on real
code" is the bigger pull at that moment. E last: has to follow
everything it's benchmarking.

After these five tracks, BVDD (M9) has real workloads worth
optimizing for — IC3 frames on real call graphs, equivalence
products from Track D, CEGAR abstractions operating on non-trivial
state spaces.

---

## Status tracking

Tracks are self-contained enough to ship sub-phases independently.
Each commit should cite which track + sub-phase it advances, the
same way Phase 6 commits cited `6.1`..`6.7`.

This roadmap is a living document — update as tracks complete or
scope shifts.
