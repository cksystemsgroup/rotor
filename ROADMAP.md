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

**Status.** Three backends added, two fully exercised in CI:

1. **Bitwuzla (`BitwuzlaBMC`).** ✅ **Shipped (A.1).** Uses
   Bitwuzla's in-process Python bindings (`pip install bitwuzla`).
   Measured on rotor fixtures:
   - `tiny_mask` bound=30: Z3 1.6s → Bitwuzla 1.2s (30% faster)
   - `bounded_counter` bound=30: Z3 26.4s → Bitwuzla 6.0s (4.4× faster)
2. **rIC3 (`Ric3`).** ⚠️ **Adapter shipped (A.2); live tests skip.**
   Subprocess bridge ready. Requires nightly Rust:
   `rustup install nightly && cargo +nightly install rIC3 --locked`.
   Parser covers the `true` / `false` / `sat` / `unsat` / `safe` /
   `unsafe` output tokens observed across rIC3 versions.
3. **BtorMC (`BtorMC`).** ⚠️ **Adapter shipped (A.3); live tests skip.**
   Subprocess bridge ready. Not packaged in Debian/Ubuntu apt;
   build from the Bitwuzla source tree (`./configure.py && ninja`).
   Supports both `bmc` and `kind` (k-induction) modes.

The subprocess bridges return `unknown` with a clear reason when the
external binary is not on PATH, so a portfolio containing them works
in environments where only a subset of the tools are installed.

**Exit criterion.** ✅ Fully met for A.1 (Bitwuzla). Met as code for
A.2/A.3 — live verification against `bounded_counter` depends on
installing the external tools, which the current dev sandbox can't
do (`rust-lang.org` unreachable from the build environment). Any
richer environment that has rIC3 or btormc on PATH will exercise
the bridges automatically via the skipped `@skipif` tests.

**Follow-ups.**
- Extend `rotor reach` CLI with an `--engine {z3bmc,bitwuzla,z3spacer,
  ric3,btormc}` flag for direct backend selection.
- Add a default portfolio that races every available engine
  (skipping the ones whose binaries are missing).
- Wire CEGAR to optionally delegate the abstract check to Bitwuzla
  (currently hardcoded to Z3Spacer).

---

## Track B — Full RV64I+M+C ISA

**Blocker.** rotor decodes RV64I base only. GCC/clang with default
flags emit RVC (compressed) and RV64M (mul/div); any real binary
fails on `decode() is None`.

**Status.**

1. **RV64M.** ✅ **Shipped (B.1).** `mul`, `mulh`, `mulhsu`, `mulhu`,
   `div`, `divu`, `rem`, `remu`, plus the `-w` 32-bit variants.
   Decoder + ISA lowering (with ITE wrappers for the RISC-V div/rem
   edge cases — div-by-zero, INT_MIN/-1 overflow) + witness
   simulator + disasm formatting + liveness + CEGAR register-read
   classification. Fixture: `mult.elf` built with `-march=rv64im`.
   Exercised across both Z3 and Bitwuzla backends via the
   L0-equivalence harness on SsaEmitter / DagEmitter / IdentityEmitter.
2. **ecall/ebreak stubs.** ✅ **Shipped (B.2).** SYSTEM opcode
   decoded for `ecall` (imm=0) and `ebreak` (imm=1); other SYSTEM
   instructions (CSR, sret, mret, wfi) deliberately return None
   until a privileged-mode model exists. Lowering: halt the machine
   cleanly by self-looping PC at the instruction's own address
   — no register writes, no memory writes. Honest semantics
   without a syscall model: any PC after a halt is correctly
   `unreachable` on that path.
3. **RVC (compressed).** ✅ **Shipped (B.3a + B.3b).** Full coverage
   of the RV64GC compressed subset minus floating-point:
   - B.3a threads `size` through `Instruction`, `Decoded`, `_fall`,
     and the witness simulator so the pipeline computes correct
     `pc + 2` fall-throughs on 16-bit instructions.
   - B.3b ships `rotor/btor2/riscv/rvc.py` with `expand_rvc(word16)`
     covering ~32 compressed mnemonics across all three quadrants,
     plus a variable-length scanner in `binary.py::instructions()`
     that detects compressed vs. full-width by the low-two bits and
     yields `Instruction(pc, expanded_word, size)`.
   Fixture: `rvc.elf` built with `-march=rv64imc`, containing
   mixed 2-byte and 4-byte instruction streams. Exercised by the
   L0-equivalence harness across IdentityEmitter / DagEmitter /
   SsaEmitter.
4. **Real gcc -O2 fixture.** ✅ **Shipped (B.4).** `bitops.elf`
   is a small bit-manipulation library built with
   `-march=rv64imc -O2`, combining all three Track B additions:
   RV64M arithmetic in `shifted_mul`, RVC compressed instructions
   throughout, branches and a loop in `popcount`. Exercised by
   the L0-equivalence harness across every emitter.

**Exit criterion.** ✅ Met. Rotor now decodes `gcc -O2
-march=rv64imc` output end-to-end: RV64I base, RV64M, RVC, and
ecall/ebreak stubs. Non-leaf support (real function calls) is
Track C's territory.

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
