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

**Status.** ✅ **Shipped** (phases C.1–C.3 in one commit).

The key insight: Phase 6's ra constraint was a global invariant
— it required `ra ∉ fn.pc_range` at every cycle. Inside the
analyzed set, a `jal` legitimately writes an intra-set PC
(`pc + 4`) into ra during execution; the global form would
reject that trace. Track C moves the ra constraint to cycle-0
only by driving ra's init from a fresh `init_ra` input and
constraining the input rather than ra itself. Once that's done,
multi-function analysis composes naturally — the dispatch tree
just grows to cover all included functions and `jal` + `ret`
update pc like any other instruction.

**Delivered:**

1. ✅ **`EntryAssumptions` dataclass** (`rotor.btor2.builder`)
   — first-class object holding `excluded_pc_ranges`. Future
   sp / a0..a7 / callee-saved fields slot in alongside without
   another ra-style rewrite. `from_functions(binary, names)`
   constructor computes the union of PC ranges from a list.
2. ✅ **Multi-function `build_reach` / `build_verify` /
   `build_find_input`** via a new `include_fns` parameter.
   When provided, the listed function names are decoded into
   the same dispatch tree as the entry fn; the ra constraint
   widens to exclude the union; `_return_site_bad` filters
   ret-PC enumeration to the entry fn only (callee rets are
   not observation sites).
3. ✅ **Scope-aware ra constraint** — `_assume_ra_outside_ranges`
   builds an AND over each excluded range.
4. **Call-graph scoping** — explicit (user lists callees via
   `--include-fn`) is shipped. Automatic static discovery from
   `jal` immediates is a natural follow-up but not required
   for the exit criterion.

**Exposed at:**
- `api.can_reach(..., include_fns=[...])`
- `api.verify(..., include_fns=[...])`
- `api.find_input(..., include_fns=[...])`
- `rotor reach --include-fn CALLEE` (repeatable)
- Same flag on `rotor verify` / `rotor find-input`

**Exit criterion.** ✅ Met. `tests/fixtures/nonleaf.elf` is a
`double_square` function that makes a real `jal` to `square`.
Without `include_fns`, the ret is `unreachable` (PC stuck on
square.start); with `include_fns=["square"]`, the ret is
`reachable` in 8 steps, with a ra witness outside the analyzed
set and sp unrestricted.

**Not shipped (future):**
- Automatic callee discovery from static `jal` immediates.
- Shared-memory story for non-leaf equivalence.
- Unbounded (Spacer) path for `include_fns` — Spacer bypasses
  the emitter seam and doesn't thread include_fns today.

---

## Track D — Complete verb set (M3b)

**Blocker.** Rotor advertises four verbs; one ships. Users who
want anything beyond reachability hit a wall.

**Deliverable.**

1. **`verify(function, predicate)`** — ✅ **Shipped (D.1).**
   Register-comparison predicate DSL: `verify(function, register,
   comparison, rhs)` evaluates `regs[register] OP rhs` at every
   `ret` inside the function; rotor compiles to `bad = (pc at
   ret) ∧ ¬predicate`. Bounded by default, `unbounded=True`
   routes to Z3 Spacer. Exposed as `api.verify(...)` and
   `rotor verify` CLI subcommand. Reuses `ReachResult` (aliased
   as `VerifyResult`) since both verbs share the same verdict
   vocabulary. Richer predicates (lambdas, conjunctions,
   memory-region constraints) are future work.
2. **`find_input(function, output_condition)`** — ✅ **Shipped
   (D.2).** Synthesizes an initial-register assignment such that
   `regs[R] OP rhs` holds at a return site. Same spec shape as
   `verify` but with flipped polarity: `reachable` means
   "witness found, `initial_regs` is the synthesized input".
   Shared `_return_site_bad` helper with `verify` (one
   `negate: bool` flag is the only difference). Exposed as
   `api.find_input(...)` and `rotor find-input` CLI subcommand.
   Richer witness output (SARIF / JSON) is future work.
3. **`are_equivalent(other_binary, function)`** — ✅ **Shipped
   (D.3).** Product construction: two copies of the machine run
   in parallel inside one BTOR2 Model (state-prefixed `a_` /
   `b_`) with all 31 architectural registers init'd from a
   shared set of input nodes. Each side's output register is
   latched at its first return; `bad = (both returned) ∧
   (captured_a ≠ captured_b)`. Exposed as
   `api.are_equivalent(other_binary_path, function)` and
   `rotor equivalent` CLI subcommand. Scope: leaf functions
   only (no shared memory); BMC only. Unbounded equivalence
   via Spacer on the product is future work.

**Exit criterion.** ✅ Met. All three verbs (`verify`,
`find_input`, `are_equivalent`) have fixtures, CLI subcommands,
API methods, and integration tests. Together with `can_reach`
and `cegar_reach`, rotor's advertised verb surface is fully
shipped.

---

## Track E — Benchmark harness + positioning

**Status.** ✅ **Shipped.** Harness and seed corpus are in; the
HWMCC portion is on hold until a local checkout is available
(the sandbox that produced these numbers can't reach
hwmcc.github.io).

**Delivered:**

1. ✅ **`rotor.bench` module** — runs every backend against a
   corpus of BTOR2 benchmarks, records per-(benchmark, engine)
   verdicts and elapsed times, classifies outcomes as SOLVED /
   UNSOLVED, and computes PAR-2 aggregate scores (HWMCC
   convention: unsolved runs penalized as 2×timeout).
2. ✅ **`rotor benchmark` CLI** with two corpus sources:
   - `--fixtures` materializes rotor's L0-equivalence corpus
     as bench entries (44 benchmarks at bound=10).
   - `--btor2-dir PATH` loads every `*.btor2` file in a
     directory. Expected verdicts can ride along in `.expected`
     sidecar files; missing → accept any conclusive verdict.
     Use this with a local HWMCC BV-track checkout.
3. ✅ **`BENCHMARKS.md`** — auto-generated from a fixtures run.
   Headline from that run on the current sandbox (bound=10,
   timeout=15s):

   | Engine    | Solved  | PAR-2  |
   |-----------|---------|--------|
   | z3-bmc    | 44/44   | 87.1s  |
   | z3-spacer | 29/44   | 456.7s |
   | bitwuzla  | 44/44   | 0.6s   |
   | portfolio | 44/44   | 0.8s   |

   Bitwuzla dominates on the fixture corpus — expected, it's
   the HWMCC BV-track winner for a reason. Spacer times out on
   memory-heavy models (the array-theory ceiling rotor has
   tracked since Phase 6.3). The portfolio is within noise of
   the fastest single engine and is strictly safer: it catches
   whatever any racer catches.

   Unique-solve analysis on this corpus: none — every benchmark
   is within reach of at least two engines. An HWMCC-scale run
   is where the portfolio earns its keep via Spacer / rIC3 PDR
   decisions that no BMC engine can return.

**Not shipped here (future):**
- Live HWMCC corpus — requires local checkout of a ~GB-scale
  archive; the seed-corpus harness works the moment you point
  `--btor2-dir` at it.
- rIC3 / BtorMC rows in the shootout. Their subprocess bridges
  (Track A.2 / A.3) include `shutil.which` guards, so the
  shootout would pick them up automatically where the binaries
  are installed; the current sandbox has neither.
- Regression gating in CI: wire the harness into a nightly job
  that fails on PAR-2 regressions past a threshold.

**Exit criterion.** Met in spirit: rotor now has a reproducible
shootout harness, a first shipped `BENCHMARKS.md` demonstrating
the solver-stack story, and graceful fallback for engines /
corpora that aren't available locally.

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
