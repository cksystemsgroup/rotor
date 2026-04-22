"""Solver-stack shootout harness.

Runs every rotor backend against a benchmark corpus and reports
per-engine / per-benchmark outcomes plus aggregate metrics. The
output is designed for `BENCHMARKS.md` but is also accessible
programmatically for CI regression tracking.

Corpus entries are (name, model_factory, expected_verdict) triples.
`model_factory` is a zero-arg callable returning a fresh BTOR2
`Model` — factories are used so the parser / builder runs inside
the per-engine wall-clock measurement only for engines that
consume a fresh Model (Z3 needs a fresh context anyway; factories
keep the surface uniform).

Two corpus loaders ship:
  - `rotor_fixture_corpus()` materializes every entry in the
    L0-equivalence corpus as a bench entry. Requires only the
    files already in tests/fixtures.
  - `btor2_dir_corpus(path)` loads every `*.btor2` file in a
    directory. Use this with HWMCC single-property BV benchmarks
    when they're available locally.

Per-engine results are classified as SOLVED (verdict matches the
corpus expectation, or no expectation was given and the verdict is
not `unknown`) or UNSOLVED (timeout / unknown / mismatch). PAR-2
scores penalize unsolved runs with 2×timeout to compare engines on
a consistent scale, following HWMCC convention.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from rotor.btor2.nodes import Model
from rotor.solvers.base import SolverBackend, SolverResult
from rotor.solvers.portfolio import Portfolio


@dataclass(frozen=True)
class BenchEntry:
    name: str
    model_factory: Callable[[], Model]
    expected_verdict: Optional[str] = None           # None = no ground truth


@dataclass(frozen=True)
class RunOutcome:
    entry: str
    engine: str
    verdict: str
    elapsed: float
    solved: bool
    reason: Optional[str] = None


@dataclass
class ShootoutResult:
    entries: list[BenchEntry]
    engines: list[str]
    outcomes: list[RunOutcome] = field(default_factory=list)
    timeout: float = 30.0
    bound: int = 20

    def grouped(self) -> dict[tuple[str, str], RunOutcome]:
        """Index outcomes by (entry_name, engine_name)."""
        return {(o.entry, o.engine): o for o in self.outcomes}

    def par2(self, engine: str) -> float:
        """PAR-2 score: sum of elapsed times, with unsolved runs
        penalized as 2×timeout. Lower is better."""
        total = 0.0
        for o in self.outcomes:
            if o.engine != engine:
                continue
            total += o.elapsed if o.solved else 2.0 * self.timeout
        return total

    def solved_count(self, engine: str) -> int:
        return sum(1 for o in self.outcomes if o.engine == engine and o.solved)


# ---------------------------------------------------------------------------
# Shootout driver
# ---------------------------------------------------------------------------

EngineSpec = tuple[str, Callable[[], SolverBackend]]  # (name, factory)


def run_shootout(
    entries: Iterable[BenchEntry],
    engines: Iterable[EngineSpec],
    *,
    bound: int = 20,
    timeout: float = 30.0,
) -> ShootoutResult:
    """Run every engine against every entry. Each engine instance is
    fresh per run (avoids shared mutable solver state across
    concurrently-sensitive backends like Z3 contexts).

    The portfolio sees each entry once — it's treated as just
    another engine row. Users who want to compare "portfolio vs.
    individual engines" pass a Portfolio factory in `engines`.
    """
    entries = list(entries)
    engines = list(engines)
    result = ShootoutResult(
        entries=entries,
        engines=[e[0] for e in engines],
        bound=bound,
        timeout=timeout,
    )
    for entry in entries:
        for engine_name, engine_factory in engines:
            outcome = _run_one(entry, engine_name, engine_factory, bound, timeout)
            result.outcomes.append(outcome)
    return result


def _run_one(
    entry: BenchEntry,
    engine_name: str,
    engine_factory: Callable[[], SolverBackend],
    bound: int,
    timeout: float,
) -> RunOutcome:
    start = time.time()
    try:
        model = entry.model_factory()
        engine = engine_factory()
        # Portfolio's check_reach signature differs from SolverBackend —
        # it takes only the model. Handle both cases.
        if isinstance(engine, Portfolio):
            sr: SolverResult = engine.check_reach(model)
        else:
            sr = engine.check_reach(model, bound=bound, timeout=timeout)
    except Exception as exc:                         # pragma: no cover
        elapsed = time.time() - start
        return RunOutcome(
            entry=entry.name, engine=engine_name,
            verdict="error", elapsed=elapsed,
            solved=False, reason=f"{type(exc).__name__}: {exc}",
        )

    solved = _is_solved(sr.verdict, entry.expected_verdict)
    return RunOutcome(
        entry=entry.name, engine=engine_name,
        verdict=sr.verdict, elapsed=sr.elapsed,
        solved=solved, reason=sr.reason,
    )


def _is_solved(actual: str, expected: Optional[str]) -> bool:
    if actual == "unknown" or actual == "error":
        return False
    if expected is None:
        return True                                  # any conclusive verdict counts
    # Allow `proved` (global) to count when the expectation was the
    # weaker `unreachable` — unbounded engines produce strictly
    # stronger verdicts than BMC on the same safe fixture.
    if expected == "unreachable" and actual == "proved":
        return True
    return actual == expected


# ---------------------------------------------------------------------------
# Corpus loaders
# ---------------------------------------------------------------------------

def rotor_fixture_corpus() -> list[BenchEntry]:
    """Convert rotor's L0-equivalence corpus into bench entries.

    Each (binary, function, target_offset) triple becomes a
    benchmark whose factory opens the binary, compiles the
    ReachSpec through `IdentityEmitter`, and returns the resulting
    Model. Binaries are closed at factory end so repeated runs
    don't leak ELF handles.

    No `expected_verdict` is carried: the corpus's expected
    verdicts are bound-specific (they assume each entry's native
    bound), but the shootout runs every benchmark at one uniform
    bound. Any conclusive verdict — reachable, unreachable, or
    proved — counts as SOLVED.
    """
    from pathlib import Path
    from rotor.binary import RISCVBinary
    from rotor.ir.emitter import IdentityEmitter
    from rotor.ir.spec import ReachSpec
    from tests.equivalence.corpus import CORPUS

    repo_root = Path(__file__).resolve().parents[1]
    entries: list[BenchEntry] = []
    for c in CORPUS:
        def factory(c=c, repo_root=repo_root):
            path = repo_root / c.binary_relpath
            with RISCVBinary(path) as b:
                fn = b.function(c.function)
                return IdentityEmitter(b).emit(
                    ReachSpec(function=c.function, target_pc=fn.start + c.target_offset)
                )
        entries.append(BenchEntry(name=c.name, model_factory=factory))

    # Hand-picked adversarial entries — safety claims BMC cannot
    # prove at any finite bound but Spacer can close with an
    # inductive invariant. These exercise the solver-stack
    # complementarity: BMC is fast on SAT, Spacer is uniquely
    # strong on UNSAT.
    from pathlib import Path
    counter_path = repo_root / "tests/fixtures/counter.elf"
    if counter_path.exists():
        def tiny_mask_dead_factory():
            with RISCVBinary(counter_path) as b:
                return IdentityEmitter(b).emit(
                    ReachSpec(function="tiny_mask", target_pc=0x1117c)
                )
        entries.append(BenchEntry(
            name="tiny_mask-dead-branch-SAFE",
            model_factory=tiny_mask_dead_factory,
        ))

    return entries


def btor2_dir_corpus(directory) -> list[BenchEntry]:
    """Load every `*.btor2` file in `directory` as a bench entry.

    Expected verdicts are read from an optional `.expected` file
    per benchmark (one word: `reachable` / `unreachable` / `proved`);
    missing files mean "no ground truth", and any conclusive
    verdict will count as solved.
    """
    from pathlib import Path
    from rotor.btor2.parser import from_path

    directory = Path(directory)
    entries: list[BenchEntry] = []
    for btor2_path in sorted(directory.glob("*.btor2")):
        expected_file = btor2_path.with_suffix(".expected")
        expected = expected_file.read_text().strip() if expected_file.exists() else None

        def factory(p=btor2_path):
            r = from_path(p)
            if not r.ok:
                raise ValueError(f"parse failed: {r.diagnostics}")
            return r.model

        entries.append(BenchEntry(
            name=btor2_path.stem, model_factory=factory,
            expected_verdict=expected,
        ))
    return entries


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def format_markdown(result: ShootoutResult) -> str:
    """Render a shootout result as a Markdown report with a
    per-entry table and aggregate PAR-2 scores.
    """
    lines: list[str] = []
    lines.append("# Solver shootout")
    lines.append("")
    lines.append(f"Corpus: {len(result.entries)} benchmarks, "
                 f"bound={result.bound}, timeout={result.timeout:.0f}s.")
    lines.append("")
    lines.append("Cell format: `VERDICT TIME`. Verdicts are **BUG** "
                 "(bad state reached, SAT), **SAFE** (bad unreachable at "
                 "this bound, UNSAT), **PROVED** (inductive invariant "
                 "found — safe for all bounds), or **—** (solver "
                 "returned unknown / timed out). Parenthesized elapsed "
                 "times mark runs classified UNSOLVED.")
    lines.append("")

    # Per-entry table.
    grouped = result.grouped()
    header = "| Benchmark | " + " | ".join(result.engines) + " |"
    sep = "|" + "---|" * (1 + len(result.engines))
    lines.append(header)
    lines.append(sep)
    for entry in result.entries:
        row = [entry.name]
        for engine in result.engines:
            o = grouped.get((entry.name, engine))
            row.append(_cell(o) if o else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Aggregate (lower PAR-2 is better)")
    lines.append("")
    lines.append("| Engine | Solved | PAR-2 (s) |")
    lines.append("|---|---|---|")
    for engine in result.engines:
        solved = result.solved_count(engine)
        par2 = result.par2(engine)
        lines.append(f"| {engine} | {solved}/{len(result.entries)} | {par2:.1f} |")

    # Unique-solve analysis.
    lines.append("")
    lines.append("## Unique solves")
    lines.append("")
    unique_any = False
    for engine in result.engines:
        uniquely = []
        for entry in result.entries:
            o = grouped.get((entry.name, engine))
            if o is None or not o.solved:
                continue
            others = [
                grouped.get((entry.name, e))
                for e in result.engines if e != engine
            ]
            if not any(oo and oo.solved for oo in others):
                uniquely.append(entry.name)
        if uniquely:
            unique_any = True
            lines.append(f"- **{engine}**: {', '.join(uniquely)}")
    if not unique_any:
        lines.append("_No engine has unique solves on this corpus._")

    lines.append("")
    return "\n".join(lines)


def _cell(o: RunOutcome) -> str:
    """Compact per-cell rendering: verdict + elapsed or timeout marker."""
    label = {
        "reachable":   "BUG",
        "unreachable": "SAFE",
        "proved":      "PROVED",
        "unknown":     "—",
        "error":       "ERR",
    }.get(o.verdict, o.verdict)
    if not o.solved:
        return f"{label} ({o.elapsed:.1f}s)"
    ms = o.elapsed * 1000
    if ms < 1000:
        return f"{label} {ms:.0f}ms"
    return f"{label} {o.elapsed:.2f}s"
