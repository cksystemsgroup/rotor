"""rIC3 subprocess bridge.

rIC3 is a Rust-native PDR/IC3 model checker that consistently wins
the BV tracks at HWMCC. Rotor consumes it as an external process:
write the BTOR2 Model to a temp file, invoke `rIC3 <file>`, parse
the verdict.

Positioning (Track A.2, ROADMAP.md). Spacer handles many bitvector
programs but struggles on frame-heavy loops; rIC3 is the engine
most likely to close rotor's `bounded_counter` fixture within a
reasonable time budget. When it is available, the portfolio races
it against Z3Spacer (in-process PDR) and Bitwuzla/Z3 BMC; whichever
lands the verdict first wins.

Install note. rIC3 requires nightly Rust:

    rustup install nightly
    cargo +nightly install rIC3 --locked

When the binary isn't on PATH the backend returns `unknown` with a
clear reason, so the portfolio can still race the other engines
without hard-failing.

Output format. rIC3 writes `true` / `false` on stdout for safe /
reachable, plus an optional invariant block and counterexample
witness. We parse these conservatively — structural witness
extraction is future work, and this bridge currently reports
`reachable` without a step/initial-regs payload.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from typing import Optional

from rotor.btor2.nodes import Model
from rotor.btor2.printer import to_text
from rotor.solvers.base import SolverResult


class Ric3:
    """Subprocess bridge to the rIC3 Rust PDR model checker."""
    name = "rIC3"

    def __init__(self, binary: str = "rIC3", extra_args: Optional[list[str]] = None) -> None:
        self._binary = binary
        self._extra_args = list(extra_args) if extra_args else []

    @property
    def available(self) -> bool:
        return shutil.which(self._binary) is not None

    def check_reach(
        self,
        model: Model,
        bound: int = 0,                          # unbounded engine; bound ignored
        timeout: Optional[float] = None,
    ) -> SolverResult:
        start = time.time()
        if not self.available:
            return SolverResult(
                verdict="unknown",
                bound=0,
                backend=self.name,
                reason=f"{self._binary} binary not found in PATH",
                elapsed=time.time() - start,
            )

        with tempfile.NamedTemporaryFile(
            suffix=".btor2", mode="w", delete=False, encoding="utf-8",
        ) as tf:
            tf.write(to_text(model))
            btor2_path = tf.name

        try:
            try:
                proc = subprocess.run(
                    [self._binary, *self._extra_args, btor2_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return SolverResult(
                    verdict="unknown",
                    bound=0,
                    backend=self.name,
                    reason=f"timeout after {timeout}s",
                    elapsed=time.time() - start,
                )
        finally:
            os.unlink(btor2_path)

        elapsed = time.time() - start
        return _parse_ric3(proc.stdout, proc.stderr, proc.returncode, self.name, elapsed)


def _parse_ric3(
    stdout: str, stderr: str, returncode: int, name: str, elapsed: float,
) -> SolverResult:
    """Classify rIC3's output into a SolverResult.

    rIC3 prints its verdict on the last non-empty stdout line. Exact
    tokens observed across versions: `true` / `safe` / `unsat` for
    safety, `false` / `unsafe` / `sat` for reachability, and a
    permissive `unknown` when it gives up.

    Invariant and witness extraction are conservative — we expose
    the full tool output as the invariant string on `proved` (so
    users can inspect it manually) and skip witness parsing on
    `reachable`. Better extraction lands when the witness shape
    stabilizes across rIC3 versions.
    """
    combined = (stdout or "") + "\n" + (stderr or "")
    last_words = [ln.strip().lower() for ln in (stdout or "").splitlines() if ln.strip()]

    safe_tokens = {"true", "safe", "unsat", "proved"}
    bug_tokens = {"false", "unsafe", "sat", "reachable"}

    def _ends_with_any(words: list[str], tokens: set[str]) -> bool:
        return bool(words) and words[-1] in tokens

    if _ends_with_any(last_words, safe_tokens):
        return SolverResult(
            verdict="proved",
            bound=0,
            backend=name,
            elapsed=elapsed,
            invariant=_extract_invariant(combined),
        )
    if _ends_with_any(last_words, bug_tokens):
        return SolverResult(
            verdict="reachable",
            bound=0,
            backend=name,
            elapsed=elapsed,
        )
    return SolverResult(
        verdict="unknown",
        bound=0,
        backend=name,
        elapsed=elapsed,
        reason=(
            f"rIC3 exit {returncode}; could not classify output "
            f"(head: {combined[:200]!r})"
        ),
    )


def _extract_invariant(output: str) -> Optional[str]:
    """Pull an invariant block from rIC3 output if one is present.

    rIC3 versions differ on where and how they print the inductive
    invariant; treat the whole tool output as the certificate text
    for now so the information isn't lost.
    """
    if not output or not output.strip():
        return None
    return output.strip()
