"""BtorMC subprocess bridge.

BtorMC is the reference native BTOR2 BMC / k-induction engine from
the Boolector / Bitwuzla family. It consumes rotor's BTOR2 output
directly with no format conversion — the reason to add it as a
separate backend (versus calling Bitwuzla through its Python API) is
that BtorMC ships a k-induction driver that can answer `proved` on
sufficiently simple transition systems, while Bitwuzla-the-SMT-solver
only does pure BMC.

Positioning (Track A.3, ROADMAP.md). Portfolio races BtorMC alongside
Z3BMC, BitwuzlaBMC, and Z3Spacer; whichever engine closes the
obligation first wins.

Install note. BtorMC is not distributed as an apt package on typical
Linux distributions. Build from source:

    git clone https://github.com/Bitwuzla/bitwuzla
    cd bitwuzla && ./configure.py && cd build && ninja

Or use the historical boolector build. When the binary is not on
PATH the backend returns `unknown` with a clear reason, so the
portfolio can still race the other engines.

Modes. BtorMC supports `bmc` (bounded) and `kind` (k-induction)
modes. We default to `kind` since it subsumes BMC: it confirms
reachability within the bound and additionally answers `proved` if
k-induction closes. Override via the `mode` constructor arg.
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


class BtorMC:
    """Subprocess bridge to the BtorMC BTOR2 model checker."""
    name = "btormc"

    def __init__(
        self,
        binary: str = "btormc",
        mode: str = "kind",                      # "bmc" | "kind"
        extra_args: Optional[list[str]] = None,
    ) -> None:
        if mode not in ("bmc", "kind"):
            raise ValueError(f"mode must be 'bmc' or 'kind', got {mode!r}")
        self._binary = binary
        self._mode = mode
        self._extra_args = list(extra_args) if extra_args else []

    @property
    def available(self) -> bool:
        return shutil.which(self._binary) is not None

    def check_reach(
        self,
        model: Model,
        bound: int = 20,                         # used for bmc mode only
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

        argv = [self._binary]
        if self._mode == "kind":
            argv.append("--kind")
        else:
            argv.extend(["-kmax", str(bound)])
        argv.extend(self._extra_args)
        argv.append(btor2_path)

        try:
            try:
                proc = subprocess.run(
                    argv,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return SolverResult(
                    verdict="unknown",
                    bound=bound,
                    backend=self.name,
                    reason=f"timeout after {timeout}s",
                    elapsed=time.time() - start,
                )
        finally:
            os.unlink(btor2_path)

        elapsed = time.time() - start
        return _parse_btormc(
            proc.stdout, proc.stderr, proc.returncode, self.name, bound, elapsed,
        )


def _parse_btormc(
    stdout: str, stderr: str, returncode: int,
    name: str, bound: int, elapsed: float,
) -> SolverResult:
    """Classify BtorMC output into a SolverResult.

    BtorMC emits `sat` / `unsat` on the last stdout line in BMC mode.
    In kind mode it additionally prints `unknown` when the k-inductive
    frontier doesn't close. Witness extraction is conservative — we
    report the tool output as the invariant string on `proved`
    verdicts and skip initial-reg parsing on `reachable`.
    """
    combined = (stdout or "") + "\n" + (stderr or "")
    last_words = [ln.strip().lower() for ln in (stdout or "").splitlines() if ln.strip()]

    def _ends_with(words: list[str], token: str) -> bool:
        return bool(words) and words[-1] == token

    if _ends_with(last_words, "unsat"):
        # In kind mode, unsat = globally safe.
        return SolverResult(
            verdict="proved",
            bound=bound,
            backend=name,
            elapsed=elapsed,
            invariant=(combined.strip() or None),
        )
    if _ends_with(last_words, "sat"):
        return SolverResult(
            verdict="reachable",
            bound=bound,
            backend=name,
            elapsed=elapsed,
        )
    return SolverResult(
        verdict="unknown",
        bound=bound,
        backend=name,
        elapsed=elapsed,
        reason=(
            f"btormc exit {returncode}; could not classify output "
            f"(head: {combined[:200]!r})"
        ),
    )
