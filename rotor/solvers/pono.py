"""Pono subprocess bridge.

Pono (Stanford UPSCALE group) is a multi-engine SMT-based model
checker with native BTOR2 support. It bundles k-BMC, IC3IA
(interpolation-based IC3), mBIC3 (Bradley-style bit-level IC3),
plus k-induction and interpolation backends behind one CLI. Rotor
uses it as a subprocess: write BTOR2 to a temp file, invoke
`pono --engine MODE file.btor2`, parse the verdict.

Positioning. Pono is the direct replacement for the separate rIC3
and BtorMC adapters: one maintained codebase gives us both BMC
(`--engine bmc`) and IC3 (`--engine ic3ia` / `--engine mbic3`) at
once. Its QF_ABV handling is stronger than Z3 Spacer's — precisely
the `bounded_counter`-class workload where rotor has previously
been stuck. Adapter pattern mirrors `Ric3`/`BtorMC`: PATH probe +
graceful `unknown` fallback when the binary isn't installed.

Install. Build from source:

    git clone https://github.com/upscale-project/pono
    cd pono && ./contrib/setup-smt-switch.sh
    ./contrib/setup-btor2tools.sh
    mkdir build && cd build && cmake .. && make -j

The `pono` binary ends up in `build/`; add it to PATH.

Output format. Pono writes `sat` / `unsat` / `unknown` on stdout
(typically as the last non-empty line). Witness and invariant
blocks print between the engine log and the final verdict when
`--witness` / `--print-invar` are passed; this bridge enables
`--print-invar` by default so the invariant is captured on
`proved`. Witness parsing is best-effort — Pono's witness format
varies between versions.
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

# Engines Pono exposes via --engine. Names match Pono's CLI directly
# so debugging Pono-level behavior from rotor output is straightforward.
_SUPPORTED_MODES: frozenset[str] = frozenset({
    "bmc", "bmc-sp", "ind", "mbic3", "ic3ia", "ic3sa", "ic3bits",
    "interp", "sygus-pdr",
})

# Which modes are unbounded (`proved` possible) vs bounded (at best
# `unreachable` up to k). Determines how the parser classifies
# `unsat` — under bounded engines unsat means "safe up to k";
# under unbounded engines it means globally safe (proved).
_UNBOUNDED_MODES: frozenset[str] = frozenset({
    "ind", "mbic3", "ic3ia", "ic3sa", "ic3bits", "interp", "sygus-pdr",
})


class Pono:
    """Subprocess bridge to the Pono BTOR2 model checker."""
    name = "pono"

    def __init__(
        self,
        mode: str = "bmc",
        binary: str = "pono",
        extra_args: Optional[list[str]] = None,
    ) -> None:
        if mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_SUPPORTED_MODES)}, got {mode!r}"
            )
        self._mode = mode
        self._binary = binary
        self._extra_args = list(extra_args) if extra_args else []
        # Name includes the engine so portfolio result rows are
        # unambiguous when multiple Pono racers are in the same race.
        self.name = f"pono-{mode}"

    @property
    def available(self) -> bool:
        return shutil.which(self._binary) is not None

    def check_reach(
        self,
        model: Model,
        bound: int = 20,
        timeout: Optional[float] = None,
    ) -> SolverResult:
        start = time.time()
        if not self.available:
            return SolverResult(
                verdict="unknown",
                bound=bound,
                backend=self.name,
                reason=f"{self._binary} binary not found in PATH",
                elapsed=time.time() - start,
            )

        with tempfile.NamedTemporaryFile(
            suffix=".btor2", mode="w", delete=False, encoding="utf-8",
        ) as tf:
            tf.write(to_text(model))
            btor2_path = tf.name

        argv = [
            self._binary,
            "--engine", self._mode,
            "--bound", str(bound),
            "--print-invar",
            *self._extra_args,
            btor2_path,
        ]

        try:
            try:
                proc = subprocess.run(
                    argv, capture_output=True, text=True, timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return SolverResult(
                    verdict="unknown", bound=bound, backend=self.name,
                    reason=f"timeout after {timeout}s",
                    elapsed=time.time() - start,
                )
        finally:
            os.unlink(btor2_path)

        elapsed = time.time() - start
        return _parse_pono(
            proc.stdout, proc.stderr, proc.returncode,
            self.name, bound, elapsed, unbounded=self._mode in _UNBOUNDED_MODES,
        )


def _parse_pono(
    stdout: str, stderr: str, returncode: int,
    name: str, bound: int, elapsed: float, *, unbounded: bool,
) -> SolverResult:
    """Classify Pono's output into a SolverResult.

    Pono writes `sat` / `unsat` / `unknown` as the verdict, typically
    on the last non-empty stdout line. Some versions wrap it in
    additional context (`Property sat at step 3`, `proved safe`);
    we accept any of the common phrasings.

    On `unsat`, unbounded engines (IC3 variants, k-induction,
    interpolation) yield `proved`; bounded BMC yields `unreachable`.
    On `sat`, both yield `reachable`; witness extraction is
    best-effort and today only surfaces the raw CEX block as a
    `reason` string for user inspection.
    """
    combined = (stdout or "") + "\n" + (stderr or "")
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]

    last = lines[-1].lower() if lines else ""
    sat_tokens = {"sat", "unsafe", "false"}
    unsat_tokens = {"unsat", "safe", "true", "proved"}

    def _ends_with_token(s: str, tokens: set[str]) -> bool:
        words = s.split()
        return bool(words) and words[-1] in tokens

    if any(_ends_with_token(ln.lower(), sat_tokens) for ln in lines):
        return SolverResult(
            verdict="reachable", bound=bound, backend=name, elapsed=elapsed,
            reason=_extract_witness(combined),
        )
    if any(_ends_with_token(ln.lower(), unsat_tokens) for ln in lines):
        if unbounded:
            return SolverResult(
                verdict="proved", bound=0, backend=name, elapsed=elapsed,
                invariant=_extract_invariant(combined),
            )
        return SolverResult(
            verdict="unreachable", bound=bound, backend=name, elapsed=elapsed,
        )
    return SolverResult(
        verdict="unknown", bound=bound, backend=name, elapsed=elapsed,
        reason=(
            f"pono exit {returncode}; could not classify output "
            f"(head: {combined[:200]!r})"
        ),
    )


def _extract_invariant(output: str) -> Optional[str]:
    """Pull an `invariant:` or `--print-invar` block from Pono's
    output if present. Treats the entire stdout as the certificate
    text when a marker is absent; callers can inspect manually."""
    if not output or not output.strip():
        return None
    lowered = output.lower()
    for marker in ("invariant:", "inductive invariant"):
        idx = lowered.find(marker)
        if idx >= 0:
            return output[idx:].strip()
    return output.strip()


def _extract_witness(output: str) -> Optional[str]:
    """Same best-effort policy as _extract_invariant, for witness
    blocks after a `sat` verdict."""
    if not output or not output.strip():
        return None
    lowered = output.lower()
    for marker in ("witness", "counterexample", "trace:"):
        idx = lowered.find(marker)
        if idx >= 0:
            return output[idx:].strip()
    return None
