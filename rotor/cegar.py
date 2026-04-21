"""Counterexample-guided abstraction refinement for reachability.

Wraps `build_reach` + Z3Spacer + the concrete witness simulator into
an iterative loop that:

    1. Starts with every register havoc'd — the cheapest abstraction.
    2. Runs Z3Spacer on the abstract model.
    3. If Spacer says `proved`, the abstraction is safe and the
       concrete model is also safe (sound over-approximation).
    4. If Spacer says `reachable`, extracts a counterexample via
       Z3BMC on the same abstract model and replays it concretely
       through `rotor.witness.simulate`.
    5. If the concrete replay hits `target_pc`, the CEX is real — the
       concrete model is reachable.
    6. Otherwise the CEX is spurious — refine by unhavoc'ing every
       register read by the concrete replay path, then loop.

Termination: each iteration strictly shrinks the havoc set (a
finite set of register indices). In the worst case CEGAR converges
to the fully concrete model, at which point it returns whatever the
underlying engines return. A `max_iterations` cap is enforced for
when Spacer keeps flipping `unknown` or when the solver can't produce
a useful CEX.

Refinement strategy is intentionally simple — unhavoc all registers
read along the concrete path — so CEGAR converges quickly on small
fixtures. Smarter predicate-abstraction strategies (Houdini, learned
predicates) are future work; PLAN.md's Open Questions section flags
this explicitly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from rotor.binary import RISCVBinary
from rotor.btor2.builder import build_reach
from rotor.btor2.riscv.decoder import Decoded
from rotor.ir.spec import ReachSpec
from rotor.solvers.base import SolverResult
from rotor.solvers.z3bv import Z3BMC
from rotor.solvers.z3spacer import Z3Spacer
from rotor.witness import simulate

NREGS = 32


# Register-read classification per mnemonic. Used to build the set of
# registers to unhavoc after a spurious CEX. Over-approximation is OK
# — unhavoc'ing an unnecessary register only costs solver time, never
# correctness. Under-approximation would loop CEGAR forever.

_READS_RS1_AND_RS2 = frozenset({
    # R-type arithmetic
    "add", "sub", "and", "or", "xor", "slt", "sltu", "sll", "srl", "sra",
    "addw", "subw", "sllw", "srlw", "sraw",
    # Branches
    "beq", "bne", "blt", "bge", "bltu", "bgeu",
    # Stores (rs1 = base, rs2 = value)
    "sb", "sh", "sw", "sd",
})

_READS_RS1_ONLY = frozenset({
    # I-type arithmetic / shifts
    "addi", "slti", "sltiu", "xori", "ori", "andi",
    "slli", "srli", "srai",
    "addiw", "slliw", "srliw", "sraiw",
    # Indirect jump
    "jalr",
    # Loads
    "lb", "lh", "lw", "ld", "lbu", "lhu", "lwu",
})


def _regs_read_by(d: Decoded) -> set[int]:
    s: set[int] = set()
    if d.mnem in _READS_RS1_AND_RS2:
        s.add(d.rs1)
        s.add(d.rs2)
    elif d.mnem in _READS_RS1_ONLY:
        s.add(d.rs1)
    # lui / auipc / jal / fence: no register reads.
    s.discard(0)
    return s


@dataclass(frozen=True)
class CegarConfig:
    max_iterations: int = 16
    spacer_timeout: float = 30.0
    bmc_bound: int = 40                   # for extracting CEX from abstract model
    bmc_timeout: float = 15.0
    simulate_max_steps: int = 100         # concrete replay depth


@dataclass(frozen=True)
class CegarHistory:
    """Per-iteration diagnostic: which verdict, how many regs havoc'd."""
    iteration: int
    havoc_count: int
    spacer_verdict: str
    spacer_elapsed: float


def cegar_reach(
    binary: RISCVBinary,
    spec: ReachSpec,
    config: Optional[CegarConfig] = None,
) -> SolverResult:
    """CEGAR-wrapped reachability check.

    Returns a `SolverResult` whose `backend` field reports how many
    CEGAR iterations were consumed. `invariant` is populated when
    the final Spacer verdict was `proved`.
    """
    cfg = config or CegarConfig()
    fn = binary.function(spec.function)
    havoc: set[int] = set(range(1, NREGS))
    history: list[CegarHistory] = []
    start = time.time()

    for iteration in range(cfg.max_iterations):
        model = build_reach(binary, spec, havoc_regs=havoc)
        r_abs = Z3Spacer().check_reach(model, bound=0, timeout=cfg.spacer_timeout)
        history.append(CegarHistory(
            iteration=iteration,
            havoc_count=len(havoc),
            spacer_verdict=r_abs.verdict,
            spacer_elapsed=r_abs.elapsed,
        ))

        if r_abs.verdict == "proved":
            return SolverResult(
                verdict="proved",
                bound=0,
                elapsed=time.time() - start,
                backend=f"cegar({iteration + 1}it,{len(havoc)}havoc)",
                invariant=r_abs.invariant,
            )

        if r_abs.verdict == "unknown":
            return SolverResult(
                verdict="unknown",
                bound=0,
                elapsed=time.time() - start,
                backend=f"cegar({iteration + 1}it,{len(havoc)}havoc)",
                reason=f"Spacer on abstract model: {r_abs.reason}",
            )

        # r_abs.verdict == "reachable". Pull a concrete CEX out of the
        # abstract model via BMC — Spacer's Python API does not expose
        # witness traces, so BMC on the same model is the cheapest way
        # to materialize the initial-state assignment that reaches bad.
        r_cex = Z3BMC().check_reach(model, bound=cfg.bmc_bound, timeout=cfg.bmc_timeout)
        if r_cex.verdict != "reachable":
            return SolverResult(
                verdict="unknown",
                bound=0,
                elapsed=time.time() - start,
                backend=f"cegar({iteration + 1}it,{len(havoc)}havoc)",
                reason=f"Spacer said reachable but BMC returned {r_cex.verdict}",
            )

        steps = simulate(binary, fn, r_cex.initial_regs, max_steps=cfg.simulate_max_steps)
        real_step = next((s.step for s in steps if s.pc == spec.target_pc), None)
        if real_step is not None:
            return SolverResult(
                verdict="reachable",
                bound=real_step,
                step=real_step,
                initial_regs=r_cex.initial_regs,
                elapsed=time.time() - start,
                backend=f"cegar({iteration + 1}it,{len(havoc)}havoc)",
            )

        # Spurious — refine by unhavoc'ing registers the concrete
        # replay actually read.
        read_regs: set[int] = set()
        for s in steps:
            if s.decoded is not None:
                read_regs |= _regs_read_by(s.decoded)
        new_havoc = havoc - read_regs
        if new_havoc == havoc:
            # Refinement saturated yet Spacer insists reachable: the
            # abstraction can't exclude this CEX with register-level
            # refinement alone. Escalate to fully concrete.
            if havoc:
                new_havoc = set()
            else:
                return SolverResult(
                    verdict="unknown",
                    bound=0,
                    elapsed=time.time() - start,
                    backend=f"cegar({iteration + 1}it,0havoc)",
                    reason="spurious CEX in fully concrete model",
                )
        havoc = new_havoc

    return SolverResult(
        verdict="unknown",
        bound=0,
        elapsed=time.time() - start,
        backend=f"cegar({cfg.max_iterations}it-max,{len(havoc)}havoc)",
        reason=f"max_iterations={cfg.max_iterations} reached",
    )
