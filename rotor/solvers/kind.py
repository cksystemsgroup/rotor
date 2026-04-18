"""k-induction solver backed by Bitwuzla.

For a BTOR2 model with bad properties B = {b_1, ..., b_n}, k-induction tries
to prove *unbounded safety* — that none of the b_i is ever satisfiable
regardless of how many transitions fire — by splitting the problem in two:

    Base case   :  For s_0 drawn from the init predicate, and any trace
                   s_0 → s_1 → ... → s_{k-1}, no b_i holds in any of
                   these states. Equivalent to BMC up to depth k−1.

    Inductive   :  For *any* sequence s_0, ..., s_k that satisfies the
    step              transition relation and in which no b_i holds at
                      steps 0..k−1, no b_i holds at step k either.

If both succeed, the bad property is ``k-inductive`` and cannot be reached
from the initial state by any number of transitions.

k-induction is strictly weaker than IC3 — many natural properties are not
directly k-inductive for small k, and IC3 can discover the strengthening
automatically. But k-induction is enough for common loop-counter and
monotonic-state invariants that make up most practical verification queries,
and has the very nice property of not requiring any external solver beyond
Bitwuzla, which the rest of the rotor stack already uses.
"""

from __future__ import annotations

import time
from typing import Any

from rotor.btor2.nodes import NodeDAG
from rotor.solvers.base import CheckResult, SolverBackend


class KInductionSolver(SolverBackend):
    """Unbounded safety via k-induction, in-process through Bitwuzla."""

    name = "kind"

    def __init__(
        self,
        config: object | None = None,
        max_k: int = 10,
    ) -> None:
        self.config = config
        self.max_k = max_k

    def supports_unbounded(self) -> bool:
        return True

    # ---------------------------------------------------------------- check

    def check(self, btor2: str, bound: int) -> CheckResult:
        start = time.monotonic()
        try:
            import bitwuzla  # noqa: F401
        except ImportError:
            return CheckResult(
                verdict="unknown",
                solver=self.name,
                elapsed=time.monotonic() - start,
                stderr="bitwuzla package not installed",
            )

        from rotor.btor2 import parse_btor2

        try:
            dag = parse_btor2(btor2)
        except Exception as err:
            return CheckResult(
                verdict="unknown",
                solver=self.name,
                elapsed=time.monotonic() - start,
                stderr=f"BTOR2 parse failure: {err}",
            )

        result = self._check_dag(dag, bound)
        result.elapsed = time.monotonic() - start
        return result

    # ---------------------------------------------------------------- driver

    def _check_dag(self, dag: NodeDAG, bound: int) -> CheckResult:
        """Run the base case, then the inductive step at increasing k."""
        from rotor.solvers.bmc import BitwuzlaUnroller

        max_k = min(self.max_k, bound)

        # Base case: is any bad reachable from init in ≤ max_k steps?
        base = BitwuzlaUnroller(dag)
        base_result = base.check(max_k)
        if base_result.verdict == "sat":
            # Real CEX — safety violated.
            return CheckResult(
                verdict="sat",
                steps=base_result.steps,
                witness=base_result.witness,
                solver=self.name,
                stderr="base case: counterexample found within bound",
            )
        if base_result.verdict == "unknown":
            return CheckResult(
                verdict="unknown",
                solver=self.name,
                stderr=f"base case unknown: {base_result.stderr}",
            )

        # Inductive step: try k-induction for k = 1..max_k.
        for k in range(1, max_k + 1):
            if self._inductive_step(dag, k):
                return CheckResult(
                    verdict="unsat",
                    steps=None,
                    invariant=self._describe_invariant(dag, k),
                    solver=self.name,
                    stderr=f"property is {k}-inductive",
                )

        return CheckResult(
            verdict="unknown",
            solver=self.name,
            stderr=(
                f"not k-inductive for k ≤ {max_k}; BMC UNSAT to bound={max_k} "
                "but unbounded proof not found"
            ),
        )

    # ---------------------------------------------------------- inductive step

    def _inductive_step(self, dag: NodeDAG, k: int) -> bool:
        """Return True iff assuming !bad at steps 0..k-1 forces !bad at step k.

        Concretely:  build an unroller that starts from a fresh symbolic
        state, assume no bad fires at steps 0..k-1, and ask whether *any*
        bad can fire at step k. UNSAT ⇒ k-inductive.
        """
        from rotor.solvers.bmc import BitwuzlaUnroller

        u = BitwuzlaUnroller(dag, skip_init=True)
        u.materialize_through(k)

        Kind = u._bw_mod.Kind
        bads = u.bad_nodes()
        if not bads:
            return True

        # Assert constraints at every step 0..k to tie ends to the transition
        # relation of the real model, not arbitrary states.
        for c in u.constraint_nodes():
            for i in range(k + 1):
                u.bw.assert_formula(u._bool_of(c.args[0], i))

        # Assume no bad fires at steps 0..k-1.
        for i in range(k):
            not_any_bad = u.tm.mk_term(
                Kind.AND,
                [
                    u.tm.mk_term(Kind.NOT, [u.bad_at(b, i)])
                    for b in bads
                ],
            ) if len(bads) > 1 else u.tm.mk_term(
                Kind.NOT, [u.bad_at(bads[0], i)]
            )
            u.bw.assert_formula(not_any_bad)

        # Ask: can any bad fire at step k?
        if len(bads) == 1:
            disjunction = u.bad_at(bads[0], k)
        else:
            disjunction = u.tm.mk_term(
                Kind.OR, [u.bad_at(b, k) for b in bads]
            )
        u.bw.push(1)
        u.bw.assert_formula(disjunction)
        result = u.bw.check_sat()
        u.bw.pop(1)

        import bitwuzla
        return result == bitwuzla.Result.UNSAT

    # --------------------------------------------------- invariant description

    @staticmethod
    def _describe_invariant(dag: NodeDAG, k: int) -> str:
        """Produce a human-readable description of the proved property.

        The k-inductive proof certifies the negation of every ``bad``
        property. We list each by its symbol so downstream tools can
        correlate the certificate with the original check.
        """
        bads = [n for n in dag.nodes() if n.op == "bad"]
        names = [n.symbol or f"bad_{n.nid}" for n in bads]
        formatted = ", ".join(repr(n) for n in names) or "<none>"
        return (
            f"{k}-inductive invariant: for all reachable states, none of "
            f"the bad properties holds: {formatted}"
        )
