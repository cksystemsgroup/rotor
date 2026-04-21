"""Z3 Spacer-based IC3/PDR backend.

Translates a BTOR2 Model into a Constrained Horn Clause system and
queries Z3's Spacer engine for unbounded safety. Spacer is PDR/IC3
over SMT — it returns a verdict without a depth bound.

Output verdicts:

    proved      — safety holds at every depth; `invariant` carries the
                  inductive invariant Spacer inferred.
    reachable   — concrete counterexample exists (a reachable bad state).
    unknown     — Spacer gave up (often on array-heavy models).

Encoding (single relation over the concatenation of all state vars):

    Inv(S) :- S satisfies every `init` binding.
    Inv(S') :- Inv(S) /\ constraints(S) /\ S' = next(S, inputs).
    Query: ∃ S. Inv(S) /\ bad(S).

`bound` is accepted for `SolverBackend` Protocol compatibility but
ignored — PDR is unbounded by construction. `timeout` is honored via
the Fixedpoint engine's `timeout` parameter.

Memory models (array sort) work for small fixtures; Spacer's array
support is known to be weaker than its bitvector support, so large
array-heavy models may return `unknown` where BMC would still answer.
That is the expected tradeoff, not a regression: the portfolio races
Z3Spacer against Z3BMC and takes whichever lands first.
"""

from __future__ import annotations

import time
from typing import Optional

import z3

from rotor.btor2.nodes import ArraySort, Model, Node, Sort
from rotor.solvers.base import SolverResult
from rotor.solvers.z3bv import _fold


class Z3Spacer:
    name = "z3-spacer"

    def check_reach(
        self,
        model: Model,
        bound: int = 0,
        timeout: Optional[float] = None,
    ) -> SolverResult:
        start = time.time()

        ctx = z3.Context()
        bv0 = z3.BitVecVal(0, 1, ctx=ctx)
        bv1 = z3.BitVecVal(1, 1, ctx=ctx)

        states = [n for n in model.nodes if n.kind == "state"]
        inits = [n for n in model.nodes if n.kind == "init"]
        nexts = [n for n in model.nodes if n.kind == "next"]
        bads = [n for n in model.nodes if n.kind == "bad"]
        constraints = [n for n in model.nodes if n.kind == "constraint"]
        inputs = [n for n in model.nodes if n.kind == "input"]

        if not bads:
            # Nothing to check; vacuously safe.
            return SolverResult(
                verdict="proved",
                bound=0,
                elapsed=time.time() - start,
                backend=self.name,
                invariant="true",
            )

        pre_vars: dict[int, z3.ExprRef] = {st.id: _state_var(st, "pre", ctx) for st in states}
        post_vars: dict[int, z3.ExprRef] = {st.id: _state_var(st, "post", ctx) for st in states}
        init_inputs: dict[int, z3.ExprRef] = {n.id: _input_var(n, "init", ctx) for n in inputs}
        trans_inputs: dict[int, z3.ExprRef] = {n.id: _input_var(n, "trans", ctx) for n in inputs}

        state_sorts = [_z3_sort(st.sort, ctx) for st in states]
        inv = z3.Function("Inv", *state_sorts, z3.BoolSort(ctx=ctx))

        fp = z3.Fixedpoint(ctx=ctx)
        fp.set("engine", "spacer")
        if timeout is not None:
            fp.set("timeout", int(timeout * 1000))

        for v in pre_vars.values():
            fp.declare_var(v)
        for v in post_vars.values():
            fp.declare_var(v)
        for v in init_inputs.values():
            fp.declare_var(v)
        for v in trans_inputs.values():
            fp.declare_var(v)
        fp.register_relation(inv)

        pre_args = [pre_vars[st.id] for st in states]
        post_args = [post_vars[st.id] for st in states]

        # Init rule: every init'd state is bound to its init expression,
        # every constraint holds at cycle 0.
        init_vals = _fold(model, pre_vars, init_inputs, {}, ctx, bv0, bv1)
        init_body: list[z3.BoolRef] = []
        for init in inits:
            state, expr = init.operands
            init_body.append(pre_vars[state.id] == init_vals[expr.id])
        for c in constraints:
            init_body.append(init_vals[c.operands[0].id] == bv1)
        fp.rule(inv(*pre_args), init_body if init_body else [z3.BoolVal(True, ctx=ctx)])

        # Transition rule: post = next(pre, inputs), subject to constraints.
        trans_vals = _fold(model, pre_vars, trans_inputs, {}, ctx, bv0, bv1)
        next_expr_id: dict[int, int] = {
            nxt.operands[0].id: nxt.operands[1].id for nxt in nexts
        }
        trans_body: list[z3.BoolRef] = [inv(*pre_args)]
        for c in constraints:
            trans_body.append(trans_vals[c.operands[0].id] == bv1)
        for st in states:
            post_v = post_vars[st.id]
            if st.id in next_expr_id:
                trans_body.append(post_v == trans_vals[next_expr_id[st.id]])
            else:
                trans_body.append(post_v == pre_vars[st.id])
        fp.rule(inv(*post_args), trans_body)

        # Query: ∃ S. Inv(S) /\ any bad expression holds.
        query_vals = _fold(model, pre_vars, init_inputs, {}, ctx, bv0, bv1)
        bad_disjuncts = [query_vals[b.operands[0].id] == bv1 for b in bads]
        bad_expr = bad_disjuncts[0] if len(bad_disjuncts) == 1 else z3.Or(*bad_disjuncts)
        query = z3.And(inv(*pre_args), bad_expr)

        try:
            result = fp.query(query)
        except z3.Z3Exception as exc:
            return SolverResult(
                verdict="unknown",
                bound=0,
                elapsed=time.time() - start,
                backend=self.name,
                reason=f"Z3Exception: {exc}",
            )
        elapsed = time.time() - start

        if result == z3.unsat:
            invariant_str: Optional[str] = None
            try:
                invariant_str = str(fp.get_cover_delta(-1, inv))
            except Exception:                                   # pragma: no cover
                invariant_str = None
            return SolverResult(
                verdict="proved",
                bound=0,
                elapsed=elapsed,
                backend=self.name,
                invariant=invariant_str,
            )
        if result == z3.sat:
            return SolverResult(
                verdict="reachable",
                bound=0,
                elapsed=elapsed,
                backend=self.name,
            )
        return SolverResult(
            verdict="unknown",
            bound=0,
            elapsed=elapsed,
            backend=self.name,
            reason=str(result),
        )


# ---------------------------------------------------------------------------

def _z3_sort(sort, ctx: z3.Context):
    if isinstance(sort, ArraySort):
        return z3.ArraySort(
            z3.BitVecSort(sort.index.width, ctx=ctx),
            z3.BitVecSort(sort.element.width, ctx=ctx),
        )
    assert isinstance(sort, Sort)
    return z3.BitVecSort(sort.width, ctx=ctx)


def _state_var(state: Node, tag: str, ctx: z3.Context) -> z3.ExprRef:
    sort = state.sort
    if isinstance(sort, ArraySort):
        return z3.Const(
            f"{tag}_{state.name}",
            z3.ArraySort(
                z3.BitVecSort(sort.index.width, ctx=ctx),
                z3.BitVecSort(sort.element.width, ctx=ctx),
            ),
        )
    assert isinstance(sort, Sort)
    return z3.BitVec(f"{tag}_{state.name}", sort.width, ctx=ctx)


def _input_var(node: Node, tag: str, ctx: z3.Context) -> z3.ExprRef:
    assert isinstance(node.sort, Sort), "input nodes are bitvec-sorted"
    return z3.BitVec(f"{tag}_{node.name}", node.sort.width, ctx=ctx)
