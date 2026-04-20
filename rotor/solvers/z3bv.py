"""Z3-based BMC backend.

Translates a BTOR2 Model into Z3 bitvector expressions, unrolls the
transition relation to a bounded depth, and asks Z3 whether any bad
state is reachable within that depth.

Every check_reach call runs on a fresh z3.Context — the Z3 Python API
is not thread-safe with a shared default context, and the portfolio
backend runs backends concurrently. Each fresh context is cheap and
lets us race without synchronization.

M6 adds array-sort support: memory states become `z3.Array(bv64, bv8)`,
`read` lowers to `z3.Select`, and `write` lowers to `z3.Store`.
"""

from __future__ import annotations

import time
from typing import Optional, Union

import z3

from rotor.btor2.nodes import ArraySort, Model, Node, Sort
from rotor.solvers.base import SolverResult


class Z3BMC:
    name = "z3-bmc"

    def check_reach(
        self,
        model: Model,
        bound: int,
        timeout: Optional[float] = None,
    ) -> SolverResult:
        if bound < 0:
            raise ValueError(f"bound must be >= 0, got {bound}")
        start = time.time()

        ctx = z3.Context()                             # thread-safe isolation
        bv1 = z3.BitVecVal(1, 1, ctx=ctx)
        bv0 = z3.BitVecVal(0, 1, ctx=ctx)

        states = [n for n in model.nodes if n.kind == "state"]
        inits = [n for n in model.nodes if n.kind == "init"]
        nexts = [n for n in model.nodes if n.kind == "next"]
        bads = [n for n in model.nodes if n.kind == "bad"]

        solver = z3.Solver(ctx=ctx)
        if timeout is not None:
            solver.set("timeout", int(timeout * 1000))

        const_cache: dict[int, z3.BitVecRef] = {}
        initial_state_syms: dict[int, z3.ExprRef] = {}
        per_step_vals: list[dict[int, z3.ExprRef]] = []

        # cycle 0
        state_vals_0: dict[int, z3.ExprRef] = {}
        for st in states:
            sym = _fresh_state(st, 0, ctx)
            state_vals_0[st.id] = sym
            initial_state_syms[st.id] = sym
        vals_0 = _fold(model, state_vals_0, const_cache, ctx, bv0, bv1)
        per_step_vals.append(vals_0)
        for init in inits:
            state, expr = init.operands
            solver.add(state_vals_0[state.id] == vals_0[expr.id])

        # cycles 1..bound
        prev_vals = vals_0
        for k in range(1, bound + 1):
            state_vals_k: dict[int, z3.ExprRef] = {}
            for st in states:
                state_vals_k[st.id] = _fresh_state(st, k, ctx)
            vals_k = _fold(model, state_vals_k, const_cache, ctx, bv0, bv1)
            for nxt in nexts:
                state, expr = nxt.operands
                solver.add(state_vals_k[state.id] == prev_vals[expr.id])
            per_step_vals.append(vals_k)
            prev_vals = vals_k

        step_flags: list[z3.BoolRef] = []
        for k, vals_k in enumerate(per_step_vals):
            flag = z3.Bool(f"__bad@{k}", ctx=ctx)
            conds = [vals_k[b.operands[0].id] == bv1 for b in bads]
            solver.add(flag == (z3.Or(*conds) if len(conds) > 1 else conds[0]))
            step_flags.append(flag)
        solver.add(z3.Or(*step_flags) if len(step_flags) > 1 else step_flags[0])

        verdict = solver.check()
        elapsed = time.time() - start

        if verdict == z3.sat:
            z3model = solver.model()
            step = next(
                (k for k, f in enumerate(step_flags)
                 if bool(z3model.eval(f, model_completion=True))),
                None,
            )
            initial_regs: dict[str, int] = {}
            for st in states:
                if not isinstance(st.sort, Sort):
                    continue              # array-sort states aren't scalar registers
                v = z3model.eval(initial_state_syms[st.id], model_completion=True)
                initial_regs[st.name] = v.as_long()
            return SolverResult(
                verdict="reachable",
                bound=bound,
                step=step,
                initial_regs=initial_regs,
                elapsed=elapsed,
                backend=self.name,
            )
        if verdict == z3.unsat:
            return SolverResult(
                verdict="unreachable",
                bound=bound,
                elapsed=elapsed,
                backend=self.name,
            )
        return SolverResult(
            verdict="unknown",
            bound=bound,
            elapsed=elapsed,
            backend=self.name,
            reason=str(verdict),
        )


# ---------------------------------------------------------------------------

def _fresh_state(state: Node, k: int, ctx: z3.Context) -> z3.ExprRef:
    """Create a fresh Z3 variable for `state` at cycle k."""
    name = f"{state.name}@{k}"
    sort = state.sort
    if isinstance(sort, ArraySort):
        return z3.Array(
            name,
            z3.BitVecSort(sort.index.width, ctx=ctx),
            z3.BitVecSort(sort.element.width, ctx=ctx),
        )
    assert isinstance(sort, Sort)
    return z3.BitVec(name, sort.width, ctx=ctx)


def _fold(
    model: Model,
    state_vals: dict[int, z3.ExprRef],
    const_cache: dict[int, z3.BitVecRef],
    ctx: z3.Context,
    bv0: z3.BitVecRef,
    bv1: z3.BitVecRef,
) -> dict[int, z3.ExprRef]:
    vals: dict[int, z3.ExprRef] = {}
    for n in model.nodes:
        if n.kind in ("sort", "array_sort"):
            continue
        if n.kind == "const":
            if n.id not in const_cache:
                (value,) = n.operands
                const_cache[n.id] = z3.BitVecVal(value, n.sort.width, ctx=ctx)
            vals[n.id] = const_cache[n.id]
        elif n.kind == "state":
            vals[n.id] = state_vals[n.id]
        elif n.kind == "input":
            raise NotImplementedError("input nodes not supported by M1/M2 backend")
        elif n.kind == "op":
            vals[n.id] = _apply_op(n.opname, [vals[o.id] for o in n.operands], bv0, bv1)
        elif n.kind == "ite":
            c, t, e = [vals[o.id] for o in n.operands]
            vals[n.id] = z3.If(c == bv1, t, e, ctx=ctx)
        elif n.kind == "slice":
            a, hi, lo = n.operands
            vals[n.id] = z3.Extract(hi, lo, vals[a.id])
        elif n.kind == "ext":
            a, extra = n.operands
            if n.opname == "uext":
                vals[n.id] = z3.ZeroExt(extra, vals[a.id])
            else:
                vals[n.id] = z3.SignExt(extra, vals[a.id])
        elif n.kind == "read":
            array, addr = n.operands
            vals[n.id] = z3.Select(vals[array.id], vals[addr.id])
        elif n.kind == "write":
            array, addr, value = n.operands
            vals[n.id] = z3.Store(vals[array.id], vals[addr.id], vals[value.id])
        elif n.kind in ("init", "next", "bad"):
            continue
        else:
            raise AssertionError(f"_fold: unknown kind {n.kind!r}")
    return vals


def _apply_op(
    opname: str,
    args: list[z3.BitVecRef],
    bv0: z3.BitVecRef,
    bv1: z3.BitVecRef,
) -> z3.BitVecRef:
    a = args[0]
    b = args[1] if len(args) > 1 else None
    if opname == "add":
        return a + b
    if opname == "sub":
        return a - b
    if opname == "and":
        return a & b
    if opname == "or":
        return a | b
    if opname == "xor":
        return a ^ b
    if opname == "not":
        return ~a
    if opname == "neg":
        return -a
    if opname == "sll":
        return a << b
    if opname == "srl":
        return z3.LShR(a, b)
    if opname == "sra":
        return a >> b
    if opname == "eq":
        return z3.If(a == b, bv1, bv0)
    if opname == "neq":
        return z3.If(a != b, bv1, bv0)
    if opname == "ult":
        return z3.If(z3.ULT(a, b), bv1, bv0)
    if opname == "ulte":
        return z3.If(z3.ULE(a, b), bv1, bv0)
    if opname == "ugt":
        return z3.If(z3.UGT(a, b), bv1, bv0)
    if opname == "ugte":
        return z3.If(z3.UGE(a, b), bv1, bv0)
    if opname == "slt":
        return z3.If(a < b, bv1, bv0)
    if opname == "slte":
        return z3.If(a <= b, bv1, bv0)
    if opname == "sgt":
        return z3.If(a > b, bv1, bv0)
    if opname == "sgte":
        return z3.If(a >= b, bv1, bv0)
    if opname == "concat":
        return z3.Concat(a, b)
    raise ValueError(f"unknown op: {opname!r}")
