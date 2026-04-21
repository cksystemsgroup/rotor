"""Bitwuzla-based BMC backend.

Translates a BTOR2 Model into Bitwuzla bitvector + array terms,
unrolls the transition relation to a bounded depth, and asks
Bitwuzla whether any bad state is reachable within that depth.

Bitwuzla's Python bindings ship an in-process API similar in shape
to Z3's. We use a fresh `TermManager` + `Bitwuzla` per `check_reach`
call so concurrent portfolio races do not share mutable state.

Positioning (Track A, ROADMAP.md). Bitwuzla typically outperforms
Z3 on pure-BV workloads — often by an order of magnitude on the
HWMCC BV track. Rotor's portfolio races it against Z3BMC so
whichever engine lands the verdict first wins.

The backend is optional: imports gated on the `bitwuzla` Python
package. Callers that don't have it installed should not pull this
module.
"""

from __future__ import annotations

import time
from typing import Optional

import bitwuzla as bz

from rotor.btor2.nodes import ArraySort, Model, Node, Sort
from rotor.solvers.base import SolverResult


class BitwuzlaBMC:
    name = "bitwuzla-bmc"

    def check_reach(
        self,
        model: Model,
        bound: int,
        timeout: Optional[float] = None,
    ) -> SolverResult:
        if bound < 0:
            raise ValueError(f"bound must be >= 0, got {bound}")
        start = time.time()

        tm = bz.TermManager()
        opts = bz.Options()
        opts.set(bz.Option.PRODUCE_MODELS, True)
        if timeout is not None:
            opts.set(bz.Option.TIME_LIMIT_PER, max(1, int(timeout * 1000)))
        solver = bz.Bitwuzla(tm, opts)

        bv1 = tm.mk_bv_sort(1)
        one_bv1 = tm.mk_bv_value(bv1, 1)

        states = [n for n in model.nodes if n.kind == "state"]
        inits = [n for n in model.nodes if n.kind == "init"]
        nexts = [n for n in model.nodes if n.kind == "next"]
        bads = [n for n in model.nodes if n.kind == "bad"]
        constraints = [n for n in model.nodes if n.kind == "constraint"]
        inputs = [n for n in model.nodes if n.kind == "input"]

        const_cache: dict[int, object] = {}
        initial_state_syms: dict[int, object] = {}
        per_step_vals: list[dict[int, object]] = []

        # cycle 0
        state_vals_0: dict[int, object] = {}
        for st in states:
            sym = _fresh_state(tm, st, 0)
            state_vals_0[st.id] = sym
            initial_state_syms[st.id] = sym
        input_vals_0 = _fresh_inputs(tm, inputs, 0)
        vals_0 = _fold(model, tm, state_vals_0, input_vals_0, const_cache, bv1)
        per_step_vals.append(vals_0)
        for init in inits:
            state, expr = init.operands
            solver.assert_formula(tm.mk_term(bz.Kind.EQUAL,
                                             [state_vals_0[state.id], vals_0[expr.id]]))

        # cycles 1..bound
        prev_vals = vals_0
        for k in range(1, bound + 1):
            state_vals_k: dict[int, object] = {
                st.id: _fresh_state(tm, st, k) for st in states
            }
            input_vals_k = _fresh_inputs(tm, inputs, k)
            vals_k = _fold(model, tm, state_vals_k, input_vals_k, const_cache, bv1)
            for nxt in nexts:
                state, expr = nxt.operands
                solver.assert_formula(tm.mk_term(bz.Kind.EQUAL,
                                                 [state_vals_k[state.id], prev_vals[expr.id]]))
            per_step_vals.append(vals_k)
            prev_vals = vals_k

        step_flags: list[object] = []
        for k, vals_k in enumerate(per_step_vals):
            for c in constraints:
                solver.assert_formula(tm.mk_term(bz.Kind.EQUAL,
                                                 [vals_k[c.operands[0].id], one_bv1]))
            conds = [tm.mk_term(bz.Kind.EQUAL,
                                [vals_k[b.operands[0].id], one_bv1]) for b in bads]
            flag = tm.mk_const(tm.mk_bool_sort(), f"__bad@{k}")
            rhs = conds[0] if len(conds) == 1 else tm.mk_term(bz.Kind.OR, conds)
            solver.assert_formula(tm.mk_term(bz.Kind.EQUAL, [flag, rhs]))
            step_flags.append(flag)
        disj = step_flags[0] if len(step_flags) == 1 else tm.mk_term(bz.Kind.OR, step_flags)
        solver.assert_formula(disj)

        verdict = solver.check_sat()
        elapsed = time.time() - start

        if verdict == bz.Result.SAT:
            step: Optional[int] = None
            for k, flag in enumerate(step_flags):
                if _bool_value(solver.get_value(flag)):
                    step = k
                    break
            initial_regs: dict[str, int] = {}
            for st in states:
                if not isinstance(st.sort, Sort):
                    continue                         # array states aren't scalars
                v = solver.get_value(initial_state_syms[st.id])
                initial_regs[st.name] = _bv_value_int(v, st.sort.width)
            return SolverResult(
                verdict="reachable",
                bound=bound,
                step=step,
                initial_regs=initial_regs,
                elapsed=elapsed,
                backend=self.name,
            )
        if verdict == bz.Result.UNSAT:
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

def _fresh_state(tm, state: Node, k: int):
    name = f"{state.name}@{k}"
    sort = state.sort
    if isinstance(sort, ArraySort):
        return tm.mk_const(
            tm.mk_array_sort(
                tm.mk_bv_sort(sort.index.width),
                tm.mk_bv_sort(sort.element.width),
            ),
            name,
        )
    assert isinstance(sort, Sort)
    return tm.mk_const(tm.mk_bv_sort(sort.width), name)


def _fresh_inputs(tm, inputs: list[Node], k: int) -> dict[int, object]:
    vals: dict[int, object] = {}
    for n in inputs:
        assert isinstance(n.sort, Sort), "input nodes are bitvec-sorted"
        vals[n.id] = tm.mk_const(tm.mk_bv_sort(n.sort.width), f"{n.name}@{k}")
    return vals


def _fold(
    model: Model,
    tm,
    state_vals: dict[int, object],
    input_vals: dict[int, object],
    const_cache: dict[int, object],
    bv1_sort,
) -> dict[int, object]:
    one = tm.mk_bv_value(bv1_sort, 1)
    zero = tm.mk_bv_value(bv1_sort, 0)
    vals: dict[int, object] = {}
    for n in model.nodes:
        if n.kind in ("sort", "array_sort"):
            continue
        if n.kind == "const":
            if n.id not in const_cache:
                (value,) = n.operands
                const_cache[n.id] = tm.mk_bv_value(tm.mk_bv_sort(n.sort.width), value)
            vals[n.id] = const_cache[n.id]
        elif n.kind == "state":
            vals[n.id] = state_vals[n.id]
        elif n.kind == "input":
            vals[n.id] = input_vals[n.id]
        elif n.kind == "op":
            vals[n.id] = _apply_op(tm, n.opname, [vals[o.id] for o in n.operands], zero, one)
        elif n.kind == "ite":
            c, t, e = [vals[o.id] for o in n.operands]
            cond = tm.mk_term(bz.Kind.EQUAL, [c, one])
            vals[n.id] = tm.mk_term(bz.Kind.ITE, [cond, t, e])
        elif n.kind == "slice":
            a, hi, lo = n.operands
            vals[n.id] = tm.mk_term(bz.Kind.BV_EXTRACT, [vals[a.id]], [hi, lo])
        elif n.kind == "ext":
            a, extra = n.operands
            kind = bz.Kind.BV_ZERO_EXTEND if n.opname == "uext" else bz.Kind.BV_SIGN_EXTEND
            vals[n.id] = tm.mk_term(kind, [vals[a.id]], [extra])
        elif n.kind == "read":
            array, addr = n.operands
            vals[n.id] = tm.mk_term(bz.Kind.ARRAY_SELECT, [vals[array.id], vals[addr.id]])
        elif n.kind == "write":
            array, addr, value = n.operands
            vals[n.id] = tm.mk_term(bz.Kind.ARRAY_STORE,
                                    [vals[array.id], vals[addr.id], vals[value.id]])
        elif n.kind in ("init", "next", "bad", "constraint"):
            continue
        else:
            raise AssertionError(f"_fold: unknown kind {n.kind!r}")
    return vals


def _apply_op(tm, opname: str, args, bv0, bv1):
    a = args[0]
    b = args[1] if len(args) > 1 else None
    K = bz.Kind
    simple = {
        "add": K.BV_ADD, "sub": K.BV_SUB,
        "and": K.BV_AND, "or": K.BV_OR, "xor": K.BV_XOR,
        "sll": K.BV_SHL, "srl": K.BV_SHR, "sra": K.BV_ASHR,
        "concat": K.BV_CONCAT,
    }
    if opname in simple:
        return tm.mk_term(simple[opname], [a, b])
    if opname == "not":
        return tm.mk_term(K.BV_NOT, [a])
    if opname == "neg":
        return tm.mk_term(K.BV_NEG, [a])
    # Boolean-result comparisons wrap back into a 1-bit bitvector.
    cmp_kinds = {
        "eq":   (K.EQUAL,    False),
        "neq":  (K.DISTINCT, False),
        "ult":  (K.BV_ULT,   False),
        "ulte": (K.BV_ULE,   False),
        "ugt":  (K.BV_UGT,   False),
        "ugte": (K.BV_UGE,   False),
        "slt":  (K.BV_SLT,   False),
        "slte": (K.BV_SLE,   False),
        "sgt":  (K.BV_SGT,   False),
        "sgte": (K.BV_SGE,   False),
    }
    if opname in cmp_kinds:
        kind, _ = cmp_kinds[opname]
        cond = tm.mk_term(kind, [a, b])
        return tm.mk_term(K.ITE, [cond, bv1, bv0])
    raise ValueError(f"unknown op: {opname!r}")


def _bool_value(term) -> bool:
    v = term.value()
    if isinstance(v, bool):
        return v
    # Sometimes Bitwuzla returns the boolean as a string or int.
    return bool(v)


def _bv_value_int(term, width: int) -> int:
    """Extract an integer from a Bitwuzla BV value term."""
    raw = term.value()
    if isinstance(raw, int):
        return raw & ((1 << width) - 1)
    # Bitwuzla's default string form is a binary literal like "00001010".
    s = str(raw)
    if s.startswith("#b"):
        s = s[2:]
    return int(s, 2) & ((1 << width) - 1)
