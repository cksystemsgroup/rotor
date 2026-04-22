"""CVC5-based BMC backend.

Third in-process BMC engine alongside `Z3BMC` and `BitwuzlaBMC`.
Translates a BTOR2 Model into CVC5 bitvector + array terms,
unrolls the transition relation to a bounded depth, and asks CVC5
whether any bad state is reachable within that depth.

Positioning. CVC5 is actively maintained and frequently
uncorrelated with Z3 and Bitwuzla on BV workloads — it can solve
benchmarks the other two struggle with (and vice versa). Adding
it to the portfolio increases coverage without hurting the
fastest-wins race: whichever engine lands the verdict first still
wins, and slow engines cost nothing extra.

API notes. Every `check_reach` creates a fresh `TermManager` +
`Solver` pair so concurrent portfolio races don't share mutable
state (CVC5's `Solver` is not thread-safe across instances). Logic
is set to `QF_ABV` (quantifier-free bitvectors + arrays), which
covers everything rotor's `build_reach` emits.

The backend is optional: import of this module requires the
`cvc5` pip package. Callers that don't have it installed should
not pull this module — `rotor.solvers.default_portfolio()` guards
the import.
"""

from __future__ import annotations

import time
from typing import Optional

import cvc5
from cvc5 import Kind, Solver, TermManager

from rotor.btor2.nodes import ArraySort, Model, Node, Sort
from rotor.solvers.base import SolverResult


class CVC5BMC:
    name = "cvc5-bmc"

    def check_reach(
        self,
        model: Model,
        bound: int,
        timeout: Optional[float] = None,
    ) -> SolverResult:
        if bound < 0:
            raise ValueError(f"bound must be >= 0, got {bound}")
        start = time.time()

        tm = TermManager()
        s = Solver(tm)
        s.setOption("produce-models", "true")
        s.setLogic("QF_ABV")
        if timeout is not None:
            s.setOption("tlimit-per", str(max(1, int(timeout * 1000))))

        bv1 = tm.mkBitVectorSort(1)
        one_bv1 = tm.mkBitVector(1, 1)

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
            sym = _state_const(tm, st, 0)
            state_vals_0[st.id] = sym
            initial_state_syms[st.id] = sym
        input_vals_0 = _input_consts(tm, inputs, 0)
        vals_0 = _fold(model, tm, state_vals_0, input_vals_0, const_cache, bv1)
        per_step_vals.append(vals_0)
        for init in inits:
            state, expr = init.operands
            s.assertFormula(tm.mkTerm(Kind.EQUAL,
                                      state_vals_0[state.id], vals_0[expr.id]))

        # cycles 1..bound
        prev_vals = vals_0
        for k in range(1, bound + 1):
            state_vals_k: dict[int, object] = {
                st.id: _state_const(tm, st, k) for st in states
            }
            input_vals_k = _input_consts(tm, inputs, k)
            vals_k = _fold(model, tm, state_vals_k, input_vals_k, const_cache, bv1)
            for nxt in nexts:
                state, expr = nxt.operands
                s.assertFormula(tm.mkTerm(Kind.EQUAL,
                                          state_vals_k[state.id], prev_vals[expr.id]))
            per_step_vals.append(vals_k)
            prev_vals = vals_k

        step_flags: list[object] = []
        for k, vals_k in enumerate(per_step_vals):
            for c in constraints:
                s.assertFormula(tm.mkTerm(Kind.EQUAL,
                                          vals_k[c.operands[0].id], one_bv1))
            conds = [tm.mkTerm(Kind.EQUAL,
                               vals_k[b.operands[0].id], one_bv1) for b in bads]
            flag = tm.mkConst(tm.getBooleanSort(), f"__bad@{k}")
            rhs = conds[0] if len(conds) == 1 else tm.mkTerm(Kind.OR, *conds)
            s.assertFormula(tm.mkTerm(Kind.EQUAL, flag, rhs))
            step_flags.append(flag)
        disj = step_flags[0] if len(step_flags) == 1 else tm.mkTerm(Kind.OR, *step_flags)
        s.assertFormula(disj)

        verdict = s.checkSat()
        elapsed = time.time() - start

        if verdict.isSat():
            step: Optional[int] = None
            for k, flag in enumerate(step_flags):
                v = s.getValue(flag)
                # Booleans come back as a Term whose string is "true"/"false".
                if str(v).lower() == "true":
                    step = k
                    break
            initial_regs: dict[str, int] = {}
            for st in states:
                if not isinstance(st.sort, Sort):
                    continue
                bv_str = s.getValue(initial_state_syms[st.id]).getBitVectorValue()
                initial_regs[st.name] = _bv_int(bv_str, st.sort.width)
            return SolverResult(
                verdict="reachable",
                bound=bound,
                step=step,
                initial_regs=initial_regs,
                elapsed=elapsed,
                backend=self.name,
            )
        if verdict.isUnsat():
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

def _cvc5_sort(tm: TermManager, sort):
    if isinstance(sort, ArraySort):
        return tm.mkArraySort(
            tm.mkBitVectorSort(sort.index.width),
            tm.mkBitVectorSort(sort.element.width),
        )
    assert isinstance(sort, Sort)
    return tm.mkBitVectorSort(sort.width)


def _state_const(tm: TermManager, state: Node, k: int):
    return tm.mkConst(_cvc5_sort(tm, state.sort), f"{state.name}@{k}")


def _input_consts(tm: TermManager, inputs: list[Node], k: int) -> dict[int, object]:
    vals: dict[int, object] = {}
    for n in inputs:
        assert isinstance(n.sort, Sort), "input nodes are bitvec-sorted"
        vals[n.id] = tm.mkConst(tm.mkBitVectorSort(n.sort.width), f"{n.name}@{k}")
    return vals


def _fold(
    model: Model,
    tm: TermManager,
    state_vals: dict[int, object],
    input_vals: dict[int, object],
    const_cache: dict[int, object],
    bv1_sort,
) -> dict[int, object]:
    one = tm.mkBitVector(1, 1)
    zero = tm.mkBitVector(1, 0)
    vals: dict[int, object] = {}
    for n in model.nodes:
        if n.kind in ("sort", "array_sort"):
            continue
        if n.kind == "const":
            if n.id not in const_cache:
                (value,) = n.operands
                const_cache[n.id] = tm.mkBitVector(n.sort.width, value)
            vals[n.id] = const_cache[n.id]
        elif n.kind == "state":
            vals[n.id] = state_vals[n.id]
        elif n.kind == "input":
            vals[n.id] = input_vals[n.id]
        elif n.kind == "op":
            vals[n.id] = _apply_op(tm, n.opname, [vals[o.id] for o in n.operands], zero, one)
        elif n.kind == "ite":
            c, t, e = [vals[o.id] for o in n.operands]
            cond = tm.mkTerm(Kind.EQUAL, c, one)
            vals[n.id] = tm.mkTerm(Kind.ITE, cond, t, e)
        elif n.kind == "slice":
            a, hi, lo = n.operands
            op = tm.mkOp(Kind.BITVECTOR_EXTRACT, hi, lo)
            vals[n.id] = tm.mkTerm(op, vals[a.id])
        elif n.kind == "ext":
            a, extra = n.operands
            kind = (Kind.BITVECTOR_ZERO_EXTEND if n.opname == "uext"
                    else Kind.BITVECTOR_SIGN_EXTEND)
            op = tm.mkOp(kind, extra)
            vals[n.id] = tm.mkTerm(op, vals[a.id])
        elif n.kind == "read":
            array, addr = n.operands
            vals[n.id] = tm.mkTerm(Kind.SELECT, vals[array.id], vals[addr.id])
        elif n.kind == "write":
            array, addr, value = n.operands
            vals[n.id] = tm.mkTerm(Kind.STORE,
                                   vals[array.id], vals[addr.id], vals[value.id])
        elif n.kind in ("init", "next", "bad", "constraint"):
            continue
        else:
            raise AssertionError(f"_fold: unknown kind {n.kind!r}")
    return vals


def _apply_op(tm: TermManager, opname: str, args, bv0, bv1):
    a = args[0]
    b = args[1] if len(args) > 1 else None
    K = Kind
    simple = {
        "add":  K.BITVECTOR_ADD,  "sub":  K.BITVECTOR_SUB,
        "and":  K.BITVECTOR_AND,  "or":   K.BITVECTOR_OR,   "xor": K.BITVECTOR_XOR,
        "sll":  K.BITVECTOR_SHL,  "srl":  K.BITVECTOR_LSHR, "sra": K.BITVECTOR_ASHR,
        "concat": K.BITVECTOR_CONCAT,
        # M extension. RISC-V edge cases are handled by the ISA
        # lowering's ITE wrappers before these ops are emitted.
        "mul":  K.BITVECTOR_MULT,
        "udiv": K.BITVECTOR_UDIV, "sdiv": K.BITVECTOR_SDIV,
        "urem": K.BITVECTOR_UREM, "srem": K.BITVECTOR_SREM,
    }
    if opname in simple:
        return tm.mkTerm(simple[opname], a, b)
    if opname == "not":
        return tm.mkTerm(K.BITVECTOR_NOT, a)
    if opname == "neg":
        return tm.mkTerm(K.BITVECTOR_NEG, a)
    cmp_kinds = {
        "eq":   K.EQUAL,
        "neq":  K.DISTINCT,
        "ult":  K.BITVECTOR_ULT,
        "ulte": K.BITVECTOR_ULE,
        "ugt":  K.BITVECTOR_UGT,
        "ugte": K.BITVECTOR_UGE,
        "slt":  K.BITVECTOR_SLT,
        "slte": K.BITVECTOR_SLE,
        "sgt":  K.BITVECTOR_SGT,
        "sgte": K.BITVECTOR_SGE,
    }
    if opname in cmp_kinds:
        cond = tm.mkTerm(cmp_kinds[opname], a, b)
        return tm.mkTerm(K.ITE, cond, bv1, bv0)
    raise ValueError(f"unknown op: {opname!r}")


def _bv_int(bv_str: str, width: int) -> int:
    """CVC5's getBitVectorValue() returns a binary string (no `#b`
    prefix by default). Convert to int, masked to `width`."""
    return int(bv_str, 2) & ((1 << width) - 1)
