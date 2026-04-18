"""BTOR2 → Bitwuzla BMC unroller.

Bitwuzla is an SMT solver, not a model checker: it does not understand
BTOR2's sequential extensions (``state``, ``init``, ``next``, ``bad``).
This module bridges the gap by walking a :class:`~rotor.btor2.NodeDAG`,
materializing each state's value at every BMC step as a fresh Bitwuzla
term, and incrementally asking the solver whether any ``bad`` predicate is
reachable.

Algorithm (standard BMC):

    step 0:  state[s,0] = eval(init[s], 0)          or fresh symbol
             check ⋁ bad[i](step 0)
    step k:  state[s,k] = eval(next[s].value, k-1)  with state refs
                                                    using step k-1
             check ⋁ bad[i](step k)
             assert ⋀ constraint[i](step k)

On SAT, a :class:`~rotor.witness.Witness`-compatible trace of state
assignments is extracted via ``bitwuzla.get_value`` on every state term at
every step.
"""

from __future__ import annotations

import time
from typing import Any

from rotor.btor2.nodes import Node, NodeDAG, Sort
from rotor.solvers.base import CheckResult
from rotor.witness import Witness, WitnessAssignment, WitnessFrame


class BitwuzlaUnroller:
    """BMC unroller over a :class:`NodeDAG` using Bitwuzla as the SMT backend."""

    def __init__(self, dag: NodeDAG, *, skip_init: bool = False) -> None:
        try:
            import bitwuzla  # noqa: F401
        except ImportError as err:
            raise ImportError(
                "bitwuzla Python package required for BitwuzlaUnroller"
            ) from err
        self.dag = dag
        self._skip_init = skip_init
        self._setup_bitwuzla()
        self._collect_structural_nodes()

    # ---------------------------------------------------------------- setup

    def _setup_bitwuzla(self) -> None:
        import bitwuzla
        self._bw_mod = bitwuzla
        self.tm = bitwuzla.TermManager()
        self.opts = bitwuzla.Options()
        self.opts.set(bitwuzla.Option.PRODUCE_MODELS, True)
        self.bw = bitwuzla.Bitwuzla(self.tm, self.opts)

        # Caches.
        self._sort_cache: dict[int, Any] = {}           # dag sort.nid → bw Sort
        self._term_cache: dict[tuple[int, int], Any] = {}  # (node.nid, step) → Term
        # state.nid → { step → Term } for materialized state values.
        self._state_at: dict[int, dict[int, Any]] = {}
        # input.nid → { step → Term }.
        self._input_at: dict[int, dict[int, Any]] = {}

    def _collect_structural_nodes(self) -> None:
        self._state_nodes: list[Node] = []
        self._input_nodes: list[Node] = []
        self._init_nodes: list[Node] = []
        self._next_nodes: list[Node] = []
        self._bad_nodes: list[Node] = []
        self._constraint_nodes: list[Node] = []
        for node in self.dag.nodes():
            if node.op == "state":
                self._state_nodes.append(node)
            elif node.op == "input":
                self._input_nodes.append(node)
            elif node.op == "init":
                self._init_nodes.append(node)
            elif node.op == "next":
                self._next_nodes.append(node)
            elif node.op == "bad":
                self._bad_nodes.append(node)
            elif node.op == "constraint":
                self._constraint_nodes.append(node)

    # ---------------------------------------------------------------- public

    def check(self, bound: int) -> CheckResult:
        """Unroll up to ``bound`` steps; return on first conclusive result."""
        import bitwuzla
        start = time.monotonic()

        if not self._bad_nodes:
            return CheckResult(
                verdict="unsat",
                solver="bitwuzla-bmc",
                elapsed=time.monotonic() - start,
                stderr="no bad properties in model",
            )

        try:
            self._materialize_states(0)
            for k in range(0, bound + 1):
                if k > 0:
                    self._materialize_states(k)
                # Assert constraints active at step k.
                for c in self._constraint_nodes:
                    self.bw.assert_formula(self._bool_of(c.args[0], k))
                # Query: is any bad condition satisfiable at step k?
                bad_disj = self._disjunction(
                    [self._bool_of(b.args[0], k) for b in self._bad_nodes]
                )
                self.bw.push(1)
                self.bw.assert_formula(bad_disj)
                result = self.bw.check_sat()
                if result == bitwuzla.Result.SAT:
                    witness = self._extract_witness(k)
                    self.bw.pop(1)
                    return CheckResult(
                        verdict="sat",
                        steps=k,
                        witness=self._witness_to_frames(witness),
                        solver="bitwuzla-bmc",
                        elapsed=time.monotonic() - start,
                    )
                self.bw.pop(1)
            return CheckResult(
                verdict="unsat",
                steps=bound,
                solver="bitwuzla-bmc",
                elapsed=time.monotonic() - start,
                stderr=f"no bad property reached within bound={bound}",
            )
        except Exception as err:  # pragma: no cover — Bitwuzla-internal failures
            return CheckResult(
                verdict="unknown",
                solver="bitwuzla-bmc",
                elapsed=time.monotonic() - start,
                stderr=f"{type(err).__name__}: {err}",
            )

    # ------------------------------------------------------------ sorts

    def _sort_of(self, sort: Sort) -> Any:
        if sort.nid in self._sort_cache:
            return self._sort_cache[sort.nid]
        if sort.kind == "bitvec":
            assert sort.width is not None
            bw_sort = self.tm.mk_bv_sort(sort.width)
        else:
            assert sort.index_sort is not None and sort.elem_sort is not None
            bw_sort = self.tm.mk_array_sort(
                self._sort_of(sort.index_sort),
                self._sort_of(sort.elem_sort),
            )
        self._sort_cache[sort.nid] = bw_sort
        return bw_sort

    # ------------------------------------------------------ state stepping

    def _materialize_states(self, k: int) -> None:
        """Populate ``self._state_at[*, k]`` for all state nodes."""
        if k == 0:
            init_map = {n.args[0].nid: n for n in self._init_nodes}
            for state in self._state_nodes:
                init = init_map.get(state.nid)
                if init is not None and not self._skip_init:
                    term = self._term_of(init.args[1], 0)
                else:
                    # Uninitialized state (or skip_init mode) → fresh
                    # symbolic constant at step 0, modelling "some arbitrary
                    # reachable state" for induction queries.
                    term = self.tm.mk_const(
                        self._sort_of(state.sort),
                        f"{state.symbol or f'state{state.nid}'}_0",
                    )
                self._state_at.setdefault(state.nid, {})[0] = term
        else:
            # state value at step k = next.value_expression at step k-1.
            next_map = {n.args[0].nid: n for n in self._next_nodes}
            # First pass: compute all new state values with references to
            # step k-1 using the already-cached terms.
            new_values: dict[int, Any] = {}
            for state in self._state_nodes:
                nx = next_map.get(state.nid)
                if nx is not None:
                    new_values[state.nid] = self._term_of(nx.args[1], k - 1)
                else:
                    # No transition: latch = prior value.
                    new_values[state.nid] = self._state_at[state.nid][k - 1]
            for nid, term in new_values.items():
                self._state_at[nid][k] = term

    # --------------------------------------------------------- evaluation

    def _term_of(self, node: Node, k: int) -> Any:
        """Bitwuzla term for ``node`` evaluated at step ``k``."""
        key = (node.nid, k)
        cached = self._term_cache.get(key)
        if cached is not None:
            return cached
        term = self._compute(node, k)
        self._term_cache[key] = term
        return term

    def _compute(self, node: Node, k: int) -> Any:
        bw = self._bw_mod
        Kind = bw.Kind

        op = node.op
        if op == "state":
            return self._state_at[node.nid][k]
        if op == "input":
            per = self._input_at.setdefault(node.nid, {})
            if k not in per:
                per[k] = self.tm.mk_const(
                    self._sort_of(node.sort),
                    f"{node.symbol or f'input{node.nid}'}@{k}",
                )
            return per[k]

        sort = self._sort_of(node.sort)
        if op == "zero":
            if node.sort.kind == "bitvec":
                return self.tm.mk_bv_zero(sort)
            # Array with all-zero elements.
            assert node.sort.elem_sort is not None
            elem_zero = self.tm.mk_bv_zero(self._sort_of(node.sort.elem_sort))
            return self.tm.mk_const_array(sort, elem_zero)
        if op == "one":
            return self.tm.mk_bv_one(sort)
        if op == "ones":
            return self.tm.mk_bv_ones(sort)
        if op in ("constd", "consth", "const"):
            # Mask negatives into their two's-complement representation.
            width = node.sort.width or 0
            value = node.params[0] & ((1 << width) - 1)
            return self.tm.mk_bv_value(sort, value)

        # N-ary operators.
        args = [self._term_of(a, k) for a in node.args]

        if op in _KIND_MAP:
            return self.tm.mk_term(_KIND_MAP[op], args)
        if op == "not":
            # bitvector bitwise NOT, unless the sort is bitvec 1 used as Bool.
            if node.sort.kind == "bitvec" and node.sort.width == 1:
                return self.tm.mk_term(Kind.BV_NOT, args)
            return self.tm.mk_term(Kind.BV_NOT, args)
        if op == "neq":
            eq = self.tm.mk_term(Kind.EQUAL, args)
            return self._bool_to_bv1(self.tm.mk_term(Kind.NOT, [eq]))
        if op == "eq":
            return self._bool_to_bv1(self.tm.mk_term(Kind.EQUAL, args))
        if op in _BOOL_CMP_MAP:
            kind = _BOOL_CMP_MAP[op]
            return self._bool_to_bv1(self.tm.mk_term(kind, args))
        if op == "ite":
            # args: [cond (bv1), then, else]
            cond = self._bv1_to_bool(args[0])
            return self.tm.mk_term(Kind.ITE, [cond, args[1], args[2]])
        if op == "sext":
            return self.tm.mk_term(Kind.BV_SIGN_EXTEND, [args[0]], [node.params[0]])
        if op == "uext":
            return self.tm.mk_term(Kind.BV_ZERO_EXTEND, [args[0]], [node.params[0]])
        if op == "slice":
            hi, lo = node.params[0], node.params[1]
            return self.tm.mk_term(Kind.BV_EXTRACT, [args[0]], [hi, lo])
        if op == "read":
            return self.tm.mk_term(Kind.ARRAY_SELECT, args)
        if op == "write":
            return self.tm.mk_term(Kind.ARRAY_STORE, args)

        raise NotImplementedError(f"BMC unroller: unsupported op {op!r}")

    # ----------------------------------------------------------- helpers

    def bad_at(self, bad_node: Node, k: int) -> Any:
        """Public: evaluate a ``bad`` node's condition at step ``k`` as a Bool."""
        if bad_node.op == "bad":
            return self._bool_of(bad_node.args[0], k)
        return self._bool_of(bad_node, k)

    def materialize_through(self, k: int) -> None:
        """Public: ensure states are materialized for steps 0..k."""
        self._materialize_states(0)
        for step in range(1, k + 1):
            self._materialize_states(step)

    def bad_nodes(self) -> list[Node]:
        return list(self._bad_nodes)

    def constraint_nodes(self) -> list[Node]:
        return list(self._constraint_nodes)

    def _bool_of(self, node: Node, k: int) -> Any:
        """Evaluate ``node`` (which is bitvec 1) at step k, return a Bool."""
        term = self._term_of(node, k)
        return self._bv1_to_bool(term)

    def _bv1_to_bool(self, term: Any) -> Any:
        Kind = self._bw_mod.Kind
        one = self.tm.mk_bv_one(self.tm.mk_bv_sort(1))
        return self.tm.mk_term(Kind.EQUAL, [term, one])

    def _bool_to_bv1(self, term: Any) -> Any:
        Kind = self._bw_mod.Kind
        bv1 = self.tm.mk_bv_sort(1)
        return self.tm.mk_term(
            Kind.ITE,
            [term, self.tm.mk_bv_one(bv1), self.tm.mk_bv_zero(bv1)],
        )

    def _disjunction(self, bools: list[Any]) -> Any:
        Kind = self._bw_mod.Kind
        if not bools:
            return self.tm.mk_false()
        if len(bools) == 1:
            return bools[0]
        return self.tm.mk_term(Kind.OR, bools)

    # ---------------------------------------------------------- witness

    # Arrays with index widths at or below this threshold are fully sampled
    # during witness extraction (one entry per index). Wider arrays — memory,
    # typically 32-bit indexed — would produce billions of entries and are
    # elided unless caller supplies an explicit index list (TODO).
    _ARRAY_EXPAND_INDEX_WIDTH = 5

    def _extract_witness(self, last_step: int) -> list[dict[str, Any]]:
        """Read state values from the model at each step.

        Scalar bitvector states are queried directly. Small array states
        (index width ≤ 5, which covers the 32-entry register file) are
        expanded: we issue ``array_select`` queries for every index and emit
        one assignment per ``symbol[index]``. Larger arrays (memory) are
        reported as a single opaque entry.
        """
        import bitwuzla
        Kind = bitwuzla.Kind
        frames: list[dict[str, Any]] = []
        for k in range(last_step + 1):
            assignments: list[tuple[str, int | None, Any, Any]] = []
            for state in self._state_nodes:
                term = self._state_at[state.nid].get(k)
                if term is None:
                    continue
                symbol = state.symbol or str(state.nid)

                if (state.sort.kind == "array"
                        and state.sort.index_sort is not None
                        and (state.sort.index_sort.width or 0)
                              <= self._ARRAY_EXPAND_INDEX_WIDTH):
                    idx_sort = self._sort_of(state.sort.index_sort)
                    n_entries = 1 << (state.sort.index_sort.width or 0)
                    for i in range(n_entries):
                        idx_term = self.tm.mk_bv_value(idx_sort, i)
                        sel = self.tm.mk_term(Kind.ARRAY_SELECT, [term, idx_term])
                        try:
                            value = self.bw.get_value(sel)
                        except Exception:
                            continue
                        assignments.append(
                            (symbol, i, state.sort.elem_sort, value)
                        )
                else:
                    try:
                        value = self.bw.get_value(term)
                    except Exception:
                        continue
                    assignments.append((symbol, None, state.sort, value))
            frames.append({"step": k, "kind": "state", "assignments": assignments})
        return frames

    def _witness_to_frames(
        self, raw_frames: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Translate raw Bitwuzla-term values into the JSON shape the rest
        of the rotor stack consumes (matching BtorMC's witness frames)."""
        frames: list[dict[str, Any]] = []
        for entry in raw_frames:
            assignments: dict[str, int] = {}
            for symbol, index, sort, value_term in entry["assignments"]:
                if sort is None or sort.kind != "bitvec":
                    continue
                try:
                    value = int(value_term.value(10))
                except Exception:
                    try:
                        value = int(value_term.value(2), 2)
                    except Exception:
                        continue
                key = symbol if index is None else f"{symbol}[{index}]"
                assignments[key] = value
            frames.append(
                {"step": entry["step"], "kind": "state", "assignments": assignments}
            )
        return frames


# ──────────────────────────────────────────────────────────────────────────
# Op-to-Kind tables (populated lazily; Kind import depends on bitwuzla being
# installed, so we populate them inside the module-level function below).
# ──────────────────────────────────────────────────────────────────────────

_KIND_MAP: dict[str, Any] = {}
_BOOL_CMP_MAP: dict[str, Any] = {}


def _populate_kind_maps() -> None:
    try:
        import bitwuzla
    except ImportError:
        return
    Kind = bitwuzla.Kind
    _KIND_MAP.update({
        "add": Kind.BV_ADD,
        "sub": Kind.BV_SUB,
        "mul": Kind.BV_MUL,
        "udiv": Kind.BV_UDIV,
        "urem": Kind.BV_UREM,
        "sdiv": Kind.BV_SDIV,
        "srem": Kind.BV_SREM,
        "and": Kind.BV_AND,
        "or": Kind.BV_OR,
        "xor": Kind.BV_XOR,
        "sll": Kind.BV_SHL,
        "srl": Kind.BV_SHR,
        "sra": Kind.BV_ASHR,
        "concat": Kind.BV_CONCAT,
        "inc": Kind.BV_INC,
        "dec": Kind.BV_DEC,
        "neg": Kind.BV_NEG,
        "redor": Kind.BV_REDOR,
        "redand": Kind.BV_REDAND,
        "redxor": Kind.BV_REDXOR,
    })
    _BOOL_CMP_MAP.update({
        "ult": Kind.BV_ULT,
        "ulte": Kind.BV_ULE,
        "ugt": Kind.BV_UGT,
        "ugte": Kind.BV_UGE,
        "slt": Kind.BV_SLT,
        "slte": Kind.BV_SLE,
        "sgt": Kind.BV_SGT,
        "sgte": Kind.BV_SGE,
    })


_populate_kind_maps()
