"""Compile (binary, function, question) into a BTOR2 Model.

The resulting Model has:

    state pc                          init = fn.start
    state x1..x31                     no init (free / symbolic)
    state mem    : bv64 -> bv8        init = write-chain over free_mem
    state free_mem : bv64 -> bv8      no init (free base array)
    next  pc   = dispatch ITE on pc over instruction PCs in fn
    next  xN   = dispatch ITE on pc over writes to xN in fn
    next  mem  = dispatch ITE on pc over memory writes in fn
    bad   pc == target_pc             (for can_reach)

Plus a constraint encoding the **function-entry assumption**:

    constraint (ra & ~1) ∉ [fn.start, fn.end)

Without this, free `ra` lets the solver park an arbitrary intra-fn
PC in ra and have any subsequent `ret` jump to it — so every
even-aligned PC in the function becomes trivially reachable, which
collapses `can_reach` into a bound-counting exercise. The constraint
reflects a real caller's return address: it lives outside the
function being analyzed.

This is the first piece of what will grow into a richer **entry-state
model** over time: in a bootloaded program the loader sets `sp` to a
valid stack, `ra` to an exit sentinel, and `a0..a7` to argument
values per the ABI. Rotor currently models only leaf functions, so
`ra` is the only entry value that materially shifts what `can_reach`
answers. When non-leaf functions and real call boundaries land, this
constraint becomes one facet of a proper `EntryAssumptions` object.

x0 is a constant 0 node, not a state.  PCs outside fn keep their current
value (self-loop), which is how the machine "halts" after a return
unless the return target is also inside fn.

M6 adds the SMT array memory model. Loadable bytes from the ELF's
PT_LOAD segments are baked into mem's init expression via a chain of
`write` nodes over a free base array. Anything outside those segments
(stack, uninitialized heap, .bss) reads back as free bitvector bytes,
which is the right approximation: a store followed by a matching load
on the stack returns the stored value exactly, while uninitialized
reads remain nondeterministic — which is how rotor surfaces memory
bugs rather than masking them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

from rotor.binary import Function, RISCVBinary
from rotor.btor2.nodes import ArraySort, Model, Node, Sort
from rotor.btor2.riscv.decoder import decode
from rotor.btor2.riscv.isa import MEMORY_MNEMONICS, lower
from rotor.ir.spec import EquivalenceSpec, FindInputSpec, ReachSpec, VerifySpec

BV1 = Sort(1)
BV8 = Sort(8)
BV64 = Sort(64)
MEM_SORT = ArraySort(index=BV64, element=BV8)
NREGS = 32


class UnsupportedInstruction(ValueError):
    def __init__(self, pc: int, word: int) -> None:
        super().__init__(f"unsupported instruction 0x{word:08x} at pc 0x{pc:x}")
        self.pc = pc
        self.word = word


@dataclass(frozen=True)
class EntryAssumptions:
    """Cycle-0 assumptions about the machine state at function entry.

    Rotor's early builds hard-coded one assumption directly into
    `_build_machine`: the caller's return address (`ra`) lies
    outside the analyzed function's PC range, so `ret` doesn't
    spuriously land back inside the same function under solver
    pressure. Track C promoted this to a first-class object so that:

    1. Non-leaf analyses can extend `excluded_pc_ranges` to cover
       every function in the analyzed set — a callee `ret` must
       leave the set, while a caller's intra-set `jal` writing
       `pc+4` into ra during execution stays unconstrained.
    2. Future sp / argument / callee-saved assumptions slot in as
       new fields without another ra-style hard-coded rewrite.

    The scope-aware refactor that made this useful was moving the
    ra constraint from a global invariant to a cycle-0 constraint.
    `_build_machine` now ties `ra` to a fresh input `init_ra`;
    constraints apply to that input (which is only used in init),
    so the ra value at later cycles — e.g. after a jal wrote `pc+4`
    — is free, as it should be.
    """
    excluded_pc_ranges: tuple[tuple[int, int], ...] = ()

    @staticmethod
    def from_functions(binary: RISCVBinary, function_names) -> "EntryAssumptions":
        ranges = []
        for name in function_names:
            fn = binary.function(name)
            ranges.append((fn.start, fn.end))
        return EntryAssumptions(excluded_pc_ranges=tuple(ranges))


@dataclass
class _Machine:
    """Internal: the state and structure shared by every question-
    building function. All `build_*` functions populate a Model with
    the same machine description (pc + regs + optional mem, transition
    relation, entry assumptions) and then add their verb-specific
    `bad` expression on top."""
    m: Model
    regs: list[Node]
    pc: Node
    mem: Optional[Node]
    decoded: list[tuple]                             # list[(Instruction, Decoded)]
    entry_fn: "Function"                             # observation scope for verify /
                                                     # find_input / equivalent (callee
                                                     # rets are ignored)


def _assume_ra_outside_ranges(m: Model, ra_source: Node,
                              ranges: tuple[tuple[int, int], ...]) -> None:
    """Constrain `ra_source` to lie outside every `(start, end)`
    range (bit-0 masked off to match `jalr`'s `& ~1` target).

    Expressed as a BTOR2 constraint; since `ra_source` is typically
    a cycle-0-only BTOR2 input (driving ra's init), the constraint
    bites only at entry — subsequent cycles in which `ra` holds an
    intra-set PC (e.g. after a `jal` wrote `pc+4`) are unaffected.
    """
    if not ranges:
        return
    ra_masked = m.op("and", BV64, ra_source, m.const(BV64, 0xFFFF_FFFF_FFFF_FFFE))
    conds = []
    for (start, end) in ranges:
        below = m.op("ult", BV1, ra_masked, m.const(BV64, start))
        at_or_above = m.op("ugte", BV1, ra_masked, m.const(BV64, end))
        conds.append(m.op("or", BV1, below, at_or_above))
    combined = conds[0]
    for c in conds[1:]:
        combined = m.op("and", BV1, combined, c)
    m.constraint(combined)


def _build_machine(
    binary: RISCVBinary,
    function_name: str,
    builder: Optional[Model],
    havoc_regs: Optional[set[int]],
    *,
    prefix: str = "",
    shared_reg_inits: Optional[dict[int, Node]] = None,
    include_fns: Optional[list[str]] = None,
) -> _Machine:
    """Build the shared machine model (regs, pc, mem, transition
    relation, entry assumptions) for a function. Verb-specific
    `build_*` callers append their own `bad` expression after.

    `prefix`, when non-empty, is applied to every state and input
    name (e.g. `a_` / `b_` for the two sides of an equivalence
    product). `shared_reg_inits`, when provided, maps register index
    → initial-value expression; both sides of an equivalence
    product pass the same map, so their cycle-0 register states
    are equated without forcing equality at later cycles.

    `include_fns`, when non-empty, names additional functions to
    fold into the PC dispatch. Their instructions are decoded and
    added to the transition relation so intra-set `jal`s land on
    real instructions and callee `ret`s legitimately return to
    caller PCs. The ra entry assumption is widened to exclude the
    union of all included fns' PC ranges. This is how Track C
    supports non-leaf analysis.
    """
    entry_fn = binary.function(function_name)
    extra_fns = [binary.function(n) for n in (include_fns or [])]
    included = [entry_fn] + extra_fns
    assumptions = EntryAssumptions(
        excluded_pc_ranges=tuple((f.start, f.end) for f in included)
    )

    m = builder if builder is not None else Model()
    havoc = set(havoc_regs) if havoc_regs else set()
    havoc.discard(0)                              # x0 is a constant, never havoc'd

    zero = m.const(BV64, 0)
    regs: list[Node] = [zero]
    for i in range(1, NREGS):
        name = f"{prefix}x{i}"
        if i in havoc:
            regs.append(m.input(BV64, name))
        else:
            state = m.state(BV64, name)
            regs.append(state)
            if shared_reg_inits is not None and i in shared_reg_inits:
                m.init(state, shared_reg_inits[i])

    # ra cycle-0 assumption: the entry-time ra must point outside
    # every analyzed function's PC range. We drive ra's init from
    # a fresh input (`init_ra`) and constrain that input rather
    # than ra itself — so after a `jal` writes `pc+4` into ra
    # during execution, the constraint doesn't spuriously fire.
    if 1 not in havoc and assumptions.excluded_pc_ranges:
        if shared_reg_inits is not None and 1 in shared_reg_inits:
            # Equivalence path: ra is already init'd to a shared
            # input by the caller. Constrain the shared input so
            # both sides' entry ra's agree.
            ra_source = shared_reg_inits[1]
        else:
            ra_source = m.input(BV64, f"{prefix}init_ra")
            m.init(regs[1], ra_source)
        _assume_ra_outside_ranges(m, ra_source, assumptions.excluded_pc_ranges)

    pc = m.state(BV64, f"{prefix}pc")
    m.init(pc, m.const(BV64, entry_fn.start))

    decoded: list[tuple] = []
    for f in included:
        for inst in binary.instructions(f):
            d = decode(inst.word)
            if d is None:
                raise UnsupportedInstruction(inst.pc, inst.word)
            if inst.size != d.size:
                d = dataclasses.replace(d, size=inst.size)
            decoded.append((inst, d))
    uses_memory = any(d.mnem in MEMORY_MNEMONICS for _, d in decoded)

    mem: Optional[Node] = None
    free_mem: Optional[Node] = None
    if uses_memory:
        free_mem = m.state(MEM_SORT, f"{prefix}free_mem")
        mem = m.state(MEM_SORT, f"{prefix}mem")
        mem_init = free_mem
        for vaddr, byte in binary.loadable_bytes():
            mem_init = m.write(mem_init, m.const(BV64, vaddr), m.const(BV8, byte))
        m.init(mem, mem_init)

    next_pc = pc
    next_regs: dict[int, Node] = {
        i: regs[i] for i in range(1, NREGS) if i not in havoc
    }
    next_mem = mem

    for inst, d in decoded:
        writes, npc, mem_write = lower(d, inst.pc, m, regs, mem)
        here = m.op("eq", BV1, pc, m.const(BV64, inst.pc))
        next_pc = m.ite(here, npc, next_pc)
        for rd, expr in writes.items():
            if rd in havoc:
                continue
            next_regs[rd] = m.ite(here, expr, next_regs[rd])
        if mem_write is not None:
            assert next_mem is not None
            next_mem = m.ite(here, mem_write, next_mem)

    for i, nxt in next_regs.items():
        m.next(regs[i], nxt)
    m.next(pc, next_pc)
    if mem is not None:
        assert next_mem is not None and free_mem is not None
        m.next(mem, next_mem)
        m.next(free_mem, free_mem)

    return _Machine(m=m, regs=regs, pc=pc, mem=mem, decoded=decoded, entry_fn=entry_fn)


def build_reach(
    binary: RISCVBinary,
    spec: ReachSpec,
    builder: Optional[Model] = None,
    havoc_regs: Optional[set[int]] = None,
    include_fns: Optional[list[str]] = None,
) -> Model:
    """Compile a ReachSpec into a BTOR2 Model.

    `builder`, when provided, must be Model-compatible. IR layers pass a
    hash-consing / simplifying subclass; L0 uses a plain Model.

    `havoc_regs`, when non-empty, is the CEGAR abstraction primitive:
    each register index in the set is replaced by a per-cycle BTOR2
    `input` instead of a `state` with computed next-state. Reads see
    a fresh symbolic value every cycle; writes are dropped. This
    over-approximates real behavior — any trace in the concrete model
    is still a trace here, but the abstraction admits extra traces
    where the register's value diverges from its computed semantics.
    Rotor's CEGAR loop (rotor/cegar.py) drives refinement by removing
    registers from this set one counterexample at a time. `x0` and
    `pc` are never havoc'd; the entry-state ra-outside-fn constraint
    is elided when ra (x1) is havoc'd, since the over-approximation
    already admits the path the constraint would have blocked.
    """
    mc = _build_machine(binary, spec.function, builder, havoc_regs,
                        include_fns=include_fns)
    # Reach obligation: pc == target.
    target = mc.m.const(BV64, spec.target_pc)
    mc.m.bad(mc.m.op("eq", BV1, mc.pc, target))
    return mc.m


def _return_site_bad(mc: "_Machine", spec, function_name: str, *, negate: bool) -> Node:
    """Build the shared `bad` expression for verb specs that observe
    a register at every return site of the function.

    `negate=True` gives verify semantics — bad = (pc at ret) ∧
    ¬predicate — so `reachable` means the predicate failed.
    `negate=False` gives find_input semantics — bad = (pc at ret) ∧
    predicate — so `reachable` means the predicate is achievable
    and the CEX is the synthesized input.
    """
    m = mc.m
    # Observe only the entry function's rets — a callee's ret doesn't
    # terminate the overall analysis, it just returns control to the
    # caller whose instructions are also in the dispatch.
    entry_range = (mc.entry_fn.start, mc.entry_fn.end)
    ret_pcs = [
        inst.pc for inst, d in mc.decoded
        if d.mnem == "jalr" and d.rd == 0 and d.rs1 == 1 and d.imm == 0
        and entry_range[0] <= inst.pc < entry_range[1]
    ]
    if not ret_pcs:
        raise ValueError(
            f"function {function_name!r} has no ret instruction; "
            f"this verb requires at least one return site"
        )

    at_ret = m.op("eq", BV1, mc.pc, m.const(BV64, ret_pcs[0]))
    for ret_pc in ret_pcs[1:]:
        at_ret = m.op("or", BV1, at_ret, m.op("eq", BV1, mc.pc, m.const(BV64, ret_pc)))

    reg_val = mc.regs[spec.register]
    rhs_const = m.const(BV64, spec.rhs & ((1 << 64) - 1))
    predicate = m.op(spec.comparison, BV1, reg_val, rhs_const)
    if negate:
        predicate = m.op("eq", BV1, predicate, m.const(BV1, 0))
    return m.op("and", BV1, at_ret, predicate)


def build_verify(
    binary: RISCVBinary,
    spec: "VerifySpec",
    builder: Optional[Model] = None,
    havoc_regs: Optional[set[int]] = None,
    include_fns: Optional[list[str]] = None,
) -> Model:
    """Compile a VerifySpec into a BTOR2 Model.

    bad = (pc at any `ret` in fn) ∧ ¬(regs[R] OP rhs)

    `proved` means the predicate holds on every execution path
    that returns from the function; `reachable` means some input
    makes the predicate fail at a return site.

    Identifying ret PCs is static: we scan `decoded` for `jalr
    x0, x1, 0` (canonical `ret`) within the entry fn's PC range.
    Callee rets (under `include_fns`) are not observation sites —
    only the entry fn's rets count as the function returning.
    """
    mc = _build_machine(binary, spec.function, builder, havoc_regs,
                        include_fns=include_fns)
    mc.m.bad(_return_site_bad(mc, spec, spec.function, negate=True))
    return mc.m


def build_find_input(
    binary: RISCVBinary,
    spec: "FindInputSpec",
    builder: Optional[Model] = None,
    havoc_regs: Optional[set[int]] = None,
    include_fns: Optional[list[str]] = None,
) -> Model:
    """Compile a FindInputSpec into a BTOR2 Model.

    bad = (pc at any `ret` in fn) ∧ (regs[R] OP rhs)

    `reachable` means an input exists that makes the predicate hold
    at a return site; `initial_regs` is the synthesized witness.
    `unreachable` (bounded) or `proved` (unbounded) means no such
    input exists within the bound / on any execution path.
    """
    mc = _build_machine(binary, spec.function, builder, havoc_regs,
                        include_fns=include_fns)
    mc.m.bad(_return_site_bad(mc, spec, spec.function, negate=False))
    return mc.m


def build_equivalence(
    binary_a: RISCVBinary,
    function_a: str,
    binary_b: RISCVBinary,
    function_b: str,
    *,
    output_register: int = 10,
    builder: Optional[Model] = None,
) -> Model:
    """Compile an equivalence question into a BTOR2 Model.

    Two copies of the RISC-V machine run in parallel inside one Model
    (state names prefixed `a_` and `b_`). All 31 architectural
    registers (x1..x31) are init'd from a shared set of BTOR2 input
    nodes, so the two sides start in identical machine state but
    evolve independently from there. Memory is not shared — leaf
    functions that don't touch memory are in scope today; a shared-
    memory overlay for non-leaf equivalence belongs to Track C
    / future work.

    bad = (pc_a at any ret in fn_a)
        ∧ (pc_b at any ret in fn_b)
        ∧ (regs_a[output_register] ≠ regs_b[output_register])

    Reachable → the two sides disagree on the output register at
    some pair of return sites given identical inputs, and
    initial_regs is the counterexample input. Unreachable → no
    disagreement up to the BMC bound.
    """
    m = builder if builder is not None else Model()

    # Shared initial values for every architectural register. Using
    # BTOR2 `input` nodes guarantees the cycle-0 binding is fresh
    # (so the two sides can evolve independently) but the init
    # expressions on both sides reference the *same* input, so they
    # agree at cycle 0.
    shared: dict[int, Node] = {
        i: m.input(BV64, f"arg_x{i}") for i in range(1, NREGS)
    }

    mc_a = _build_machine(
        binary_a, function_a, builder=m, havoc_regs=None,
        prefix="a_", shared_reg_inits=shared,
    )
    mc_b = _build_machine(
        binary_b, function_b, builder=m, havoc_regs=None,
        prefix="b_", shared_reg_inits=shared,
    )

    range_a = (mc_a.entry_fn.start, mc_a.entry_fn.end)
    range_b = (mc_b.entry_fn.start, mc_b.entry_fn.end)
    ret_pcs_a = [
        inst.pc for inst, d in mc_a.decoded
        if d.mnem == "jalr" and d.rd == 0 and d.rs1 == 1 and d.imm == 0
        and range_a[0] <= inst.pc < range_a[1]
    ]
    ret_pcs_b = [
        inst.pc for inst, d in mc_b.decoded
        if d.mnem == "jalr" and d.rd == 0 and d.rs1 == 1 and d.imm == 0
        and range_b[0] <= inst.pc < range_b[1]
    ]
    if not ret_pcs_a:
        raise ValueError(f"function {function_a!r} (side A) has no ret instruction")
    if not ret_pcs_b:
        raise ValueError(f"function {function_b!r} (side B) has no ret instruction")

    def _any_ret(pc_node, ret_pcs):
        cond = m.op("eq", BV1, pc_node, m.const(BV64, ret_pcs[0]))
        for p in ret_pcs[1:]:
            cond = m.op("or", BV1, cond, m.op("eq", BV1, pc_node, m.const(BV64, p)))
        return cond

    at_ret_a_now = _any_ret(mc_a.pc, ret_pcs_a)
    at_ret_b_now = _any_ret(mc_b.pc, ret_pcs_b)

    # Each side's "at ret" holds for only one cycle (the cycle before
    # the ret executes). Since the two machines run in lockstep but
    # the fn's have different ret depths, we can't ask "both at ret
    # simultaneously". Instead, latch each side's output when it
    # reaches a ret, and compare latched values once both have
    # returned.
    false_bv1 = m.const(BV1, 0)
    returned_a = m.state(BV1, "returned_a")
    returned_b = m.state(BV1, "returned_b")
    m.init(returned_a, false_bv1)
    m.init(returned_b, false_bv1)
    m.next(returned_a, m.op("or", BV1, returned_a, at_ret_a_now))
    m.next(returned_b, m.op("or", BV1, returned_b, at_ret_b_now))

    captured_a = m.state(BV64, "captured_a")
    captured_b = m.state(BV64, "captured_b")
    m.init(captured_a, m.const(BV64, 0))
    m.init(captured_b, m.const(BV64, 0))
    # Capture on the transition into "returned" (the cycle where pc
    # is at a ret and returned is still false). Subsequent cycles
    # keep the captured value.
    capture_now_a = m.op("and", BV1, at_ret_a_now,
                         m.op("eq", BV1, returned_a, false_bv1))
    capture_now_b = m.op("and", BV1, at_ret_b_now,
                         m.op("eq", BV1, returned_b, false_bv1))
    m.next(captured_a, m.ite(capture_now_a, mc_a.regs[output_register], captured_a))
    m.next(captured_b, m.ite(capture_now_b, mc_b.regs[output_register], captured_b))

    both_returned = m.op("and", BV1, returned_a, returned_b)
    diff = m.op("neq", BV1, captured_a, captured_b)
    m.bad(m.op("and", BV1, both_returned, diff))
    return m


def build_reach_by_name(binary: RISCVBinary, function: str, target_pc: int) -> Model:
    return build_reach(binary, ReachSpec(function=function, target_pc=target_pc))
