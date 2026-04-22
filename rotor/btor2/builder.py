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
from rotor.ir.spec import ReachSpec, VerifySpec

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


def _build_machine(
    binary: RISCVBinary,
    function_name: str,
    builder: Optional[Model],
    havoc_regs: Optional[set[int]],
) -> _Machine:
    """Build the shared machine model (regs, pc, mem, transition
    relation, entry assumptions) for a function. Verb-specific
    `build_*` callers append their own `bad` expression after."""
    fn = binary.function(function_name)
    m = builder if builder is not None else Model()
    havoc = set(havoc_regs) if havoc_regs else set()
    havoc.discard(0)                              # x0 is a constant, never havoc'd

    zero = m.const(BV64, 0)
    regs: list[Node] = [zero]
    for i in range(1, NREGS):
        if i in havoc:
            regs.append(m.input(BV64, f"x{i}"))
        else:
            regs.append(m.state(BV64, f"x{i}"))

    if 1 not in havoc:
        ra = regs[1]
        ra_masked = m.op("and", BV64, ra, m.const(BV64, 0xFFFF_FFFF_FFFF_FFFE))
        below = m.op("ult", BV1, ra_masked, m.const(BV64, fn.start))
        at_or_above = m.op("ugte", BV1, ra_masked, m.const(BV64, fn.end))
        m.constraint(m.op("or", BV1, below, at_or_above))

    pc = m.state(BV64, "pc")
    m.init(pc, m.const(BV64, fn.start))

    decoded: list[tuple] = []
    for inst in binary.instructions(fn):
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
        free_mem = m.state(MEM_SORT, "free_mem")
        mem = m.state(MEM_SORT, "mem")
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

    return _Machine(m=m, regs=regs, pc=pc, mem=mem, decoded=decoded)


def build_reach(
    binary: RISCVBinary,
    spec: ReachSpec,
    builder: Optional[Model] = None,
    havoc_regs: Optional[set[int]] = None,
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
    mc = _build_machine(binary, spec.function, builder, havoc_regs)
    # Reach obligation: pc == target.
    target = mc.m.const(BV64, spec.target_pc)
    mc.m.bad(mc.m.op("eq", BV1, mc.pc, target))
    return mc.m


def build_verify(
    binary: RISCVBinary,
    spec: "VerifySpec",
    builder: Optional[Model] = None,
    havoc_regs: Optional[set[int]] = None,
) -> Model:
    """Compile a VerifySpec into a BTOR2 Model.

    The `bad` state is defined as:

        bad = (pc at any `ret` in fn) ∧ ¬(regs[R] OP rhs)

    So `proved` means the predicate holds on every execution path
    that returns from the function; `reachable` means some input
    makes the predicate fail at a return site.

    Identifying ret PCs is static: we scan `decoded` for `jalr
    x0, x1, 0` (canonical `ret`). Functions that return through a
    different jalr encoding (indirect dispatch back to the caller)
    won't trigger, which is a current limitation; expanding to
    "any jalr that targets an outside-fn PC per the ra-constraint"
    is a natural refinement.
    """
    mc = _build_machine(binary, spec.function, builder, havoc_regs)
    m = mc.m

    # Enumerate return-instruction PCs. `ret` is the canonical `jalr
    # x0, x1, 0`; we include all jalrs that use ra as the base and
    # discard the link register, since compilers emit only that form
    # for function returns.
    ret_pcs = [
        inst.pc for inst, d in mc.decoded
        if d.mnem == "jalr" and d.rd == 0 and d.rs1 == 1 and d.imm == 0
    ]
    if not ret_pcs:
        raise ValueError(
            f"function {spec.function!r} has no ret instruction; "
            f"verify requires at least one return site"
        )

    at_ret = m.op("eq", BV1, mc.pc, m.const(BV64, ret_pcs[0]))
    for ret_pc in ret_pcs[1:]:
        at_ret = m.op("or", BV1, at_ret, m.op("eq", BV1, mc.pc, m.const(BV64, ret_pc)))

    reg_val = mc.regs[spec.register]
    rhs_const = m.const(BV64, spec.rhs & ((1 << 64) - 1))
    predicate = m.op(spec.comparison, BV1, reg_val, rhs_const)
    fails_pred = m.op("eq", BV1, predicate, m.const(BV1, 0))   # !predicate
    m.bad(m.op("and", BV1, at_ret, fails_pred))
    return m


def build_reach_by_name(binary: RISCVBinary, function: str, target_pc: int) -> Model:
    return build_reach(binary, ReachSpec(function=function, target_pc=target_pc))
