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

from rotor.binary import Function, RISCVBinary
from rotor.btor2.nodes import ArraySort, Model, Node, Sort
from rotor.btor2.riscv.decoder import decode
from rotor.btor2.riscv.isa import lower
from rotor.ir.spec import ReachSpec

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


def build_reach(binary: RISCVBinary, spec: ReachSpec) -> Model:
    fn = binary.function(spec.function)
    m = Model()

    # Register file: x0 is a constant 0; x1..x31 are free initial states.
    zero = m.const(BV64, 0)
    regs: list[Node] = [zero]
    for i in range(1, NREGS):
        regs.append(m.state(BV64, f"x{i}"))

    # PC state, initialized to the function entry.
    pc = m.state(BV64, "pc")
    entry = m.const(BV64, fn.start)
    m.init(pc, entry)

    # Memory: a fresh free base array overlaid with all ELF loadable bytes.
    # `free_mem` models the arbitrary initial contents of memory locations
    # not covered by any PT_LOAD segment (stack, heap, .bss). `mem` is the
    # program-visible memory, equal at cycle 0 to free_mem with each
    # loadable byte written at its virtual address.
    free_mem = m.state(MEM_SORT, "free_mem")
    mem = m.state(MEM_SORT, "mem")
    mem_init = free_mem
    for vaddr, byte in binary.loadable_bytes():
        addr_node = m.const(BV64, vaddr)
        byte_node = m.const(BV8, byte)
        mem_init = m.write(mem_init, addr_node, byte_node)
    m.init(mem, mem_init)

    # Build per-instruction next-state contributions.
    next_pc = pc                                  # default: stuck
    next_regs: dict[int, Node] = {i: regs[i] for i in range(1, NREGS)}
    next_mem = mem                                # default: memory unchanged

    for inst in binary.instructions(fn):
        d = decode(inst.word)
        if d is None:
            raise UnsupportedInstruction(inst.pc, inst.word)
        writes, npc, mem_write = lower(d, inst.pc, m, regs, mem)
        pc_const = m.const(BV64, inst.pc)
        here = m.op("eq", BV1, pc, pc_const)
        next_pc = m.ite(here, npc, next_pc)
        for rd, expr in writes.items():
            next_regs[rd] = m.ite(here, expr, next_regs[rd])
        if mem_write is not None:
            next_mem = m.ite(here, mem_write, next_mem)

    for i in range(1, NREGS):
        m.next(regs[i], next_regs[i])
    m.next(pc, next_pc)
    m.next(mem, next_mem)
    m.next(free_mem, free_mem)                    # free_mem is just a frozen base

    # Reach obligation: pc == target.
    target = m.const(BV64, spec.target_pc)
    m.bad(m.op("eq", BV1, pc, target))
    return m


def build_reach_by_name(binary: RISCVBinary, function: str, target_pc: int) -> Model:
    return build_reach(binary, ReachSpec(function=function, target_pc=target_pc))
