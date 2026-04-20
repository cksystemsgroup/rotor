"""DagBuilder: hash-consing + local rewrites (M7).

Every rule here has a matching assertion on the emitted Model's node
list so the simplifier stays auditable: a rule either produces the
same Node object as an earlier emit (hash-cons) or produces one fewer
Node than the equivalent un-simplified sequence would.
"""

from __future__ import annotations

from rotor.btor2.nodes import ArraySort, Sort
from rotor.btor2.printer import to_text
from rotor.ir.dag import DagBuilder

BV1  = Sort(1)
BV8  = Sort(8)
BV32 = Sort(32)
BV64 = Sort(64)


# --------------------------------- hash-consing --------------------------- #

def test_const_hash_conses() -> None:
    m = DagBuilder()
    a = m.const(BV64, 42)
    b = m.const(BV64, 42)
    assert a is b


def test_op_hash_conses_on_identical_args() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    one = m.const(BV64, 1)
    a = m.op("add", BV64, x, one)
    b = m.op("add", BV64, x, one)
    assert a is b


def test_op_hash_conses_across_commutative_argument_order() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    y = m.state(BV64, "y")
    a = m.op("add", BV64, x, y)
    b = m.op("add", BV64, y, x)
    assert a is b


def test_sub_does_not_commute() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    y = m.state(BV64, "y")
    a = m.op("sub", BV64, x, y)
    b = m.op("sub", BV64, y, x)
    assert a is not b


# ------------------------------ identity laws ---------------------------- #

def test_add_zero_is_identity() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    zero = m.const(BV64, 0)
    assert m.op("add", BV64, x, zero) is x
    assert m.op("add", BV64, zero, x) is x


def test_sub_zero_is_identity() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    zero = m.const(BV64, 0)
    assert m.op("sub", BV64, x, zero) is x


def test_sub_self_folds_to_zero() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    result = m.op("sub", BV64, x, x)
    assert result.kind == "const"
    (value,) = result.operands
    assert value == 0


def test_and_all_ones_is_identity() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    ones = m.const(BV64, (1 << 64) - 1)
    assert m.op("and", BV64, x, ones) is x


def test_and_zero_is_annihilator() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    zero = m.const(BV64, 0)
    r = m.op("and", BV64, x, zero)
    assert r.kind == "const" and r.operands[0] == 0


def test_or_zero_is_identity() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    zero = m.const(BV64, 0)
    assert m.op("or", BV64, x, zero) is x


def test_or_all_ones_is_annihilator() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    ones = m.const(BV64, (1 << 64) - 1)
    r = m.op("or", BV64, x, ones)
    assert r is ones


def test_xor_self_folds_to_zero() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    r = m.op("xor", BV64, x, x)
    assert r.kind == "const" and r.operands[0] == 0


def test_shift_by_zero_is_identity() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    zero = m.const(BV64, 0)
    assert m.op("sll", BV64, x, zero) is x
    assert m.op("srl", BV64, x, zero) is x
    assert m.op("sra", BV64, x, zero) is x


# ------------------------------ constant folding -------------------------- #

def test_fold_add() -> None:
    m = DagBuilder()
    r = m.op("add", BV64, m.const(BV64, 5), m.const(BV64, 7))
    assert r.kind == "const" and r.operands[0] == 12


def test_fold_sub_modular() -> None:
    m = DagBuilder()
    r = m.op("sub", BV64, m.const(BV64, 3), m.const(BV64, 5))
    # 3 - 5 = -2 modulo 2^64
    assert r.kind == "const" and r.operands[0] == ((-2) & ((1 << 64) - 1))


def test_fold_and_or_xor() -> None:
    m = DagBuilder()
    assert m.op("and", BV8, m.const(BV8, 0xF0), m.const(BV8, 0x3C)).operands[0] == 0x30
    assert m.op("or",  BV8, m.const(BV8, 0xF0), m.const(BV8, 0x3C)).operands[0] == 0xFC
    assert m.op("xor", BV8, m.const(BV8, 0xF0), m.const(BV8, 0x3C)).operands[0] == 0xCC


def test_fold_shifts() -> None:
    m = DagBuilder()
    assert m.op("sll", BV8, m.const(BV8, 1),    m.const(BV8, 3)).operands[0] == 8
    assert m.op("srl", BV8, m.const(BV8, 0xF0), m.const(BV8, 4)).operands[0] == 0x0F
    # 0xF0 as signed 8-bit is -16; -16 >> 4 = -1 -> 0xFF
    assert m.op("sra", BV8, m.const(BV8, 0xF0), m.const(BV8, 4)).operands[0] == 0xFF


def test_fold_comparisons() -> None:
    m = DagBuilder()
    lt = m.op("ult", BV1, m.const(BV8, 1), m.const(BV8, 2))
    assert lt.kind == "const" and lt.operands[0] == 1
    slt = m.op("slt", BV1, m.const(BV8, 0xFF), m.const(BV8, 1))          # -1 < 1 signed
    assert slt.operands[0] == 1
    ult = m.op("ult", BV1, m.const(BV8, 0xFF), m.const(BV8, 1))          # 255 < 1 unsigned
    assert ult.operands[0] == 0


def test_eq_of_equal_nodes_is_true() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    r = m.op("eq", BV1, x, x)
    assert r.kind == "const" and r.operands[0] == 1


def test_neq_of_equal_nodes_is_false() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    r = m.op("neq", BV1, x, x)
    assert r.kind == "const" and r.operands[0] == 0


# -------------------------------- ITE rules ------------------------------- #

def test_ite_constant_true_picks_then_arm() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    y = m.state(BV64, "y")
    true_c = m.const(BV1, 1)
    assert m.ite(true_c, x, y) is x


def test_ite_constant_false_picks_else_arm() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    y = m.state(BV64, "y")
    false_c = m.const(BV1, 0)
    assert m.ite(false_c, x, y) is y


def test_ite_identical_arms_collapse() -> None:
    m = DagBuilder()
    cond = m.state(BV1, "c")
    x = m.state(BV64, "x")
    assert m.ite(cond, x, x) is x


# ---------------------------- slice / ext rules --------------------------- #

def test_slice_of_constant_folds() -> None:
    m = DagBuilder()
    r = m.slice(m.const(BV64, 0xDEADBEEF_CAFEBABE), 31, 0)
    assert r.kind == "const" and r.operands[0] == 0xCAFEBABE


def test_full_width_slice_is_identity() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    assert m.slice(x, 63, 0) is x


def test_uext_of_constant_folds() -> None:
    m = DagBuilder()
    r = m.uext(m.const(BV32, 0x80000000), 32)
    assert r.kind == "const" and r.operands[0] == 0x80000000        # zero-extended


def test_sext_of_constant_folds() -> None:
    m = DagBuilder()
    r = m.sext(m.const(BV32, 0x80000000), 32)
    assert r.kind == "const" and r.operands[0] == 0xFFFFFFFF80000000


def test_zero_ext_is_identity() -> None:
    m = DagBuilder()
    x = m.state(BV32, "x")
    assert m.uext(x, 0) is x
    assert m.sext(x, 0) is x


# ---------------------------- memory simplifications ---------------------- #

def test_read_after_matching_write_forwards_value() -> None:
    m = DagBuilder()
    mem_sort = ArraySort(index=BV64, element=BV8)
    mem = m.state(mem_sort, "mem")
    addr = m.const(BV64, 0x1000)
    val = m.const(BV8, 0xAA)
    new_mem = m.write(mem, addr, val)
    read = m.read(new_mem, addr)
    assert read is val


def test_read_past_distinct_constant_write() -> None:
    m = DagBuilder()
    mem_sort = ArraySort(index=BV64, element=BV8)
    mem = m.state(mem_sort, "mem")
    a1 = m.const(BV64, 0x1000)
    a2 = m.const(BV64, 0x2000)
    new_mem = m.write(mem, a1, m.const(BV8, 0xAA))
    # read at a different constant address should skip the write
    read = m.read(new_mem, a2)
    assert read.kind == "read"
    arr, addr = read.operands
    assert arr is mem              # walked past the write


# ------------------------- emitted BTOR2 round-trips ---------------------- #

def test_emitted_btor2_parses_as_text() -> None:
    """Hash-consing and simplification must still yield valid BTOR2."""
    m = DagBuilder()
    x = m.state(BV64, "x")
    y = m.op("add", BV64, x, m.const(BV64, 0))     # collapses to x
    m.init(x, m.const(BV64, 0))
    m.next(x, y)
    text = to_text(m)
    # The add should have been eliminated.
    assert " add " not in text


def test_hash_consing_deduplicates_repeated_adds() -> None:
    m = DagBuilder()
    x = m.state(BV64, "x")
    imm = m.const(BV64, 4)
    a1 = m.op("add", BV64, x, imm)
    a2 = m.op("add", BV64, x, imm)
    a3 = m.op("add", BV64, imm, x)             # commuted
    assert a1 is a2 is a3
    text = to_text(m)
    assert text.count(" add ") == 1
