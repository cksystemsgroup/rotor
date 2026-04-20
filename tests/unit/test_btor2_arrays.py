"""BTOR2 array sort + read/write printer tests (M6)."""

from rotor.btor2.nodes import ArraySort, Model, Sort
from rotor.btor2.printer import to_text


def test_array_sort_declared_and_cached() -> None:
    m = Model()
    bv64 = Sort(64)
    bv8 = Sort(8)
    a = ArraySort(index=bv64, element=bv8)
    sid_a = m.array_sort_id(bv64, bv8)
    sid_b = m.array_sort_id(bv64, bv8)
    assert sid_a == sid_b                        # hash-consed


def test_array_state_with_write_chain_init() -> None:
    m = Model()
    bv64 = Sort(64)
    bv8 = Sort(8)
    mem_sort = ArraySort(index=bv64, element=bv8)
    base = m.state(mem_sort, "base")
    mem = m.state(mem_sort, "mem")
    addr0 = m.const(bv64, 0x1000)
    byte0 = m.const(bv8, 0x42)
    init_expr = m.write(base, addr0, byte0)
    m.init(mem, init_expr)
    m.next(mem, mem)
    m.next(base, base)

    text = to_text(m)
    assert "sort array " in text                 # declared
    assert " state " in text and " mem" in text
    assert " write " in text
    assert " init " in text


def test_read_node_has_element_sort() -> None:
    m = Model()
    bv64 = Sort(64)
    bv8 = Sort(8)
    mem_sort = ArraySort(index=bv64, element=bv8)
    mem = m.state(mem_sort, "mem")
    addr = m.const(bv64, 0x20)
    b = m.read(mem, addr)
    assert b.sort == bv8
    text = to_text(m)
    assert " read " in text


def test_write_node_preserves_array_sort() -> None:
    m = Model()
    bv64 = Sort(64)
    bv8 = Sort(8)
    mem_sort = ArraySort(index=bv64, element=bv8)
    mem = m.state(mem_sort, "mem")
    addr = m.const(bv64, 0x20)
    byte = m.const(bv8, 0xAB)
    new_mem = m.write(mem, addr, byte)
    assert new_mem.sort == mem_sort
