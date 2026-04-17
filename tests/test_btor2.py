"""Tests for the BTOR2 node DAG, builder, printer, and parser."""

from __future__ import annotations

from rotor.btor2 import BTOR2Builder, BTOR2Printer, NodeDAG, parse_btor2


def test_sort_intern_deduplicates() -> None:
    dag = NodeDAG()
    a = dag.intern_sort("bitvec", width=8)
    b = dag.intern_sort("bitvec", width=8)
    assert a is b


def test_builder_basic_operators() -> None:
    b = BTOR2Builder()
    bv8 = b.bitvec(8)
    x = b.state(bv8, "x")
    one = b.one(bv8)
    x_plus_one = b.add(x, one, "increment")
    b.next(bv8, x, x_plus_one)
    text = BTOR2Printer().render(b.dag)
    assert "sort bitvec 8" in text
    assert " state " in text or " state\n" in text or " state " in text + "\n"
    assert " add " in text
    assert " one " in text
    assert " next " in text


def test_structural_sharing() -> None:
    b = BTOR2Builder()
    bv8 = b.bitvec(8)
    one_a = b.one(bv8)
    one_b = b.one(bv8)
    assert one_a is one_b

    x = b.state(bv8, "x")
    y = b.state(bv8, "y")
    # states are not deduped even if symbol matches
    assert x is not y

    # But a plain add of the same operands is.
    p = b.add(x, one_a, "p")
    q = b.add(x, one_a, "q")
    assert p is q


def test_printer_round_trip() -> None:
    b = BTOR2Builder()
    bv1 = b.bitvec(1)
    bv4 = b.bitvec(4)
    flag = b.state(bv1, "flag")
    counter = b.state(bv4, "counter")
    one_bit = b.one(bv1)
    b.init(bv1, flag, b.zero(bv1))
    b.next(bv1, flag, one_bit)
    b.next(bv4, counter, b.add(counter, b.one(bv4)))
    b.bad(flag, "flag-set")
    text = BTOR2Printer().render(b.dag)
    dag2 = parse_btor2(text)
    text2 = BTOR2Printer().render(dag2)
    assert text.count("\n") == text2.count("\n")


def test_parser_minimal() -> None:
    text = (
        "1 sort bitvec 1\n"
        "2 sort bitvec 8\n"
        "3 zero 2\n"
        "4 one 2\n"
        "5 state 2 x\n"
        "6 add 2 5 4\n"
        "7 next 2 5 6\n"
        "8 eq 1 5 3\n"
        "9 bad 1 8 x-is-zero\n"
    )
    dag = parse_btor2(text)
    ops = {n.op for n in dag.nodes()}
    assert ops == {"zero", "one", "state", "add", "next", "eq", "bad"}
    bads = [n for n in dag.nodes() if n.op == "bad"]
    assert len(bads) == 1
    assert bads[0].symbol == "x-is-zero"
