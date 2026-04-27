"""Tests for the pure BTOR2 layer.

Round-trip discipline: every hand-built model passes through ``to_text``,
then back through ``from_text``, then ``to_text`` again, and the second
emission must equal the first byte-for-byte. This is the Phase 1 exit
criterion.
"""

from __future__ import annotations

import pytest

from rotor.btor2 import (
    OP_SPECS,
    ArraySort,
    BitvecSort,
    Model,
    Node,
    from_text,
    line_to_text,
    to_text,
)


def _round_trip(model: Model) -> str:
    text1 = to_text(model)
    parsed = from_text(text1)
    assert parsed.diagnostics == [], parsed.diagnostics
    text2 = to_text(parsed.model)
    assert text1 == text2
    return text1


# -- sort declarations -------------------------------------------------------


def test_bitvec_sort_round_trip():
    m = Model()
    m.add(BitvecSort(nid=1, width=1))
    m.add(BitvecSort(nid=2, width=8))
    m.add(BitvecSort(nid=3, width=64, symbol="word"))
    text = _round_trip(m)
    assert text == "1 sort bitvec 1\n2 sort bitvec 8\n3 sort bitvec 64 word\n"


def test_array_sort_round_trip():
    m = Model()
    m.add(BitvecSort(nid=1, width=64))  # index
    m.add(BitvecSort(nid=2, width=8))   # element
    m.add(ArraySort(nid=3, index_sort=1, element_sort=2, symbol="memory"))
    text = _round_trip(m)
    assert "3 sort array 1 2 memory" in text


# -- inputs and state --------------------------------------------------------


def test_input_and_state():
    m = Model()
    m.add(BitvecSort(nid=1, width=32))
    m.add(Node(nid=2, op="input", sort=1, symbol="in0"))
    m.add(Node(nid=3, op="state", sort=1, symbol="x1"))
    text = _round_trip(m)
    assert "2 input 1 in0" in text
    assert "3 state 1 x1" in text


# -- constants ---------------------------------------------------------------


@pytest.mark.parametrize("op", ["zero", "one", "ones"])
def test_constant_shorthand(op):
    m = Model()
    m.add(BitvecSort(nid=1, width=8))
    m.add(Node(nid=2, op=op, sort=1))
    text = _round_trip(m)
    assert f"2 {op} 1\n" in text


def test_const_binary():
    m = Model()
    m.add(BitvecSort(nid=1, width=4))
    m.add(Node(nid=2, op="const", sort=1, args=("1010",)))
    text = _round_trip(m)
    assert "2 const 1 1010" in text


def test_constd_decimal():
    m = Model()
    m.add(BitvecSort(nid=1, width=8))
    m.add(Node(nid=2, op="constd", sort=1, args=("42",)))
    m.add(Node(nid=3, op="constd", sort=1, args=("-1",)))
    text = _round_trip(m)
    assert "2 constd 1 42" in text
    assert "3 constd 1 -1" in text


def test_consth_hex():
    m = Model()
    m.add(BitvecSort(nid=1, width=16))
    m.add(Node(nid=2, op="consth", sort=1, args=("dead",)))
    text = _round_trip(m)
    assert "2 consth 1 dead" in text


# -- unary ops ---------------------------------------------------------------


@pytest.mark.parametrize("op", ["not", "inc", "dec", "neg", "redand", "redor", "redxor"])
def test_unary_ops(op):
    m = Model()
    m.add(BitvecSort(nid=1, width=8))
    m.add(Node(nid=2, op="input", sort=1))
    m.add(Node(nid=3, op=op, sort=1, args=(2,)))
    text = _round_trip(m)
    assert f"3 {op} 1 2\n" in text


# -- slice and extension -----------------------------------------------------


def test_slice():
    m = Model()
    m.add(BitvecSort(nid=1, width=32))
    m.add(BitvecSort(nid=2, width=8))
    m.add(Node(nid=3, op="input", sort=1))
    m.add(Node(nid=4, op="slice", sort=2, args=(3, 7, 0), symbol="low_byte"))
    text = _round_trip(m)
    assert "4 slice 2 3 7 0 low_byte" in text


@pytest.mark.parametrize("op", ["sext", "uext"])
def test_extension(op):
    m = Model()
    m.add(BitvecSort(nid=1, width=8))
    m.add(BitvecSort(nid=2, width=32))
    m.add(Node(nid=3, op="input", sort=1))
    m.add(Node(nid=4, op=op, sort=2, args=(3, 24)))
    text = _round_trip(m)
    assert f"4 {op} 2 3 24\n" in text


# -- binary ops --------------------------------------------------------------


_BINARY_OPS = [
    op for op, spec in OP_SPECS.items() if spec.has_sort and spec.extra == 2
]


@pytest.mark.parametrize("op", _BINARY_OPS)
def test_binary_ops(op):
    m = Model()
    m.add(BitvecSort(nid=1, width=8))
    m.add(Node(nid=2, op="input", sort=1))
    m.add(Node(nid=3, op="input", sort=1))
    # For "read" we want an array; cheat by using bitvec since the parser
    # doesn't typecheck (and Phase 1 is pure-text). Read still emits as
    # "<nid> read <sort> <a> <b>", which is what we test.
    m.add(Node(nid=4, op=op, sort=1, args=(2, 3)))
    text = _round_trip(m)
    assert f"4 {op} 1 2 3\n" in text


# -- ternary ops -------------------------------------------------------------


def test_ite():
    m = Model()
    m.add(BitvecSort(nid=1, width=1))
    m.add(BitvecSort(nid=2, width=8))
    m.add(Node(nid=3, op="input", sort=1))
    m.add(Node(nid=4, op="input", sort=2))
    m.add(Node(nid=5, op="input", sort=2))
    m.add(Node(nid=6, op="ite", sort=2, args=(3, 4, 5)))
    text = _round_trip(m)
    assert "6 ite 2 3 4 5" in text


def test_write():
    m = Model()
    m.add(BitvecSort(nid=1, width=64))
    m.add(BitvecSort(nid=2, width=8))
    m.add(ArraySort(nid=3, index_sort=1, element_sort=2))
    m.add(Node(nid=4, op="state", sort=3, symbol="mem"))
    m.add(Node(nid=5, op="input", sort=1))
    m.add(Node(nid=6, op="input", sort=2))
    m.add(Node(nid=7, op="write", sort=3, args=(4, 5, 6)))
    text = _round_trip(m)
    assert "7 write 3 4 5 6" in text


def test_read():
    m = Model()
    m.add(BitvecSort(nid=1, width=64))
    m.add(BitvecSort(nid=2, width=8))
    m.add(ArraySort(nid=3, index_sort=1, element_sort=2))
    m.add(Node(nid=4, op="state", sort=3, symbol="mem"))
    m.add(Node(nid=5, op="input", sort=1))
    m.add(Node(nid=6, op="read", sort=2, args=(4, 5)))
    text = _round_trip(m)
    assert "6 read 2 4 5" in text


# -- init and next -----------------------------------------------------------


def test_init_and_next():
    m = Model()
    m.add(BitvecSort(nid=1, width=8))
    m.add(Node(nid=2, op="state", sort=1, symbol="counter"))
    m.add(Node(nid=3, op="zero", sort=1))
    m.add(Node(nid=4, op="init", sort=1, args=(2, 3)))
    m.add(Node(nid=5, op="inc", sort=1, args=(2,)))
    m.add(Node(nid=6, op="next", sort=1, args=(2, 5)))
    text = _round_trip(m)
    assert "4 init 1 2 3" in text
    assert "6 next 1 2 5" in text


# -- properties: bad / constraint / fair / output ----------------------------


@pytest.mark.parametrize("op", ["bad", "constraint", "fair", "output"])
def test_sortless_properties(op):
    m = Model()
    m.add(BitvecSort(nid=1, width=1))
    m.add(Node(nid=2, op="input", sort=1))
    m.add(Node(nid=3, op=op, args=(2,), symbol=f"{op}_sym"))
    text = _round_trip(m)
    assert f"3 {op} 2 {op}_sym" in text


# -- justice: variable-length ------------------------------------------------


def test_justice_variable_length():
    m = Model()
    m.add(BitvecSort(nid=1, width=1))
    m.add(Node(nid=2, op="input", sort=1, symbol="a"))
    m.add(Node(nid=3, op="input", sort=1, symbol="b"))
    m.add(Node(nid=4, op="input", sort=1, symbol="c"))
    m.add(Node(nid=5, op="justice", args=(3, 2, 3, 4)))
    text = _round_trip(m)
    assert "5 justice 3 2 3 4" in text


# -- comments, blank lines, whitespace tolerance -----------------------------


def test_comments_and_blank_lines_ignored():
    text = (
        "; this is a comment\n"
        "\n"
        "1 sort bitvec 8 ; trailing comments are not standard, just blank lines\n"
        "    \n"
        "2 input 1\n"
    )
    # Trailing inline comment after content is *not* standard BTOR2; we
    # don't accept it. Strip it for this test and parse just comments +
    # blanks at line start.
    safe_text = (
        "; this is a comment\n"
        "\n"
        "1 sort bitvec 8\n"
        "    \n"
        "2 input 1\n"
    )
    parsed = from_text(safe_text)
    assert parsed.diagnostics == []
    assert len(parsed.model.lines) == 2
    assert isinstance(parsed.model.lines[0], BitvecSort)
    assert parsed.model.lines[1].op == "input"


# -- diagnostics: malformed input does not raise -----------------------------


def test_diagnostic_unknown_op():
    parsed = from_text("1 sort bitvec 8\n2 frobnicate 1\n")
    assert len(parsed.model.lines) == 1
    assert any("unknown op" in d.message for d in parsed.diagnostics)


def test_diagnostic_bad_nid():
    parsed = from_text("foo bar baz\n")
    assert parsed.model.lines == []
    assert any("non-integer nid" in d.message for d in parsed.diagnostics)


def test_diagnostic_too_few_tokens():
    parsed = from_text("1\n")
    assert parsed.model.lines == []
    assert any("too few tokens" in d.message for d in parsed.diagnostics)


def test_diagnostic_short_operands():
    parsed = from_text(
        "1 sort bitvec 8\n"
        "2 input 1\n"
        "3 add 1 2\n"  # missing second operand
    )
    assert len(parsed.model.lines) == 2  # add was rejected
    assert any("expected 2" in d.message for d in parsed.diagnostics)


def test_diagnostic_unknown_sort_kind():
    parsed = from_text("1 sort wibble 8\n")
    assert parsed.model.lines == []
    assert any("unknown kind" in d.message for d in parsed.diagnostics)


# -- printer -----------------------------------------------------------------


def test_empty_model_emits_empty_string():
    assert to_text(Model()) == ""


def test_line_to_text_no_trailing_newline():
    line = BitvecSort(nid=1, width=8, symbol="byte")
    assert line_to_text(line) == "1 sort bitvec 8 byte"


# -- end-to-end model: small counter circuit ---------------------------------


def test_counter_model_round_trip():
    """A complete tiny model: an 8-bit counter with a safety property."""

    m = Model()
    m.add(BitvecSort(nid=1, width=1))
    m.add(BitvecSort(nid=2, width=8))
    m.add(Node(nid=3, op="state", sort=2, symbol="counter"))
    m.add(Node(nid=4, op="zero", sort=2))
    m.add(Node(nid=5, op="init", sort=2, args=(3, 4)))
    m.add(Node(nid=6, op="one", sort=2))
    m.add(Node(nid=7, op="add", sort=2, args=(3, 6)))
    m.add(Node(nid=8, op="next", sort=2, args=(3, 7)))
    m.add(Node(nid=9, op="ones", sort=2))
    m.add(Node(nid=10, op="eq", sort=1, args=(3, 9)))
    m.add(Node(nid=11, op="bad", args=(10,), symbol="counter_saturates"))

    text = _round_trip(m)

    expected = (
        "1 sort bitvec 1\n"
        "2 sort bitvec 8\n"
        "3 state 2 counter\n"
        "4 zero 2\n"
        "5 init 2 3 4\n"
        "6 one 2\n"
        "7 add 2 3 6\n"
        "8 next 2 3 7\n"
        "9 ones 2\n"
        "10 eq 1 3 9\n"
        "11 bad 10 counter_saturates\n"
    )
    assert text == expected
