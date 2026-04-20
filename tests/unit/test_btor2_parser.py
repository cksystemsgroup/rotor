"""Round-trip + diagnostics tests for the BTOR2 parser."""

from pathlib import Path

from rotor.binary import RISCVBinary
from rotor.btor2.builder import build_reach
from rotor.btor2.nodes import ArraySort, Model, Sort
from rotor.btor2.parser import from_text
from rotor.btor2.printer import to_text
from rotor.ir.spec import ReachSpec

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


# ---------------------------------------------------------------- round-trip
#
# The strongest correctness signal for Phase 1: every Model rotor's emitter
# produces must parse back into a Model whose re-emission is byte-identical.


def _assert_roundtrip(model: Model) -> None:
    original = to_text(model)
    result = from_text(original)
    assert result.ok, f"unexpected diagnostics: {result.diagnostics}"
    assert to_text(result.model) == original


def test_roundtrip_bitvec_only() -> None:
    m = Model()
    bv8 = Sort(8)
    x = m.state(bv8, "x")
    three = m.const(bv8, 3)
    nxt = m.op("add", bv8, x, three)
    m.next(x, nxt)
    m.init(x, m.const(bv8, 0))
    m.bad(m.op("eq", Sort(1), x, three))
    _assert_roundtrip(m)


def test_roundtrip_all_supported_ops() -> None:
    m = Model()
    bv8 = Sort(8)
    bv1 = Sort(1)
    a = m.input(bv8, "a")
    b = m.input(bv8, "b")
    m.op("add", bv8, a, b)
    m.op("sub", bv8, a, b)
    m.op("and", bv8, a, b)
    m.op("or", bv8, a, b)
    m.op("xor", bv8, a, b)
    m.op("eq", bv1, a, b)
    m.op("neq", bv1, a, b)
    m.op("ult", bv1, a, b)
    m.op("slt", bv1, a, b)
    m.op("sll", bv8, a, b)
    m.op("srl", bv8, a, b)
    m.op("sra", bv8, a, b)
    m.op("concat", Sort(16), a, b)
    _assert_roundtrip(m)


def test_roundtrip_slice_ext_ite() -> None:
    m = Model()
    big = m.input(Sort(64), "inp")
    lo = m.slice(big, 31, 0)
    uz = m.uext(lo, 32)
    sz = m.sext(lo, 32)
    c = m.const(Sort(1), 1)
    m.ite(c, uz, sz)
    _assert_roundtrip(m)


def test_roundtrip_array_read_write() -> None:
    m = Model()
    BV8, BV64 = Sort(8), Sort(64)
    MEM = ArraySort(index=BV64, element=BV8)
    mem = m.state(MEM, "mem")
    addr = m.const(BV64, 0x1000)
    val = m.const(BV8, 0xAB)
    w = m.write(mem, addr, val)
    m.init(mem, w)
    r = m.read(w, addr)
    m.bad(m.op("eq", Sort(1), r, m.const(BV8, 0)))
    _assert_roundtrip(m)


def test_roundtrip_real_build_reach_add2() -> None:
    """End-to-end: a real rotor-emitted Model must round-trip byte-for-byte."""
    with RISCVBinary(FIXTURES / "add2.elf") as b:
        fn = b.function("add2")
        spec = ReachSpec(function="add2", target_pc=fn.start + 4)
        _assert_roundtrip(build_reach(b, spec))


def test_roundtrip_real_build_reach_memops() -> None:
    """Exercises the array memory model end-to-end through the parser."""
    with RISCVBinary(FIXTURES / "memops.elf") as b:
        fn = b.function("roundtrip")
        spec = ReachSpec(function="roundtrip", target_pc=fn.start + 4)
        _assert_roundtrip(build_reach(b, spec))


# ---------------------------------------------------------------- golden
#
# Hand-written snippets verify surface-level behaviour: comments, blank
# lines, and mapping from external ids to the Model's dense ids.


def test_comments_and_blank_lines_are_skipped() -> None:
    src = (
        "; header comment\n"
        "\n"
        "1 sort bitvec 8 ; inline comment\n"
        "   \n"
        "2 constd 1 42\n"
    )
    r = from_text(src)
    assert r.ok
    assert len(r.model.nodes) == 2
    assert to_text(r.model) == "1 sort bitvec 8\n2 constd 1 42\n"


def test_sparse_external_ids_are_renumbered_densely() -> None:
    src = (
        "10 sort bitvec 8\n"
        "20 constd 10 7\n"
        "30 constd 10 11\n"
        "40 add 10 20 30\n"
    )
    r = from_text(src)
    assert r.ok
    # Model re-numbers ids densely starting at 1, both for the sort line
    # and for every operand reference.
    assert to_text(r.model) == (
        "1 sort bitvec 8\n"
        "2 constd 1 7\n"
        "3 constd 1 11\n"
        "4 add 1 2 3\n"
    )


def test_state_and_next_and_init() -> None:
    src = (
        "1 sort bitvec 8\n"
        "2 state 1 counter\n"
        "3 constd 1 0\n"
        "4 init 1 2 3\n"
        "5 constd 1 1\n"
        "6 add 1 2 5\n"
        "7 next 1 2 6\n"
    )
    r = from_text(src)
    assert r.ok
    text = to_text(r.model)
    assert "state 1 counter" in text
    assert "init 1 2 3" in text
    assert "next 1 2 6" in text


# ---------------------------------------------------------------- diagnostics
#
# Every malformed-line category produces exactly one diagnostic with the
# right line number; parsing does not abort.


def test_unsupported_tag_is_diagnosed_not_raised() -> None:
    r = from_text("1 sort bitvec 8\n2 frobnicate 1 1 1\n3 constd 1 5\n")
    assert not r.ok
    [diag] = r.diagnostics
    assert diag.line_no == 2
    assert "frobnicate" in diag.message
    # Subsequent valid line still parsed.
    assert any(n.kind == "const" for n in r.model.nodes)


def test_unknown_sort_ref_is_diagnosed() -> None:
    r = from_text("1 sort bitvec 8\n2 constd 99 5\n")
    assert not r.ok
    [diag] = r.diagnostics
    assert diag.line_no == 2
    assert "unknown sort id 99" in diag.message


def test_unknown_node_ref_is_diagnosed() -> None:
    # Sort ids live in a separate namespace from node ids, so "1" here is
    # a node id that was never declared (the sort node is not an operand).
    r = from_text("1 sort bitvec 8\n2 constd 1 7\n3 add 1 2 77\n")
    assert not r.ok
    [diag] = r.diagnostics
    assert diag.line_no == 3
    assert "unknown node id 77" in diag.message


def test_bad_arity_is_diagnosed() -> None:
    r = from_text("1 sort bitvec 8\n2 slice 1 2\n")
    assert not r.ok
    assert r.diagnostics[0].line_no == 2
    assert "slice" in r.diagnostics[0].message


def test_invalid_width_is_diagnosed() -> None:
    r = from_text("1 sort bitvec 0\n")
    assert not r.ok
    assert "width must be positive" in r.diagnostics[0].message


def test_model_assertion_becomes_diagnostic() -> None:
    # bad takes a bv1; a bv8 expression must produce a consistency error
    # rather than bubbling an AssertionError up to the caller.
    src = "1 sort bitvec 8\n2 constd 1 3\n3 bad 2\n"
    r = from_text(src)
    assert not r.ok
    assert r.diagnostics[0].line_no == 3
    assert "consistency" in r.diagnostics[0].message


def test_multiple_errors_are_all_collected() -> None:
    src = (
        "1 sort bitvec 8\n"
        "2 frobnicate 1\n"          # unsupported tag
        "3 constd 99 5\n"           # unknown sort
        "4 add 1 2 99\n"            # unknown node ref (after 2 was skipped)
        "5 constd 1 7\n"            # still parses
    )
    r = from_text(src)
    assert not r.ok
    lines = [d.line_no for d in r.diagnostics]
    assert lines == [2, 3, 4]
    # Valid lines produce nodes; errored ones don't.
    kinds = [n.kind for n in r.model.nodes]
    assert kinds.count("const") == 1


def test_empty_input_parses_to_empty_model() -> None:
    r = from_text("")
    assert r.ok
    assert r.model.nodes == ()


def test_from_path(tmp_path: Path) -> None:
    f = tmp_path / "t.btor2"
    f.write_text("1 sort bitvec 8\n2 constd 1 42\n")
    from rotor.btor2.parser import from_path

    r = from_path(f)
    assert r.ok
    assert len(r.model.nodes) == 2


# ---------------------------------------------------------------- HWMCC extensions (Phase 3)


def _const_values(r) -> list[int]:
    return [n.operands[0] for n in r.model.nodes if n.kind == "const"]


def test_zero_one_ones_normalize_to_constd() -> None:
    src = (
        "1 sort bitvec 4\n"
        "2 zero 1\n"
        "3 one 1\n"
        "4 ones 1\n"
    )
    r = from_text(src)
    assert r.ok
    assert _const_values(r) == [0, 1, 15]
    # Output always re-emits as constd.
    text = to_text(r.model)
    assert " zero " not in text and " one " not in text and " ones " not in text
    assert text.count(" constd ") == 3


def test_const_binary_and_consth_hex() -> None:
    src = (
        "1 sort bitvec 8\n"
        "2 const 1 10101010\n"
        "3 consth 1 ff\n"
        "4 consth 1 0A\n"
    )
    r = from_text(src)
    assert r.ok, r.diagnostics
    assert _const_values(r) == [0b10101010, 0xFF, 0x0A]


def test_const_rejects_non_binary() -> None:
    r = from_text("1 sort bitvec 4\n2 const 1 123\n")
    assert not r.ok
    assert "base-2" in r.diagnostics[0].message


def test_consth_rejects_non_hex() -> None:
    r = from_text("1 sort bitvec 4\n2 consth 1 xyz\n")
    assert not r.ok
    assert "base-16" in r.diagnostics[0].message


def test_constraint_emits_and_is_honoured_by_z3() -> None:
    # A simple reachability: bad expr can be satisfied only if constraint
    # permits it. With a contradictory constraint, the bad state must
    # become unreachable at any bound.
    from rotor.solvers.z3bv import Z3BMC

    src = (
        "1 sort bitvec 1\n"
        "2 state 1 s\n"
        "3 constd 1 0\n"
        "4 init 1 2 3\n"
        "5 next 1 2 2\n"            # s stays 0 forever
        "6 constd 1 1\n"
        "7 eq 1 2 6\n"              # s == 1 (impossible due to init+next)
        "8 bad 7\n"
    )
    r = from_text(src)
    assert r.ok, r.diagnostics
    # Control case: unreachable already (init 0, next = self).
    base = Z3BMC().check_reach(r.model, bound=5)
    assert base.verdict == "unreachable"

    # Add a constraint that is always false. The solver path must assert
    # the constraint and still conclude unreachable (not crash).
    src_with_c = src + "9 constd 1 0\n10 constraint 9\n11 bad 6\n"
    r2 = from_text(src_with_c)
    assert r2.ok, r2.diagnostics
    res = Z3BMC().check_reach(r2.model, bound=3)
    # Constraint is always false, so the whole path is infeasible ->
    # no reachable bad, i.e. unreachable.
    assert res.verdict == "unreachable"


def test_output_justice_fair_are_warnings_not_errors() -> None:
    src = (
        "1 sort bitvec 8\n"
        "2 constd 1 42\n"
        "3 output 1 2\n"
        "4 justice 1 2\n"
        "5 fair 1 2\n"
    )
    r = from_text(src)
    # No errors: warnings only.
    assert r.ok
    severities = [d.severity for d in r.diagnostics]
    assert severities == ["warning", "warning", "warning"]
    assert [d.line_no for d in r.diagnostics] == [3, 4, 5]
    # The warned lines do not add nodes.
    assert len([n for n in r.model.nodes if n.kind == "const"]) == 1


def test_trailing_symbol_is_dropped_on_op_lines() -> None:
    src = (
        "1 sort bitvec 8\n"
        "2 constd 1 3\n"
        "3 constd 1 5\n"
        "4 add 1 2 3 my_symbol\n"
        "5 not 1 2 neg_const\n"
    )
    r = from_text(src)
    assert r.ok, r.diagnostics
    ops = [n for n in r.model.nodes if n.kind == "op"]
    assert [o.opname for o in ops] == ["add", "not"]


def test_trailing_symbol_on_constraint_and_bad() -> None:
    src = (
        "1 sort bitvec 1\n"
        "2 state 1 s\n"
        "3 constd 1 1\n"
        "4 eq 1 2 3\n"
        "5 bad 4 safety_label\n"
        "6 constraint 4 invariant_label\n"
    )
    r = from_text(src)
    assert r.ok, r.diagnostics
    kinds = [n.kind for n in r.model.nodes]
    assert "bad" in kinds and "constraint" in kinds


def test_extended_op_set_arity_enforced() -> None:
    # Unary must get exactly one operand; binary exactly two.
    r1 = from_text("1 sort bitvec 8\n2 not 1\n")
    assert not r1.ok
    r2 = from_text("1 sort bitvec 8\n2 constd 1 3\n3 mul 1 2\n")
    assert not r2.ok
    # Happy path.
    r3 = from_text(
        "1 sort bitvec 8\n"
        "2 constd 1 3\n"
        "3 constd 1 5\n"
        "4 mul 1 2 3\n"
        "5 not 1 2\n"
        "6 redand 1 2\n"          # width mismatch; Model doesn't validate -> accepted
    )
    assert r3.ok, r3.diagnostics


def test_extended_ops_roundtrip_as_written() -> None:
    """New op names survive round-trip: parsed, stored in Model.op, printed back."""
    src = (
        "1 sort bitvec 8\n"
        "2 constd 1 1\n"
        "3 constd 1 2\n"
        "4 mul 1 2 3\n"
        "5 udiv 1 2 3\n"
        "6 ugte 1 2 3\n"
        "7 rol 1 2 3\n"
        "8 not 1 2\n"
    )
    r = from_text(src)
    assert r.ok
    text = to_text(r.model)
    for op in ("mul", "udiv", "ugte", "rol", "not"):
        assert f" {op} " in text
