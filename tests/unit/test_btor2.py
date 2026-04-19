from rotor.btor2.nodes import Model, Sort
from rotor.btor2.printer import to_text


def test_minimal_printer_roundtrip() -> None:
    m = Model()
    bv8 = Sort(8)
    x = m.state(bv8, "x")
    three = m.const(bv8, 3)
    nxt = m.op("add", bv8, x, three)
    m.next(x, nxt)
    m.init(x, m.const(bv8, 0))
    text = to_text(m)
    # Contains a sort line for width 8, a state line named "x", a constd 3, an add, init, next.
    assert "sort bitvec 8" in text
    assert " state " in text and " x" in text
    assert " constd " in text
    assert " add " in text
    assert " init " in text
    assert " next " in text
