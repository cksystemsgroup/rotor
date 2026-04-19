from rotor.btor2.riscv.decoder import decode


def test_addw() -> None:
    d = decode(0x00B5053B)             # addw a0, a0, a1
    assert d is not None
    assert (d.mnem, d.rd, d.rs1, d.rs2) == ("addw", 10, 10, 11)


def test_jalr_is_ret() -> None:
    d = decode(0x00008067)             # jalr x0, ra, 0  (== ret)
    assert d is not None
    assert (d.mnem, d.rd, d.rs1, d.imm) == ("jalr", 0, 1, 0)


def test_blt_is_bgtz() -> None:
    # bgtz a0, +0x10  == blt x0, a0, +0x10
    d = decode(0x00A04863)
    assert d is not None
    assert (d.mnem, d.rs1, d.rs2, d.imm) == ("blt", 0, 10, 16)


def test_sltu_is_snez() -> None:
    d = decode(0x00A03533)             # sltu a0, x0, a0  (== snez a0, a0)
    assert d is not None
    assert (d.mnem, d.rd, d.rs1, d.rs2) == ("sltu", 10, 0, 10)


def test_sub_is_neg() -> None:
    d = decode(0x40A00533)             # sub a0, x0, a0  (== neg a0, a0)
    assert d is not None
    assert (d.mnem, d.rd, d.rs1, d.rs2) == ("sub", 10, 0, 10)


def test_addi_is_li() -> None:
    d = decode(0x00100513)             # addi a0, x0, 1  (== li a0, 1)
    assert d is not None
    assert (d.mnem, d.rd, d.rs1, d.imm) == ("addi", 10, 0, 1)


def test_unknown_returns_none() -> None:
    assert decode(0xFFFFFFFF) is None
