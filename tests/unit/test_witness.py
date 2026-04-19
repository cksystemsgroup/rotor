from pathlib import Path

from rotor.binary import RISCVBinary
from rotor.witness import simulate

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "add2.elf"


def test_add2_two_steps_deterministic() -> None:
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("add2")
        # x10=a0=3, x11=a1=4, ra=x1=some return address.
        steps = simulate(b, fn, {"x10": 3, "x11": 4, "x1": 0xFEED}, max_steps=2)

    assert [s.pc for s in steps] == [fn.start, fn.start + 4, 0xFEEC]
    # addw a0, a0, a1 at step 0: after executing, x10 becomes 7 (seen at step 1).
    assert steps[1].registers[10] == 7
    # x11 must not have been clobbered.
    assert steps[1].registers[11] == 4


def test_sign_positive_path() -> None:
    # x10 = 5  ->  blt x0, a0 taken  ->  addi a0, x0, 1  ->  ret
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("sign")
        steps = simulate(b, fn, {"x10": 5, "x1": 0xABCD}, max_steps=3)

    # pc trajectory: 0x100b8 (blt) -> 0x100c8 (li 1) -> 0x100cc (ret) -> 0xABCC (outside fn)
    pcs = [s.pc for s in steps]
    assert pcs == [fn.start, fn.start + 0x10, fn.start + 0x14, 0xABCC]
    assert steps[2].registers[10] == 1


def test_sign_negative_path() -> None:
    # x10 = -3 (two's-complement) -> fall through -> snez, neg, ret
    neg3 = (-3) & ((1 << 64) - 1)
    with RISCVBinary(FIXTURE) as b:
        fn = b.function("sign")
        steps = simulate(b, fn, {"x10": neg3, "x1": 0x2222}, max_steps=4)

    pcs = [s.pc for s in steps]
    assert pcs == [fn.start, fn.start + 4, fn.start + 8, fn.start + 12, 0x2222]
    # After snez + neg, a0 should be -1 (two's-complement 64-bit).
    assert steps[3].registers[10] == ((-1) & ((1 << 64) - 1))
