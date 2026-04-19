from pathlib import Path

from rotor.binary import RISCVBinary

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "add2.elf"


def test_loads_riscv_elf() -> None:
    with RISCVBinary(FIXTURE) as b:
        assert b.is_64bit
        assert b.entry != 0


def test_functions_and_instructions() -> None:
    with RISCVBinary(FIXTURE) as b:
        fns = b.functions
        assert "add2" in fns
        assert "sign" in fns
        add2 = b.function("add2")
        words = [i.word for i in b.instructions(add2)]
        # addw a0, a0, a1   ; ret
        assert words == [0x00B5053B, 0x00008067]
