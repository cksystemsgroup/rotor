"""End-to-end reachability on memory-using fixtures (M6).

`memops.elf` exercises aligned loads, a store -> matching-load
round-trip, and the minimal non-leaf-style pattern of a caller-
supplied pointer.

`rodata.elf` exercises the .rodata path: the constant table is
written into the SMT array by the ELF segment initializer, so the
load through `auipc+addi` must recover the concrete table values.
"""

from pathlib import Path

from rotor import RotorAPI

MEMOPS = Path(__file__).resolve().parents[1] / "fixtures" / "memops.elf"
RODATA = Path(__file__).resolve().parents[1] / "fixtures" / "rodata.elf"


# ------------------------- memops.elf: load_sum ------------------------- #

def test_load_sum_ret_is_reachable() -> None:
    with RotorAPI(MEMOPS, default_bound=5) as api:
        fn = api.binary.function("load_sum")
        ret = fn.start + 0x0C                         # ret is the 4th instruction
        r = api.can_reach("load_sum", ret)
        assert r.verdict == "reachable"
        assert r.step == 3


def test_load_sum_second_lw_reached_at_step_1() -> None:
    with RotorAPI(MEMOPS, default_bound=3) as api:
        fn = api.binary.function("load_sum")
        r = api.can_reach("load_sum", fn.start + 0x04)
        assert r.verdict == "reachable"
        assert r.step == 1


# ------------------------- memops.elf: roundtrip ------------------------ #

def test_roundtrip_store_then_load_reach_ret() -> None:
    """sw + lw + slliw + ret: every pc is deterministically reachable."""
    with RotorAPI(MEMOPS, default_bound=5) as api:
        fn = api.binary.function("roundtrip")
        r = api.can_reach("roundtrip", fn.start + 0x0C)      # ret
        assert r.verdict == "reachable"
        assert r.step == 3


def test_roundtrip_unreachable_before_store() -> None:
    """Within bound 0, only the entry pc is reachable; the lw at +4
    needs at least one step."""
    with RotorAPI(MEMOPS, default_bound=0) as api:
        fn = api.binary.function("roundtrip")
        r = api.can_reach("roundtrip", fn.start + 0x04)
        assert r.verdict == "unreachable"


# ---------------------------- rodata.elf: pick -------------------------- #

def test_pick_ret_is_reachable() -> None:
    """pick() is 7 instructions; its ret lands at +0x18."""
    with RotorAPI(RODATA, default_bound=8) as api:
        fn = api.binary.function("pick")
        r = api.can_reach("pick", fn.start + 0x18)
        assert r.verdict == "reachable"
        assert r.step == 6


def test_pick_lw_reached_at_step_5() -> None:
    """The lw instruction — the one that reads the rodata table —
    should be reachable in exactly five steps from entry."""
    with RotorAPI(RODATA, default_bound=6) as api:
        fn = api.binary.function("pick")
        r = api.can_reach("pick", fn.start + 0x14)
        assert r.verdict == "reachable"
        assert r.step == 5
