"""M1 shipping gate: can_reach answers end-to-end on add2.elf."""

from pathlib import Path

from rotor import RotorAPI

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "add2.elf"


def test_reach_ret_of_add2() -> None:
    with RotorAPI(FIXTURE, default_bound=4) as api:
        fn = api.binary.function("add2")
        # The ret instruction sits at fn.start + 4 (after addw).
        ret_pc = fn.start + 4
        r = api.can_reach(function="add2", target_pc=ret_pc)
        assert r.verdict == "reachable"
        assert r.step == 1        # one step from entry lands on the ret


def test_reach_entry_is_trivially_reachable() -> None:
    with RotorAPI(FIXTURE, default_bound=0) as api:
        fn = api.binary.function("add2")
        r = api.can_reach(function="add2", target_pc=fn.start)
        assert r.verdict == "reachable"
        assert r.step == 0


def test_unreachable_address_within_bound() -> None:
    # Bound=1 means only pc_0 (= fn.start) and pc_1 (= fn.start+4 after
    # addw) are considered. Any other PC is unreachable. At bound >= 2
    # the `ret` instruction dispatches to x1 & ~1, which is free, so any
    # even-aligned PC becomes reachable — the bound is what makes this
    # test observable.
    with RotorAPI(FIXTURE, default_bound=1) as api:
        fn = api.binary.function("add2")
        r = api.can_reach(function="add2", target_pc=fn.start + 8)
        assert r.verdict == "unreachable"


def test_reach_return_path_of_sign() -> None:
    # sign() has two reachable return PCs (two rets). Both should be reachable.
    with RotorAPI(FIXTURE, default_bound=10) as api:
        fn = api.binary.function("sign")
        # The two rets are at offsets 0x0C and 0x14 from fn.start.
        ret1 = fn.start + 0x0C
        ret2 = fn.start + 0x14
        r1 = api.can_reach(function="sign", target_pc=ret1)
        r2 = api.can_reach(function="sign", target_pc=ret2)
        assert r1.verdict == "reachable"
        assert r2.verdict == "reachable"
