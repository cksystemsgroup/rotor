"""Tests for load/store/ECALL semantics in the native RISC-V builder."""

from __future__ import annotations

from rotor.btor2 import BTOR2Printer, RISCVMachineBuilder
from rotor.instance import ModelConfig


def _build():
    cfg = ModelConfig(is_64bit=True, cores=1, code_start=0x1000)
    b = RISCVMachineBuilder(cfg)
    model = b.build()
    return b, model


def test_halted_latch_created() -> None:
    _, model = _build()
    assert "halted" in model.state_nodes
    assert model.state_nodes["halted"].sort.width == 1


def test_unified_memory_state() -> None:
    _, model = _build()
    assert "memory" in model.state_nodes
    assert "code" in model.state_nodes
    # Data/heap/stack are no longer separate state nodes.
    assert "data" not in model.state_nodes
    assert "heap" not in model.state_nodes
    assert "stack" not in model.state_nodes


def _comments(model) -> set[str]:
    return {n.comment for n in model.dag.nodes() if n.comment}


def test_load_instructions_present() -> None:
    _, model = _build()
    comments = _comments(model)
    for load in ("lb-enabled", "lh-enabled", "lw-enabled", "lbu-enabled",
                 "lhu-enabled", "ld-enabled", "lwu-enabled"):
        assert load in comments, f"missing {load}"


def test_store_instructions_present() -> None:
    _, model = _build()
    comments = _comments(model)
    for store in ("sb-enabled", "sh-enabled", "sw-enabled", "sd-enabled"):
        assert store in comments


def test_ecall_recognized_and_halt_wired() -> None:
    _, model = _build()
    comments = _comments(model)
    assert "ecall" in comments
    # Halt latch must have an init and a next relation.
    inits = [n for n in model.init_nodes if n.args and n.args[0] is model.state_nodes["halted"]]
    nexts = [n for n in model.next_nodes if n.args and n.args[0] is model.state_nodes["halted"]]
    assert inits and nexts


def test_memory_has_next_relation() -> None:
    _, model = _build()
    memory = model.state_nodes["memory"]
    memory_nexts = [n for n in model.next_nodes if n.args and n.args[0] is memory]
    assert len(memory_nexts) == 1, "memory should have exactly one next relation"


def test_illegal_instruction_property_gated_by_halted() -> None:
    _, model = _build()
    bads = [n for n in model.property_nodes if "illegal-instruction" in n.symbol]
    assert bads, "illegal-instruction bad property should exist"
    # The condition feeding the bad should be an AND (not-any-semantic &&
    # !halted), confirming halt-gating.
    assert bads[0].args[0].op == "and"


def test_model_round_trips() -> None:
    from rotor.btor2 import parse_btor2

    _, model = _build()
    text = BTOR2Printer().render(model.dag)
    dag2 = parse_btor2(text)
    text2 = BTOR2Printer().render(dag2)
    assert text.count("\n") == text2.count("\n")


def test_store_selects_appear_in_dispatch() -> None:
    """Each store family should contribute a 'sel-mem:<name>' ITE in the
    memory dispatch cascade."""
    _, model = _build()
    comments = _comments(model)
    for store in ("sb", "sh", "sw", "sd"):
        assert f"sel-mem:{store}" in comments


def test_rv32_excludes_ld_and_sd() -> None:
    cfg = ModelConfig(is_64bit=False, word_size=32, cores=1, code_start=0x1000)
    b = RISCVMachineBuilder(cfg)
    model = b.build()
    comments = {n.comment for n in model.dag.nodes() if n.comment}
    assert "ld-enabled" not in comments
    assert "sd-enabled" not in comments
    # The 32-bit forms must still be present.
    assert "lw-enabled" in comments
    assert "sw-enabled" in comments
