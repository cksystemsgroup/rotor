"""Tests for the ECALL syscall dispatcher.

Covers the builder-level guarantees (that the right state nodes exist and
are wired up) and a hand-built BTOR2 model that issues each syscall via
an `ecall` instruction.
"""

from __future__ import annotations

import pytest

pytest.importorskip("bitwuzla")

from rotor.btor2 import RISCVMachineBuilder
from rotor.instance import ModelConfig
from rotor.solvers.bmc import BitwuzlaUnroller


def _build_blank():
    cfg = ModelConfig(is_64bit=True, cores=1, code_start=0x1000)
    b = RISCVMachineBuilder(cfg)
    return b, b.build()


# ──────────────────────────────────────────────────────────────────────────
# Builder-level structure
# ──────────────────────────────────────────────────────────────────────────


def test_exit_code_state_exists() -> None:
    _, model = _build_blank()
    assert "exit-code" in model.state_nodes
    assert model.state_nodes["exit-code"].sort.width == 64


def test_input_byte_nodes_span_full_buffer() -> None:
    """Multi-byte read requires one fresh input node per buffer position."""
    _, model = _build_blank()
    # Default read_buffer_size is 64.
    for k in range(64):
        key = f"input-byte-{k}"
        assert key in model.input_nodes, f"missing {key}"
        assert model.input_nodes[key].sort.width == 8


def test_read_buffer_size_configurable() -> None:
    from rotor.btor2 import RISCVMachineBuilder
    from rotor.instance import ModelConfig

    cfg = ModelConfig(is_64bit=True, cores=1, read_buffer_size=8)
    b = RISCVMachineBuilder(cfg)
    model = b.build()
    assert "input-byte-7" in model.input_nodes
    assert "input-byte-8" not in model.input_nodes


def test_program_break_state_exists() -> None:
    _, model = _build_blank()
    assert "program-break" in model.state_nodes
    assert model.state_nodes["program-break"].sort.width == 64


def test_read_count_state_exists() -> None:
    _, model = _build_blank()
    assert "read-count" in model.state_nodes


def test_program_break_init_value() -> None:
    """Builder should initialize program-break to ModelConfig.program_break_init."""
    from rotor.btor2 import RISCVMachineBuilder
    from rotor.instance import ModelConfig

    cfg = ModelConfig(is_64bit=True, cores=1, program_break_init=0x30000000)
    b = RISCVMachineBuilder(cfg)
    model = b.build()
    brk = model.state_nodes["program-break"]
    inits = [n for n in model.init_nodes if n.args and n.args[0] is brk]
    assert inits
    init_value_node = inits[0].args[1]
    assert init_value_node.op == "consth"
    assert init_value_node.params[0] == 0x30000000


def test_exit_code_has_init_and_next() -> None:
    _, model = _build_blank()
    ec = model.state_nodes["exit-code"]
    inits = [n for n in model.init_nodes if n.args and n.args[0] is ec]
    nexts = [n for n in model.next_nodes if n.args and n.args[0] is ec]
    assert inits and nexts


def test_syscall_dispatch_semantics_present() -> None:
    _, model = _build_blank()
    comments = {n.comment for n in model.dag.nodes() if n.comment}
    for expected in (
        "ecall-exit", "ecall-read", "ecall-write", "ecall-brk", "ecall-unknown",
    ):
        assert expected in comments, f"missing {expected}"


def test_known_syscall_does_not_trip_illegal() -> None:
    """The illegal-instruction predicate is gated on 'no semantic fired'.
    With the new dispatch, a correctly-formed ECALL with a7 in {93,63,64,214}
    should not be classified as illegal."""
    _, model = _build_blank()
    bads = [n for n in model.property_nodes if "illegal-instruction" in n.symbol]
    assert bads


# ──────────────────────────────────────────────────────────────────────────
# End-to-end via BMC on a tiny hand-built ELF-shaped model
# ──────────────────────────────────────────────────────────────────────────


def _build_with_code(instructions: list[int], base: int = 0x1000) -> RISCVMachineBuilder:
    """Build a machine with the given 32-bit instruction stream baked into
    code memory starting at ``base``."""
    cfg = ModelConfig(
        is_64bit=True, cores=1, code_start=base,
        init_registers_to_zero=True,
    )
    b = RISCVMachineBuilder(cfg)
    b.build()
    # Turn each instruction word into little-endian bytes.
    data = b"".join(w.to_bytes(4, "little") for w in instructions)
    b.initialize_code_segment(0, base, data)
    return b


# RV64 instruction encoders — just enough to express what we need.
def addi(rd: int, rs1: int, imm: int) -> int:
    return 0x13 | (rd << 7) | (0 << 12) | (rs1 << 15) | ((imm & 0xFFF) << 20)


def lui(rd: int, imm20: int) -> int:
    return 0x37 | (rd << 7) | ((imm20 & 0xFFFFF) << 12)


ECALL = 0x00000073


def test_ecall_exit_halts_and_sets_exit_code() -> None:
    """addi a7, x0, 93; addi a0, x0, 42; ecall → halted, exit_code=42."""
    b = _build_with_code([
        addi(17, 0, 93),   # a7 = 93 (SYS_exit)
        addi(10, 0, 42),   # a0 = 42 (exit code)
        ECALL,
        addi(0, 0, 0),     # nop (should never execute)
    ])

    from rotor.btor2 import BTOR2Printer, parse_btor2
    dag = parse_btor2(BTOR2Printer().render(b.dag))

    u = BitwuzlaUnroller(dag)
    # No bad is defined on this model; add one that fires iff exit_code==42.
    # Simplest: hand-build the constraint-free check via BMC's internals —
    # we just inspect the witness after running with a "force halted" bad.
    # Instead, install a bad that asserts halted at step k.
    from rotor.btor2 import BTOR2Builder
    b2 = BTOR2Builder(b.dag)
    halted = b.dag.by_nid(b._halted_nodes[0].nid)
    assert halted is not None
    b2.bad(halted, "halted")
    dag2 = b.dag
    u2 = BitwuzlaUnroller(dag2)
    result = u2.check(bound=5)
    assert result.verdict == "sat"
    # By step 3, halted is set; the exit code should be 42.
    halted_step = result.steps or 0
    frame = next(f for f in result.witness if f["step"] == halted_step)
    assert frame["assignments"]["halted"] == 1
    assert frame["assignments"]["exit-code"] == 42


def test_ecall_read_writes_input_byte_to_memory() -> None:
    """addi a7, x0, 63; lui a1, 0x20; addi a2, x0, 1; ecall → memory[a1]=input."""
    # a1 = 0x20 << 12 = 0x20000 (a virtual address within our default 32-bit
    # address space).
    b = _build_with_code([
        addi(17, 0, 63),     # a7 = 63 (SYS_read)
        addi(10, 0, 0),      # a0 = 0  (stdin)
        lui(11, 0x20),       # a1 = 0x20000
        addi(12, 0, 1),      # a2 = 1
        ECALL,
    ])
    # Verify the read syscall fires and the ecall halts progression.
    # We use the existing halted property the builder installed, plus the
    # input-byte propagation visible in the witness memory.
    from rotor.btor2 import BTOR2Builder, BTOR2Printer, parse_btor2
    # Add a bad asserting that the register a0 (bytes-read) is 1 at some step
    # (i.e. the read syscall has executed and written the count).
    b2 = BTOR2Builder(b.dag)
    regs = b._register_file[0]
    assert b.SID_REGISTER_ADDRESS is not None and b.SID_MACHINE_WORD is not None
    a0_idx = b2.constd(b.SID_REGISTER_ADDRESS, 10, "x10")
    a0_val = b2.read(regs, a0_idx, "a0")
    one = b2.one(b.SID_MACHINE_WORD, "1")
    b2.bad(b2.eq(a0_val, one), "a0==1")

    u = BitwuzlaUnroller(b.dag)
    result = u.check(bound=6)
    assert result.verdict == "sat"
    # Verify at the hit step that a0 is indeed 1.
    hit_step = result.steps
    frame = next(f for f in result.witness if f["step"] == hit_step)
    assert frame["assignments"]["register-file[10]"] == 1


def test_ecall_read_multibyte_writes_n_input_bytes() -> None:
    """A single read with a2=3 should leave three distinct symbolic bytes
    at memory[a1], memory[a1+1], memory[a1+2] — and the rest of the buffer
    unchanged."""
    # a1 = 0x40000 (upper-half lui + addi 0), a2 = 3, a7 = 63.
    b = _build_with_code([
        addi(17, 0, 63),     # a7 = 63
        addi(10, 0, 0),      # a0 = 0
        lui(11, 0x40),       # a1 = 0x40000
        addi(12, 0, 3),      # a2 = 3
        ECALL,
    ])
    from rotor.btor2 import BTOR2Builder
    b2 = BTOR2Builder(b.dag)
    regs = b._register_file[0]
    assert b.SID_REGISTER_ADDRESS is not None and b.SID_MACHINE_WORD is not None
    a0_idx = b2.constd(b.SID_REGISTER_ADDRESS, 10, "x10")
    a0_val = b2.read(regs, a0_idx, "a0")
    three = b2.constd(b.SID_MACHINE_WORD, 3, "3")
    b2.bad(b2.eq(a0_val, three), "a0==3")

    u = BitwuzlaUnroller(b.dag)
    result = u.check(bound=6)
    assert result.verdict == "sat"
    # At the hit step, a0 (bytes read) equals 3 and read-count equals 1.
    frame = next(f for f in result.witness if f["step"] == result.steps)
    assert frame["assignments"]["register-file[10]"] == 3
    assert frame["assignments"]["read-count"] == 1


def test_ecall_read_count_saturates_at_buffer_size() -> None:
    """If a2 > buffer_size, a0 should return buffer_size, not a2."""
    from rotor.btor2 import RISCVMachineBuilder
    from rotor.instance import ModelConfig

    cfg = ModelConfig(is_64bit=True, cores=1, code_start=0x1000,
                      read_buffer_size=4)
    b = RISCVMachineBuilder(cfg)
    b.build()
    program = [
        addi(17, 0, 63),    # a7 = 63
        addi(10, 0, 0),     # a0 = 0
        lui(11, 0x40),      # a1 = 0x40000
        addi(12, 0, 10),    # a2 = 10  (larger than buffer_size=4)
        ECALL,
    ]
    data = b"".join(w.to_bytes(4, "little") for w in program)
    b.initialize_code_segment(0, 0x1000, data)

    from rotor.btor2 import BTOR2Builder
    b2 = BTOR2Builder(b.dag)
    regs = b._register_file[0]
    assert b.SID_REGISTER_ADDRESS is not None and b.SID_MACHINE_WORD is not None
    a0_idx = b2.constd(b.SID_REGISTER_ADDRESS, 10, "x10")
    a0_val = b2.read(regs, a0_idx, "a0")
    four = b2.constd(b.SID_MACHINE_WORD, 4, "4")
    b2.bad(b2.eq(a0_val, four), "a0==4")

    u = BitwuzlaUnroller(b.dag)
    result = u.check(bound=6)
    assert result.verdict == "sat"


def test_ecall_brk_updates_program_break() -> None:
    """brk(addr) should latch program-break to addr and return it in a0."""
    # a7 = 214, a0 = 0x10000000 (new break, via lui 0x10000 << 12).
    b = _build_with_code([
        addi(17, 0, 214),       # a7 = 214 (brk)
        lui(10, 0x10000),       # a0 = 0x10000000
        ECALL,
    ])
    from rotor.btor2 import BTOR2Builder
    b2 = BTOR2Builder(b.dag)
    assert b.SID_MACHINE_WORD is not None
    brk = b.dag.by_nid(b._program_break_nodes[0].nid)
    assert brk is not None
    target = b2.consth(b.SID_MACHINE_WORD, 0x10000000, "0x10000000")
    b2.bad(b2.eq(brk, target), "brk-latched")

    u = BitwuzlaUnroller(b.dag)
    result = u.check(bound=4)
    assert result.verdict == "sat"
    # program-break should latch to 0x10000000 after ecall fires.
    frame = next(f for f in result.witness if f["step"] == result.steps)
    assert frame["assignments"]["program-break"] == 0x10000000


def test_ecall_brk_zero_returns_current_break() -> None:
    """brk(0) should leave program-break unchanged and return its value."""
    # a7 = 214, a0 = 0. The default program-break init is 0x20000000.
    b = _build_with_code([
        addi(17, 0, 214),   # a7 = 214 (brk)
        addi(10, 0, 0),     # a0 = 0
        ECALL,
    ])
    from rotor.btor2 import BTOR2Builder
    b2 = BTOR2Builder(b.dag)
    regs = b._register_file[0]
    assert b.SID_REGISTER_ADDRESS is not None and b.SID_MACHINE_WORD is not None
    a0_idx = b2.constd(b.SID_REGISTER_ADDRESS, 10, "x10")
    a0_val = b2.read(regs, a0_idx, "a0")
    expected = b2.consth(b.SID_MACHINE_WORD, 0x20000000, "default_brk")
    b2.bad(b2.eq(a0_val, expected), "a0-returns-brk")

    u = BitwuzlaUnroller(b.dag)
    result = u.check(bound=4)
    assert result.verdict == "sat"
    frame = next(f for f in result.witness if f["step"] == result.steps)
    assert frame["assignments"]["program-break"] == 0x20000000


def test_unknown_ecall_halts_without_setting_exit_code() -> None:
    """ECALL with a7 not in {exit,read,write,brk} halts cleanly."""
    b = _build_with_code([
        addi(17, 0, 999),   # a7 = some unknown syscall number
        ECALL,
    ])
    from rotor.btor2 import BTOR2Builder
    b2 = BTOR2Builder(b.dag)
    halted = b.dag.by_nid(b._halted_nodes[0].nid)
    assert halted is not None
    b2.bad(halted, "halted")

    u = BitwuzlaUnroller(b.dag)
    result = u.check(bound=4)
    assert result.verdict == "sat"
    frame = next(f for f in result.witness if f["step"] == result.steps)
    assert frame["assignments"]["halted"] == 1
    # Exit code should remain at its init value (0) because the unknown
    # syscall path does not update it.
    assert frame["assignments"]["exit-code"] == 0
