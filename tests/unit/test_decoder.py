"""Decoder tests covering every RV64I mnemonic supported by M5."""

import pytest

from rotor.btor2.riscv.decoder import decode

# Hand-assembled instruction words validated against riscv64-unknown-elf-objdump.
# Each row: (word, mnem, rd, rs1, rs2_or_zero, imm)
SAMPLES = [
    # Original M1 set
    (0x00B5053B, "addw",  10, 10, 11, 0),
    (0x00008067, "jalr",   0,  1,  0, 0),     # ret
    (0x00A04863, "blt",    0,  0, 10, 16),    # bgtz a0, +16
    (0x00A03533, "sltu",  10,  0, 10, 0),     # snez a0, a0
    (0x40A00533, "sub",   10,  0, 10, 0),     # neg a0, a0
    (0x00100513, "addi",  10,  0,  0, 1),     # li a0, 1

    # OP-IMM additions
    (0x00A52793, "slti",  15, 10,  0, 10),    # slti a5, a0, 10
    (0x0000B793, "sltiu", 15,  1,  0, 0),     # sltiu a5, ra, 0  (= seqz)
    (0xFFF54513, "xori",  10, 10,  0, -1),    # not a0, a0  -> xori a0, a0, -1
    (0x0FF57513, "andi",  10, 10,  0, 0xFF),  # andi a0, a0, 0xFF
    (0x07876513, "ori",   10, 14,  0, 0x78),  # ori a0, a4, 120
    (0x00351513, "slli",  10, 10,  0, 3),     # slli a0, a0, 3
    (0x00355513, "srli",  10, 10,  0, 3),     # srli a0, a0, 3
    (0x40355513, "srai",  10, 10,  0, 3),     # srai a0, a0, 3

    # OP-IMM-32
    (0x0015051B, "addiw", 10, 10,  0, 1),     # addiw a0, a0, 1
    (0x0035151B, "slliw", 10, 10,  0, 3),     # slliw a0, a0, 3
    (0x0035551B, "srliw", 10, 10,  0, 3),
    (0x4035551B, "sraiw", 10, 10,  0, 3),

    # OP additions
    (0x00B50533, "add",   10, 10, 11, 0),
    (0x00B57533, "and",   10, 10, 11, 0),
    (0x00B56533, "or",    10, 10, 11, 0),
    (0x00B54533, "xor",   10, 10, 11, 0),
    (0x00B52533, "slt",   10, 10, 11, 0),
    (0x00B51533, "sll",   10, 10, 11, 0),
    (0x00B55533, "srl",   10, 10, 11, 0),
    (0x40B55533, "sra",   10, 10, 11, 0),

    # OP-32 additions
    (0x40B5053B, "subw",  10, 10, 11, 0),
    (0x00B5153B, "sllw",  10, 10, 11, 0),
    (0x00B5553B, "srlw",  10, 10, 11, 0),
    (0x40B5553B, "sraw",  10, 10, 11, 0),

    # Branches: full set
    (0x00B50663, "beq",    0, 10, 11, 12),
    (0x00B51663, "bne",    0, 10, 11, 12),
    (0x00B55663, "bge",    0, 10, 11, 12),
    (0x00B57663, "bgeu",   0, 10, 11, 12),
    (0x00B5E663, "bltu",   0, 11, 11, 12),    # bltu rs1=11 rs2=11

    # U / J / fence
    (0x000017B7, "lui",   15,  0,  0, 0x1000),    # lui a5, 0x1
    (0x00000517, "auipc", 10,  0,  0, 0),         # auipc a0, 0
    (0x00C0006F, "jal",    0,  0,  0, 12),        # j +12
    (0x0FF0000F, "fence",  0,  0,  0, 0),         # fence iorw, iorw
]


@pytest.mark.parametrize("word,mnem,rd,rs1,rs2,imm", SAMPLES)
def test_decoder_sample(word, mnem, rd, rs1, rs2, imm) -> None:
    d = decode(word)
    assert d is not None, f"decoder rejected 0x{word:08x}"
    assert (d.mnem, d.rd, d.rs1, d.rs2, d.imm) == (mnem, rd, rs1, rs2, imm)


def test_unknown_returns_none() -> None:
    # Reserved opcode (all-zero is illegal; we choose 0xFFFF_FFFF which is also illegal).
    assert decode(0xFFFFFFFF) is None


def test_invalid_funct3_within_supported_opcode() -> None:
    # OP-IMM funct3 = 101 with funct6 = 100000 (not 000000 or 010000) is invalid.
    word = (0b100000 << 26) | (0b101 << 12) | 0b0010011
    assert decode(word) is None
