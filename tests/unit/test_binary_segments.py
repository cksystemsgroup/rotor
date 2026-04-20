"""ELF segment iteration for the M6 memory initializer."""

from pathlib import Path

from rotor.binary import RISCVBinary

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def test_loadable_bytes_includes_rodata_table() -> None:
    """rodata.c declares `static const int table[4] = {11,22,33,44}` —
    those four 32-bit little-endian values must appear verbatim in
    the stream of loadable bytes from PT_LOAD."""
    with RISCVBinary(FIX / "rodata.elf") as b:
        byte_map = dict(b.loadable_bytes())

    # Locate any contiguous run of the four encoded words.
    needle = bytes()
    for v in (11, 22, 33, 44):
        needle += v.to_bytes(4, "little")
    addrs = sorted(byte_map)
    found = False
    for i in range(len(addrs) - len(needle)):
        base = addrs[i]
        if all(byte_map.get(base + j) == needle[j] for j in range(len(needle))):
            found = True
            break
    assert found, "rodata table not found in PT_LOAD bytes"


def test_loadable_bytes_includes_text_instructions() -> None:
    """The first instruction word of add2 must appear in loadable bytes
    at its virtual address."""
    with RISCVBinary(FIX / "add2.elf") as b:
        fn = b.function("add2")
        byte_map = dict(b.loadable_bytes())
        first_word = next(b.instructions(fn)).word
    expected = first_word.to_bytes(4, "little")
    for offset, exp_byte in enumerate(expected):
        assert byte_map.get(fn.start + offset) == exp_byte


def test_loadable_bytes_addresses_are_unique() -> None:
    with RISCVBinary(FIX / "rodata.elf") as b:
        pairs = list(b.loadable_bytes())
    addrs = [a for a, _ in pairs]
    assert len(addrs) == len(set(addrs))
