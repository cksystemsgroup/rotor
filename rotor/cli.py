"""Minimal command-line entry point for Python Rotor.

Usage::

    rotor info binary.elf
    rotor btor2 binary.elf -o out.btor2
    rotor check binary.elf --bound 100 --solver btormc
"""

from __future__ import annotations

import argparse
import sys

from rotor.binary import RISCVBinary
from rotor.btor2 import CRotorBackend
from rotor.instance import ModelConfig, RotorInstance


def _cmd_info(args: argparse.Namespace) -> int:
    with RISCVBinary(args.path) as binary:
        print(f"path      : {binary._path}")
        print(f"is_64bit  : {binary.is_64bit}")
        print(f"entry     : 0x{binary.entry:x}")
        for name in ("code", "data", "rodata", "bss"):
            seg = getattr(binary, name)
            if seg:
                print(f"{name:9s} : 0x{seg.start:x}..0x{seg.start + seg.size:x} ({seg.size} bytes)")
        print(f"symbols   : {len(binary.symbols)}")
    return 0


def _cmd_btor2(args: argparse.Namespace) -> int:
    with RISCVBinary(args.path) as binary:
        config = ModelConfig(is_64bit=binary.is_64bit)
        inst = RotorInstance(binary, config)
        inst.build_machine()
        btor2_text = inst.emit_btor2()
    if args.output == "-":
        sys.stdout.write(btor2_text)
    else:
        with open(args.output, "w") as fp:
            fp.write(btor2_text)
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    with RISCVBinary(args.path) as binary:
        config = ModelConfig(
            is_64bit=binary.is_64bit,
            solver=args.solver,
            bound=args.bound,
        )
        inst = RotorInstance(binary, config)
        result = inst.check()
        print(f"verdict : {result.verdict}")
        print(f"solver  : {result.solver}")
        print(f"elapsed : {result.elapsed:.3f}s")
        if result.steps is not None:
            print(f"steps   : {result.steps}")
        if result.stderr:
            print(f"stderr  : {result.stderr.strip()[:200]}")
    return 0 if result.verdict != "unknown" else 2


def _cmd_crotor_check(args: argparse.Namespace) -> int:
    """Check whether the C Rotor binary is callable."""
    backend = CRotorBackend(args.binary)
    if backend.available():
        print(f"C Rotor available: {backend.binary}")
        return 0
    print(f"C Rotor not found: {backend.binary}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rotor", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="print ELF/DWARF summary")
    p_info.add_argument("path")
    p_info.set_defaults(func=_cmd_info)

    p_btor2 = sub.add_parser("btor2", help="emit BTOR2 for the binary")
    p_btor2.add_argument("path")
    p_btor2.add_argument("-o", "--output", default="-")
    p_btor2.set_defaults(func=_cmd_btor2)

    p_check = sub.add_parser("check", help="run a solver on the model")
    p_check.add_argument("path")
    p_check.add_argument("--solver", default="bitwuzla")
    p_check.add_argument("--bound", type=int, default=1000)
    p_check.set_defaults(func=_cmd_check)

    p_cr = sub.add_parser("crotor-check", help="check the C Rotor binary")
    p_cr.add_argument("--binary", default="rotor")
    p_cr.set_defaults(func=_cmd_crotor_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
