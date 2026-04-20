#!/usr/bin/env bash
# Rebuild the RISC-V fixtures used by rotor's integration tests.
# Produces bare-metal ELFs with no startup code so the generated
# machine model only needs to model the functions themselves.
set -euo pipefail
cd "$(dirname "$0")"

CC=riscv64-unknown-elf-gcc
FLAGS=(
    -march=rv64im          # no compressed instructions (keeps decoder simple)
    -mabi=lp64
    -O2                    # leaf functions compile to minimal sequences
    -ffreestanding
    -nostdlib
    -nostartfiles
    -fno-asynchronous-unwind-tables
    -fno-unwind-tables
    -g                     # DWARF line info for source mapping
    -Wl,--entry=add2       # avoid requiring a _start
)

"$CC" "${FLAGS[@]}" -o add2.elf add2.c
echo "built: add2.elf"

"$CC" "${FLAGS[@]/--entry=add2/--entry=branches}" -o branches.elf branches.c
echo "built: branches.elf"
