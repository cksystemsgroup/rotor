#!/bin/sh
# Rebuild the RISC-V ELF fixtures used by tests/test_integration.py.
# Requires clang with RISC-V target support (clang --print-targets shows riscv64).
set -eu

cd "$(dirname "$0")"

CC=${CC:-clang}
FLAGS="--target=riscv64-unknown-elf -march=rv64im -mabi=lp64 \
  -ffreestanding -nostdlib -static -g -O0"

for src in counter.c sum.c add2.c readbyte.c; do
    out="${src%.c}.elf"
    echo "compiling $src -> $out"
    $CC $FLAGS "$src" -o "$out"
done
