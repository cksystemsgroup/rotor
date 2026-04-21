#!/usr/bin/env bash
# Rebuild the RISC-V fixtures used by rotor's integration tests.
# Produces bare-metal ELFs with no startup code so the generated
# machine model only needs to model the functions themselves.
#
# Default toolchain: the riscv64-unknown-elf bare-metal gcc cross
# compiler. Clang (18+) with ld.lld works just as well — set
# $CC / $LD to override. The shipped binaries in this directory were
# produced by one of these two toolchains; small codegen differences
# can shift function offsets, so corpus offsets may need a refresh
# after a rebuild.
set -euo pipefail
cd "$(dirname "$0")"

CC="${CC:-riscv64-unknown-elf-gcc}"
COMMON=(
    -march=rv64i           # no compressed, no M — keeps decoder simple
    -mabi=lp64
    -O2                    # leaf functions compile to minimal sequences
    -ffreestanding
    -nostdlib
    -nostartfiles
    -fno-asynchronous-unwind-tables
    -fno-unwind-tables
    -g                     # DWARF line info for source mapping
)

build_with_gcc() {
    local entry="$1"; shift
    local out="$1"; shift
    "$CC" "${COMMON[@]}" "-Wl,--entry=$entry" -o "$out" "$@"
}

build_with_clang() {
    local entry="$1"; shift
    local out="$1"; shift
    local obj
    obj="$(mktemp --suffix=.o)"
    "$CC" --target=riscv64 "${COMMON[@]}" -c -o "$obj" "$@"
    ld.lld -m elf64lriscv "--entry=$entry" -o "$out" "$obj"
    rm -f "$obj"
}

build() {
    if [[ "$CC" == *clang* ]]; then
        build_with_clang "$@"
    else
        build_with_gcc "$@"
    fi
}

build add2      add2.elf      add2.c
echo "built: add2.elf"

build branches  branches.elf  branches.c
echo "built: branches.elf"

# M6 fixtures — exercise the memory model.
build load_sum  memops.elf    memops.c
echo "built: memops.elf"

build pick      rodata.elf    rodata.c
echo "built: rodata.elf"

# Phase 6.3 fixture — a loop whose dead branch is provable only via
# an inductive invariant. Z3Spacer proves unbounded safety; Z3BMC
# can only answer "unreachable up to bound k".
build bounded_counter counter.elf counter.c
echo "built: counter.elf"
