# Test fixtures

This directory is populated by the CI pipeline with small RISC-V ELF binaries
compiled from the C sources listed in `PLAN.md` §Testing Strategy. Typical
entries:

- `counter.elf` — simple loop with a known overflow point
- `buffer.elf` — buffer with a known overflow input
- `sort.elf` — sorting function, provably correct for small arrays
- `protocol.elf` — state machine with a known bad state
- `equivalent_a.elf` / `equivalent_b.elf` — two provably equivalent implementations
- `inequivalent_a.elf` / `inequivalent_b.elf` — two with a known diverging input

To build locally:

```
riscv64-linux-gnu-gcc -g -O0 -static counter.c -o counter.elf
```

The actual `.c` sources and build scripts are intentionally kept out of the
Python package so the repository stays lightweight; see the CI configuration
in `.github/workflows/` (future) for details.
