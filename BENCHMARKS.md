# Solver shootout

Corpus: 44 benchmarks, bound=10, timeout=15s.

Cell format: `VERDICT TIME`. Verdicts are **BUG** (bad state reached, SAT), **SAFE** (bad unreachable at this bound, UNSAT), **PROVED** (inductive invariant found — safe for all bounds), or **—** (solver returned unknown / timed out). Parenthesized elapsed times mark runs classified UNSOLVED.

| Benchmark | z3-bmc | z3-spacer | bitwuzla | cvc5-bmc | portfolio |
|---|---|---|---|---|---|
| add2-entry-trivial | BUG 42ms | BUG 30ms | BUG 8ms | BUG 24ms | BUG 34ms |
| add2-ret-one-step | BUG 465ms | BUG 39ms | BUG 8ms | BUG 21ms | BUG 15ms |
| add2-unreach-within-bound | BUG 440ms | BUG 70ms | BUG 9ms | BUG 43ms | BUG 18ms |
| sign-entry-trivial | BUG 479ms | BUG 22ms | BUG 9ms | BUG 25ms | BUG 10ms |
| sign-li-branch | BUG 616ms | BUG 42ms | BUG 17ms | BUG 44ms | BUG 17ms |
| sign-ret2-via-positive | BUG 570ms | BUG 72ms | BUG 17ms | BUG 45ms | BUG 22ms |
| sign-ret1-via-negative | BUG 639ms | BUG 1.03s | BUG 16ms | BUG 173ms | BUG 446ms |
| sign-unreach-within-bound | BUG 466ms | BUG 49ms | BUG 15ms | BUG 54ms | BUG 18ms |
| branches-entry-trivial | BUG 594ms | BUG 27ms | BUG 10ms | BUG 36ms | BUG 11ms |
| branches-after-mv-and-li | BUG 769ms | BUG 255ms | BUG 11ms | BUG 37ms | BUG 11ms |
| branches-ret-reachable | BUG 846ms | BUG 1.13s | BUG 24ms | BUG 118ms | BUG 25ms |
| branches-jal-reachable | BUG 906ms | BUG 951ms | BUG 35ms | BUG 219ms | BUG 32ms |
| branches-c4-unreach-at-2 | BUG 801ms | BUG 686ms | BUG 20ms | BUG 110ms | BUG 105ms |
| load_sum-entry-trivial | BUG 980ms | — (15.1s) | BUG 20ms | — (16.2s) | BUG 26ms |
| load_sum-second-lw | BUG 5.34s | — (15.1s) | BUG 22ms | — (16.1s) | BUG 41ms |
| load_sum-addw | BUG 5.32s | — (15.1s) | BUG 19ms | — (16.2s) | BUG 30ms |
| load_sum-ret | BUG 5.44s | — (15.1s) | BUG 19ms | — (16.1s) | BUG 19ms |
| load_sum-unreach-at-0 | BUG 5.66s | — (15.1s) | BUG 29ms | — (16.4s) | BUG 27ms |
| roundtrip-entry-trivial | BUG 5.26s | — (15.1s) | BUG 21ms | BUG 61ms | BUG 26ms |
| roundtrip-after-sw | BUG 5.61s | — (15.1s) | BUG 17ms | BUG 59ms | BUG 84ms |
| roundtrip-after-lw | BUG 5.29s | — (15.1s) | BUG 19ms | BUG 59ms | BUG 24ms |
| roundtrip-ret | BUG 5.57s | — (15.1s) | BUG 21ms | BUG 73ms | BUG 39ms |
| pick-entry-trivial | BUG 5.28s | — (15.1s) | BUG 24ms | BUG 75ms | BUG 47ms |
| pick-after-auipc | BUG 5.92s | — (15.1s) | BUG 20ms | BUG 70ms | BUG 24ms |
| pick-lw | BUG 5.67s | — (15.1s) | BUG 19ms | BUG 66ms | BUG 26ms |
| pick-ret | BUG 5.76s | — (15.1s) | BUG 23ms | BUG 74ms | BUG 21ms |
| pick-unreach-at-4 | BUG 5.81s | — (15.1s) | BUG 21ms | BUG 62ms | BUG 25ms |
| mul_add-entry-trivial | BUG 1.29s | BUG 347ms | BUG 8ms | BUG 23ms | BUG 65ms |
| mul_add-ret | BUG 2.70s | BUG 740ms | BUG 7ms | BUG 22ms | BUG 9ms |
| divmod-entry-trivial | BUG 485ms | BUG 27ms | BUG 10ms | BUG 25ms | BUG 10ms |
| divmod-ret | BUG 501ms | BUG 842ms | BUG 10ms | BUG 27ms | BUG 11ms |
| mul64-entry-trivial | BUG 518ms | BUG 18ms | BUG 8ms | BUG 26ms | BUG 28ms |
| mul64-ret | BUG 505ms | BUG 165ms | BUG 7ms | BUG 21ms | BUG 10ms |
| add_rvc-entry-trivial | BUG 453ms | BUG 18ms | BUG 8ms | BUG 23ms | BUG 29ms |
| add_rvc-ret | BUG 464ms | BUG 34ms | BUG 9ms | BUG 25ms | BUG 9ms |
| triple-entry-trivial | BUG 460ms | BUG 17ms | BUG 8ms | BUG 24ms | BUG 23ms |
| triple-ret | BUG 566ms | BUG 135ms | BUG 8ms | BUG 32ms | BUG 23ms |
| signbit-ret | BUG 513ms | BUG 41ms | BUG 9ms | BUG 21ms | BUG 8ms |
| is_power_of_two-entry | BUG 479ms | BUG 17ms | BUG 9ms | BUG 27ms | BUG 9ms |
| is_power_of_two-ret | BUG 587ms | BUG 38ms | BUG 28ms | BUG 41ms | BUG 116ms |
| popcount-entry | BUG 574ms | BUG 19ms | BUG 11ms | BUG 32ms | BUG 54ms |
| shifted_mul-entry | BUG 694ms | BUG 16ms | BUG 9ms | BUG 23ms | BUG 9ms |
| shifted_mul-ret | BUG 479ms | BUG 282ms | BUG 9ms | BUG 24ms | BUG 10ms |
| tiny_mask-dead-branch-SAFE | SAFE 503ms | — (15.0s) | SAFE 21ms | SAFE 177ms | SAFE 20ms |

## Aggregate (lower PAR-2 is better)

| Engine | Solved | PAR-2 (s) |
|---|---|---|
| z3-bmc | 44/44 | 92.3 |
| z3-spacer | 29/44 | 457.2 |
| bitwuzla | 44/44 | 0.7 |
| cvc5-bmc | 39/44 | 152.1 |
| portfolio | 44/44 | 1.7 |

## Unique solves

_No engine has unique solves on this corpus._
