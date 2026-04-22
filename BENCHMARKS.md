# Solver shootout

Corpus: 44 benchmarks, bound=10, timeout=15s.

Cell format: `VERDICT TIME`. Verdicts are **BUG** (bad state reached, SAT), **SAFE** (bad unreachable at this bound, UNSAT), **PROVED** (inductive invariant found — safe for all bounds), or **—** (solver returned unknown / timed out). Parenthesized elapsed times mark runs classified UNSOLVED.

| Benchmark | z3-bmc | z3-spacer | bitwuzla | portfolio |
|---|---|---|---|---|
| add2-entry-trivial | BUG 37ms | BUG 29ms | BUG 7ms | BUG 23ms |
| add2-ret-one-step | BUG 444ms | BUG 35ms | BUG 7ms | BUG 23ms |
| add2-unreach-within-bound | BUG 464ms | BUG 59ms | BUG 9ms | BUG 15ms |
| sign-entry-trivial | BUG 429ms | BUG 15ms | BUG 8ms | BUG 9ms |
| sign-li-branch | BUG 579ms | BUG 36ms | BUG 15ms | BUG 28ms |
| sign-ret2-via-positive | BUG 597ms | BUG 59ms | BUG 17ms | BUG 18ms |
| sign-ret1-via-negative | BUG 590ms | BUG 895ms | BUG 13ms | BUG 13ms |
| sign-unreach-within-bound | BUG 583ms | BUG 37ms | BUG 22ms | BUG 49ms |
| branches-entry-trivial | BUG 653ms | BUG 29ms | BUG 10ms | BUG 33ms |
| branches-after-mv-and-li | BUG 781ms | BUG 233ms | BUG 10ms | BUG 10ms |
| branches-ret-reachable | BUG 815ms | BUG 1.06s | BUG 19ms | BUG 21ms |
| branches-jal-reachable | BUG 810ms | BUG 852ms | BUG 29ms | BUG 29ms |
| branches-c4-unreach-at-2 | BUG 734ms | BUG 615ms | BUG 19ms | BUG 65ms |
| load_sum-entry-trivial | BUG 873ms | — (15.1s) | BUG 22ms | BUG 17ms |
| load_sum-second-lw | BUG 5.12s | — (15.1s) | BUG 17ms | BUG 19ms |
| load_sum-addw | BUG 5.29s | — (15.1s) | BUG 26ms | BUG 21ms |
| load_sum-ret | BUG 5.15s | — (15.1s) | BUG 18ms | BUG 22ms |
| load_sum-unreach-at-0 | BUG 4.99s | — (15.1s) | BUG 18ms | BUG 17ms |
| roundtrip-entry-trivial | BUG 5.03s | — (15.1s) | BUG 19ms | BUG 19ms |
| roundtrip-after-sw | BUG 5.29s | — (15.1s) | BUG 17ms | BUG 16ms |
| roundtrip-after-lw | BUG 5.17s | — (15.1s) | BUG 17ms | BUG 17ms |
| roundtrip-ret | BUG 4.90s | — (15.1s) | BUG 19ms | BUG 18ms |
| pick-entry-trivial | BUG 5.48s | — (15.1s) | BUG 17ms | BUG 17ms |
| pick-after-auipc | BUG 5.39s | — (15.1s) | BUG 18ms | BUG 19ms |
| pick-lw | BUG 5.42s | — (15.1s) | BUG 17ms | BUG 16ms |
| pick-ret | BUG 5.48s | — (15.1s) | BUG 18ms | BUG 17ms |
| pick-unreach-at-4 | BUG 5.50s | — (15.1s) | BUG 18ms | BUG 19ms |
| mul_add-entry-trivial | BUG 1.26s | BUG 355ms | BUG 8ms | BUG 28ms |
| mul_add-ret | BUG 2.07s | BUG 735ms | BUG 8ms | BUG 8ms |
| divmod-entry-trivial | BUG 470ms | BUG 25ms | BUG 8ms | BUG 33ms |
| divmod-ret | BUG 615ms | BUG 815ms | BUG 9ms | BUG 9ms |
| mul64-entry-trivial | BUG 522ms | BUG 15ms | BUG 7ms | BUG 6ms |
| mul64-ret | BUG 410ms | BUG 176ms | BUG 8ms | BUG 7ms |
| add_rvc-entry-trivial | BUG 450ms | BUG 14ms | BUG 7ms | BUG 7ms |
| add_rvc-ret | BUG 351ms | BUG 37ms | BUG 8ms | BUG 8ms |
| triple-entry-trivial | BUG 406ms | BUG 17ms | BUG 8ms | BUG 11ms |
| triple-ret | BUG 493ms | BUG 125ms | BUG 8ms | BUG 9ms |
| signbit-ret | BUG 373ms | BUG 38ms | BUG 7ms | BUG 8ms |
| is_power_of_two-entry | BUG 460ms | BUG 16ms | BUG 8ms | BUG 8ms |
| is_power_of_two-ret | BUG 585ms | BUG 34ms | BUG 22ms | BUG 24ms |
| popcount-entry | BUG 535ms | BUG 27ms | BUG 10ms | BUG 11ms |
| shifted_mul-entry | BUG 573ms | BUG 16ms | BUG 8ms | BUG 8ms |
| shifted_mul-ret | BUG 433ms | BUG 293ms | BUG 8ms | BUG 7ms |
| tiny_mask-dead-branch-SAFE | SAFE 478ms | — (15.0s) | SAFE 21ms | SAFE 20ms |

## Aggregate (lower PAR-2 is better)

| Engine | Solved | PAR-2 (s) |
|---|---|---|
| z3-bmc | 44/44 | 87.1 |
| z3-spacer | 29/44 | 456.7 |
| bitwuzla | 44/44 | 0.6 |
| portfolio | 44/44 | 0.8 |

## Unique solves

_No engine has unique solves on this corpus._
