/* Counter fixture: dead branches that IC3 can prove unreachable via
 * register-range reasoning. Designed small enough that Spacer closes
 * the invariant before M8's SSA slicing lands — the PDR engine's
 * cost scales steeply with the number of state variables, so each
 * additional live register pushes closure time up hard.
 *
 * `tiny_mask` has no loop and uses a single masked input, giving
 * Spacer an easy constant-range invariant (`x <= 3`) to discover.
 *
 * `bounded_counter` is the loop-carried version; it will likely hit
 * Spacer's ceiling until M8 prunes unused registers, but BMC on it
 * still demonstrates the "unreachable up to k, never `proved`"
 * BMC/IC3 contrast.
 *
 * Inline-asm barriers defeat clang's compile-time range analysis so
 * the dead branches actually survive into the binary.
 */

int tiny_mask(int n) {
    int x = n & 3;                      /* x ∈ {0, 1, 2, 3} */
    __asm__ volatile("" : "+r"(x));
    if (x > 10) return x ^ 0xdeadbeef;  /* dead — x is 2-bit-bounded */
    return x * 3 + 1;
}

int bounded_counter(int n) {
    int x = 0;
    for (int i = 0; i < n; i++) {
        if (i >= 100) break;
        x++;
        __asm__ volatile("" : "+r"(x));
    }
    /* Invariant: x ∈ [0, 100]. The `x > 200` branch is unreachable. */
    if (x > 200) return x ^ 0xdeadbeef;
    return x * 3 + 1;
}
