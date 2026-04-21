/* Counter fixture: a loop whose dead branch is provable only with an
 * inductive invariant. BMC at any finite bound returns `unreachable`;
 * IC3 (Z3Spacer) returns `proved` with a loop invariant certificate.
 *
 * Two clang-defeating tricks:
 *
 *   (1) An inline-asm identity barrier inside the loop stops clang
 *       from range-analyzing the counter at compile time. Without
 *       it `-O2` folds `x in [0,100]` statically and the dead
 *       branch vanishes.
 *
 *   (2) The two return paths produce syntactically unrelated
 *       expressions (`x ^ 0xdeadbeef` vs `x * 3 + 1`), so clang
 *       cannot fuse them into a branchless `slti`+`or` select —
 *       it emits a real conditional branch, giving rotor a PC to
 *       ask about.
 */

int bounded_counter(int n) {
    int x = 0;
    for (int i = 0; i < n; i++) {
        if (i >= 100) break;
        x++;
        __asm__ volatile("" : "+r"(x));
    }
    /* Invariant at this point: x in [0, 100]. The `x > 200` branch
     * is unreachable — rotor's IC3 backend must discover this. */
    if (x > 200) return x ^ 0xdeadbeef;
    return x * 3 + 1;
}
