/* Track B.4 fixture — a small bit-manipulation library built with
 * `-march=rv64imc` (the default-ish clang/gcc target for RISC-V
 * Linux). Combines the three ISA additions shipped in Track B:
 *
 *   - RV64M arithmetic in `shifted_mul`
 *   - compressed instructions throughout (leaf epilogues, simple
 *     additions, constant loads)
 *   - branches + loops in `popcount`
 *
 * Each function is leaf so `EntryAssumptions` (Track C) isn't
 * needed yet; this fixture is an existence proof that the ISA
 * stack we've shipped is enough for non-trivial user code.
 */

int is_power_of_two(unsigned int x) {
    /* x != 0 && (x & (x - 1)) == 0 */
    if (x == 0) return 0;
    return (x & (x - 1)) == 0 ? 1 : 0;
}

int popcount(unsigned int x) {
    int count = 0;
    while (x != 0) {
        count += (int)(x & 1);
        x >>= 1;
    }
    return count;
}

int shifted_mul(int a, int b, int shift) {
    /* Uses mul + sll in one function to exercise the M extension
     * alongside regular shifts. */
    int product = a * b;
    return product << (shift & 31);
}
