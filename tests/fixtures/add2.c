/* Smallest reasonable M1 fixture: a leaf function with one branch.
 * No memory access (no prologue/epilogue), so the machine model only
 * needs a register file + PC, not an SMT array for memory.
 *
 *   add2(a, b) returns a+b                    — straight-line leaf
 *   sign(x)   returns 1/0/-1 depending on x   — branching leaf
 */

int add2(int a, int b) {
    return a + b;
}

int sign(int x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}
