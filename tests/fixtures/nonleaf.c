/* Track C fixture — a non-leaf function calling a leaf through a
 * real jal. The whole point of C.1's cycle-0 ra-constraint refactor
 * is that a jal inside the analyzed set legitimately writes an
 * intra-set PC into ra without triggering the entry assumption; when
 * the callee rets, ra carries the caller's post-jal PC and execution
 * resumes. `noinline` forces the call so the BTOR2 encoding actually
 * contains a jal + jalr pair.
 *
 * `square(x)` is a leaf that returns x*x via RV64M mul.
 * `double_square(x)` calls square once, doubles the result.
 */

static int __attribute__((noinline)) square(int x) {
    return x * x;
}

int double_square(int x) {
    return 2 * square(x);
}
