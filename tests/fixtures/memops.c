/* Memory model fixture.
 *
 * load_sum exercises two aligned lw's off a caller-provided pointer —
 * a minimal non-leaf-style pattern (a pointer parameter is the thing
 * that forces real memory access even at -O2, without pulling in a
 * full stack frame we would need a calling convention for).
 *
 * roundtrip writes through the pointer and reads back through the
 * same pointer. The SMT array must identify the read with the matching
 * store, so rotor can prove that `return p[0] + p[0]` folds to
 * `return 2*x` over the memory model.
 */

int load_sum(int *p) {
    return p[0] + p[1];
}

int roundtrip(int *p, int x) {
    p[0] = x;
    *(volatile int *)p;            /* prevent clang from folding the load */
    return p[0] + p[0];
}
