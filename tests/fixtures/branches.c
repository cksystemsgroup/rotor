/* Branches fixture: exercise all six RV64I branch types in a leaf function.
 *
 * GCC at -O2 typically lowers each `if (cmp) r |= mask;` into:
 *   - one branch (beq / bne / blt / bge / bltu / bgeu)
 *   - an OR-immediate of the mask into r
 *   - a join label
 *
 * The function is leaf (no stack), so no memory operations are needed.
 */

int branches(int a, int b) {
    int r = 0;
    if (a == b)                       r |= 1;
    if (a != b)                       r |= 2;
    if (a <  b)                       r |= 4;     /* signed   */
    if (a >= b)                       r |= 8;     /* signed   */
    if ((unsigned)a <  (unsigned)b)   r |= 16;    /* unsigned */
    if ((unsigned)a >= (unsigned)b)   r |= 32;    /* unsigned */
    return r;
}
