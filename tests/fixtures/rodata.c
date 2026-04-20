/* .rodata fixture.
 *
 * `pick(i)` reads the i-th element of a file-backed const table. The
 * ELF's PT_LOAD initializer puts the four 32-bit values at known
 * virtual addresses, so rotor can follow the load through the SMT
 * array and recover the concrete element for any i.
 */

static const int table[4] = {11, 22, 33, 44};

int pick(int i) {
    return table[i & 3];
}
