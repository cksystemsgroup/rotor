// Minimal function that returns a0+a1.
// Useful for verifying the bitwuzla unroller end-to-end on an
// actually-executable program (2 instructions + ret).
int add2(int a, int b) {
    return a + b;
}

void _start(void) {
    add2(3, 4);
}
