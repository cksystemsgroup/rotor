// Reads 4 bytes from stdin and returns their sum.
// A BMC-driven search should find independent input bytes that add up
// to any target value (modulo 8-bit overflow).
//
// This exercises the multi-byte read path: a single ecall delivers
// multiple symbolic bytes into the buffer, which the program then
// reads back individually.

static unsigned char buf[4];

int sum4(void) {
    register long a0 asm("a0") = 0;
    register long a1 asm("a1") = (long)buf;
    register long a2 asm("a2") = 4;
    register long a7 asm("a7") = 63;  // Linux SYS_read
    asm volatile ("ecall"
                  : "+r"(a0)
                  : "r"(a1), "r"(a2), "r"(a7)
                  : "memory");
    return (int)buf[0] + (int)buf[1] + (int)buf[2] + (int)buf[3];
}

void _start(void) {
    sum4();
}
