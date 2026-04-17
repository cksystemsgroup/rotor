// Reads a single byte from stdin via the Linux RISC-V read syscall and
// returns it. When BMC runs this with a symbolic input byte, the search
// space is the 256 possible bytes — the solver can trivially pick any
// value to match a return-value query.
//
// The inline asm binds a0..a2 and a7 to the syscall arguments and marks
// a0 as clobbered so the compiler reloads it as the syscall return value.

static char buf[1];

int read_byte(void) {
    register long a0 asm("a0") = 0;
    register long a1 asm("a1") = (long)buf;
    register long a2 asm("a2") = 1;
    register long a7 asm("a7") = 63;  // Linux SYS_read
    asm volatile ("ecall"
                  : "+r"(a0)
                  : "r"(a1), "r"(a2), "r"(a7)
                  : "memory");
    return buf[0];
}

void _start(void) {
    read_byte();
}
