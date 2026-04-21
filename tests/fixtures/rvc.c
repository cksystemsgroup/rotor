/* RVC fixture — built with `-march=rv64imc` so the compiler emits
 * compressed (16-bit) instructions throughout. Exercises rotor's
 * RVC decompressor and variable-length instruction scanner.
 */

int add_rvc(int a, int b) {
    return a + b;
}

int triple(int x) {
    return x * 3 + 1;
}

int signbit(int x) {
    if (x < 0) return 1;
    return 0;
}
