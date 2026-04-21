/* RV64M fixture: functions that compile to mul / div / rem / mulw.
 * Built with `-march=rv64im` so the compiler actually emits the
 * M-extension instructions rather than library calls.
 */

int mul_add(int a, int b, int c) {
    return a * b + c;
}

unsigned int divmod(unsigned int a, unsigned int b) {
    return (a / b) + (a % b);
}

long long mul64(long long a, long long b) {
    return a * b;
}
