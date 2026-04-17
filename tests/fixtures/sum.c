// Computes sum(0..n-1). For n=6, sum is 15.
// Used to exercise multiplication and 32-bit integer arithmetic.
int sum_to(int n) {
    int acc = 0;
    int i = 0;
    while (i < n) {
        acc = acc + i;
        i = i + 1;
    }
    return acc;
}

void _start(void) {
    sum_to(6);
}
