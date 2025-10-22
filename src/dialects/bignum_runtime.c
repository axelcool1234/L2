// src/dialects/bignum_runtime.c
#include <stdlib.h>
#include <gmp.h>

typedef mpz_t *bignum_t; // pointer to mpz_t (allocated)

bignum_t l2_bignum_from_i32(int32_t v) {
    bignum_t p = malloc(sizeof(mpz_t));
    if (!p) return NULL;
    mpz_init((*p));
    mpz_set_si((*p), v);
    return p;
}

bignum_t l2_bignum_add(bignum_t a, bignum_t b) {
    bignum_t r = malloc(sizeof(mpz_t));
    if (!r) return NULL;
    mpz_init((*r));
    mpz_add((*r), (*a), (*b));
    return r;
}
 
// Comparison: returns 1 if a < b, else 0
int l2_bignum_lt(bignum_t a, bignum_t b) {
    return mpz_cmp(*a, *b) < 0;
}

// Comparison: returns 1 if a <= b, else 0
int l2_bignum_lte(bignum_t a, bignum_t b) {
    return mpz_cmp(*a, *b) <= 0;
}

// Comparison: returns 1 if a > b, else 0
int l2_bignum_gt(bignum_t a, bignum_t b) {
    return mpz_cmp(*a, *b) > 0;
}

// Comparison: returns 1 if a >= b, else 0
int l2_bignum_gte(bignum_t a, bignum_t b) {
    return mpz_cmp(*a, *b) >= 0;
}

// Comparison: returns 1 if a == b, else 0
int l2_bignum_eq(bignum_t a, bignum_t b) {
    return mpz_cmp(*a, *b) == 0;
}

// Comparison: returns 1 if a != b, else 0
int l2_bignum_neq(bignum_t a, bignum_t b) {
    return mpz_cmp(*a, *b) != 0;
}

void l2_bignum_print(bignum_t a) {
    gmp_printf("%Zd", (*a));
}

void l2_bignum_println(bignum_t a) {
    gmp_printf("%Zd\n", (*a));
}

void l2_bignum_free(bignum_t a) {
    if (!a) return;
    mpz_clear((*a));
    free(a);
}
