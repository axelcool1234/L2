// src/dialects/bignum_runtime.c
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <gmp.h>

typedef mpz_t *bignum_t; // pointer to mpz_t (allocated)
typedef bignum_t* bignum_vector_t; // Vector type: array of bignum_t pointers

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

// Create a vector from individual elements
// Takes the element count followed by the elements
bignum_vector_t l2_bignum_from_elements(int32_t count, ...) {
    bignum_vector_t vec = malloc(sizeof(bignum_t) * count);
    if (!vec) return NULL;
    
    va_list args;
    va_start(args, count);
    
    for (int32_t i = 0; i < count; ++i) {
        vec[i] = va_arg(args, bignum_t);
    }
    
    va_end(args);
    return vec;
}

// Extract an element from a vector at the given index
// The index is a bignum_t that gets converted to an integer
bignum_t l2_bignum_extractelement(bignum_vector_t vec, bignum_t index) {
    if (!vec || !index) return NULL;    

    int32_t idx = mpz_get_si(*index);
    return vec[idx];
}

// Insert an element into a vector at the given index
bignum_vector_t l2_bignum_insert(bignum_t element, bignum_vector_t vec, bignum_t index, int32_t vec_size) {
    if (!vec || !index || !element) return NULL;
    
    int32_t idx = mpz_get_si(*index);
    vec[idx] = element;
    return vec;
}

// Free a vector (doesn't free the contained bignums)
void l2_bignum_vector_free(bignum_vector_t vec) {
    if (vec) free(vec);
}

// Print a vector
void l2_bignum_vector_print(bignum_vector_t vec, int32_t size) {
    if (!vec) return;
    
    printf("[");
    for (int32_t i = 0; i < size; ++i) {
        if (i > 0) printf(", ");
        gmp_printf("%Zd", (*vec[i]));
    }
    printf("]");
}