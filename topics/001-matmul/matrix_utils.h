#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <vector>

namespace matmul {

// Fills an array of given size with random double values.
void InitializeRandom(double *data, int size);

// Compares the output matrix with a reference matrix to ensure correctness.
// Returns true if all elements match within the given tolerance.
bool VerifyResults(const double *C, const double *refC, int size,
                   double tolerance = 1e-3f);

// Aligned allocation for SIMD (Single Instruction, Multiple Data) operations.
// Modern CPUs require or perform better when memory accessed by vector
// instructions is aligned to specific boundaries (e.g., 64 bytes for AVX-512 /
// ARM NEON cache lines).
double *AllocateAligned(int size);

// Frees memory allocated by AllocateAligned.
void FreeAligned(double *ptr);

} // namespace matmul

#endif // MATRIX_UTILS_H
