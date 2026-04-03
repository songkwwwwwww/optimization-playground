#include "matrix_utils.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>

namespace matmul {

// Initializes the array with random real numbers between -1.0 and 1.0.
void InitializeRandom(double *data, int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-1.0f, 1.0f);
  for (int i = 0; i < size; ++i) {
    data[i] = dis(gen);
  }
}

// Verifies if the computed result matches the reference result within a
// tolerance. This is crucial for making sure optimizations don't break the
// correctness.
bool VerifyResults(const double *C, const double *refC, int size,
                   double tolerance) {
  for (int i = 0; i < size; ++i) {
    if (std::abs(C[i] - refC[i]) > tolerance) {
      std::cerr << "Verification failed at index " << i << ": " << C[i]
                << " vs " << refC[i] << std::endl;
      return false;
    }
  }
  return true;
}

// Allocates memory aligned to a 64-byte boundary.
// Proper memory alignment prevents cross-cacheline loads and is required for
// maximum performance in SIMD vector load/store instructions (e.g., AVX, NEON).
double *AllocateAligned(int size) {
  void *ptr = nullptr;
  // Align to 64 bytes for AVX-512/NEON cache line compatibility
  if (posix_memalign(&ptr, 64, size * sizeof(double)) != 0) {
    return nullptr;
  }
  std::memset(ptr, 0, size * sizeof(double));
  return static_cast<double *>(ptr);
}

// Frees aligned memory.
void FreeAligned(double *ptr) { std::free(ptr); }

} // namespace matmul
