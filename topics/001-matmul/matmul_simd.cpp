#include "matmul.h"
#include <cassert>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace matmul {

namespace {

/**
 * @brief SIMD (Single Instruction, Multiple Data) Micro-Kernel
 *
 * SIMD instructions allow the CPU to perform the same operation on multiple
 * data points simultaneously. This heavily accelerates compute-bound tasks like
 * matmul.
 *
 * Concept:
 * - We load multiple elements (e.g., 2 double-precision floats in a 128-bit
 * NEON register) into a single vector register.
 * - A single fused multiply-add (FMA) instruction multiplies them and adds them
 * to the accumulator vector.
 * - Vectorizing the innermost loop ('col') works best because of contiguous
 * memory access.
 */
inline void SimdTile(const double *A, const double *B, double *C, int size,
                     int row_block, int col_block, int inner_block) {
  for (int row = 0; row < kDefaultTileSize; ++row) {
    const double *a_row = &A[(row_block + row) * size + inner_block];
    double *c_row = &C[(row_block + row) * size + col_block];
    for (int inner = 0; inner < kDefaultTileSize; ++inner) {
      const double a_val = a_row[inner];
      const double *b_row = &B[(inner_block + inner) * size + col_block];
#if defined(__ARM_NEON)
      // Broadcast a_val into all lanes of a vector register
      const float64x2_t a_vec = vmovq_n_f64(a_val);

      // Process 2 elements at a time (128-bit / 64-bit = 2 lanes).
      for (int col = 0; col < kDefaultTileSize; col += 2) {
        float64x2_t b_vec = vld1q_f64(&b_row[col]);
        float64x2_t c_vec = vld1q_f64(&c_row[col]);

        // Fused Multiply-Add: c_vec = c_vec + a_vec * b_vec
        c_vec = vmlaq_f64(c_vec, a_vec, b_vec);

        // Store the result back to C
        vst1q_f64(&c_row[col], c_vec);
      }
#else
      // Fallback for non-NEON platforms
      for (int col = 0; col < kDefaultTileSize; ++col) {
        c_row[col] += a_val * b_row[col];
      }
#endif
    }
  }
}

} // namespace

/**
 * @brief SIMD Matrix Multiplication
 * Combines cache tiling with SIMD vectorization.
 */
void Simd(const double *A, const double *B, double *C, int rows, int columns,
          int inners) {
  assert(rows == columns);
  assert(columns == inners);
  const int size = rows;
  assert(size % kDefaultTileSize == 0);

  // Outer loops for cache tiling
  for (int row_block = 0; row_block < size; row_block += kDefaultTileSize) {
    for (int inner_block = 0; inner_block < size;
         inner_block += kDefaultTileSize) {
      for (int col_block = 0; col_block < size; col_block += kDefaultTileSize) {
        SimdTile(A, B, C, size, row_block, col_block, inner_block);
      }
    }
  }
}

} // namespace matmul
