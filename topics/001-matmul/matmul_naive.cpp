#include "matmul.h"
#include <cassert>

namespace matmul {

/**
 * @brief Naive Matrix Multiplication (row-col-inner loop order)
 *
 * This is the textbook implementation of matrix multiplication.
 * While easy to understand, it is extremely inefficient on modern hardware.
 *
 * Performance Issues:
 * - Poor Cache Utilization: In the innermost 'inner' loop,
 *   B[inner * columns + col] is
 * accessed. Because matrices are stored in row-major order, accessing elements
 * column-wise (incrementing 'inner') causes large memory jumps. This ruins
 * spatial locality, leading to frequent CPU cache misses.
 * - Time Complexity: O(rows * columns * inners)
 */
void Naive(const double *A, const double *B, double *C, int rows, int columns,
           int inners) {
  assert(rows == columns);
  assert(columns == inners);
  const int size = rows;

  for (int row = 0; row < size; ++row) {
    for (int col = 0; col < size; ++col) {
      for (int inner = 0; inner < size; ++inner) {
        C[row * size + col] += A[row * size + inner] * B[inner * size + col];
      }
    }
  }
}

} // namespace matmul
