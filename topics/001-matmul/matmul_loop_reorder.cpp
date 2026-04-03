#include "matmul.h"
#include <cassert>

namespace matmul {

/**
 * @brief Loop Reordered Matrix Multiplication (row-inner-col loop order)
 *
 * By changing the loop order from row-col-inner to row-inner-col, we
 * dramatically improve memory access patterns. This is the simplest yet most
 * effective software optimization for matrix multiplication.
 *
 * Why it's faster:
 * - Spatial Locality: The innermost loop is now 'col'. As 'col' increments, we
 *   access B[inner * columns + col] and C[row * columns + col] sequentially.
 *   Since matrices are row-major, these sequential accesses perfectly align
 *   with CPU cache lines.
 * - When memory is read sequentially, the CPU prefetcher can efficiently load
 *   upcoming data into the cache before it's needed, minimizing stall times.
 */
void LoopReorder(const double *A, const double *B, double *C, int rows,
                 int columns, int inners) {
  assert(rows == columns);
  assert(columns == inners);
  const int size = rows;

  for (int row = 0; row < size; ++row) {
    for (int inner = 0; inner < size; ++inner) {
      // a_val is loop-invariant for the innermost 'col' loop,
      // so it's loaded once and kept in a CPU register.
      double a_val = A[row * size + inner];
      for (int col = 0; col < size; ++col) {
        C[row * size + col] += a_val * B[inner * size + col];
      }
    }
  }
}

} // namespace matmul
