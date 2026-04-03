#include "matmul.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
// Linux/OpenBLAS standard header (Requires BLAS library installation)
#include <cblas.h>
#endif

namespace matmul {

/**
 * @brief Reference Implementation (Using BLAS)
 *
 * BLAS (Basic Linear Algebra Subprograms) libraries (like OpenBLAS, Intel MKL,
 * or Apple Accelerate) provide heavily optimized, assembly-level
 * implementations of matrix operations. We use CBLAS `dgemm` (Double-precision
 * General Matrix Multiply) to establish the golden standard for both
 * correctness (reference output) and performance (baseline benchmark).
 */
void Reference(const double *A, const double *B, double *C, int rows,
               int columns, int inners) {
  // cblas_dgemm equation: C = alpha * (A * B) + beta * C
  // Parameters:
  // - CblasRowMajor: Our matrices are stored in row-major order.
  // - CblasNoTrans: We are not transposing A or B before multiplication.
  // - rows, columns, inners: Dimensions of the matrices.
  // - alpha: 1.0 (Multiply the result of A*B by 1)
  // - lda, ldb, ldc: Leading dimensions. For row-major, this is the number of
  // columns.
  // - beta: 0.0 (Do not add the previous values of C)
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, columns, inners,
              1.0, A, inners, B, columns, 0.0, C, columns);
}

} // namespace matmul
