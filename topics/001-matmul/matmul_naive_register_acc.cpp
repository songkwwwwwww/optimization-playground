#include "matmul.h"
#include <cassert>

namespace matmul {

void NaiveRegisterAcc(const double *A, const double *B, double *C, int rows,
                      int columns, int inners) {
  assert(rows == columns);
  assert(columns == inners);
  const int size = rows;

  for (int row = 0; row < size; ++row) {
    for (int col = 0; col < size; ++col) {
      double sum = 0.0;
      for (int inner = 0; inner < size; ++inner) {
        sum += A[row * size + inner] * B[inner * size + col];
      }
      C[row * size + col] = sum;
    }
  }
}

} // namespace matmul
