#include "matmul.h"
#include <cassert>
#include <cstring>
#include <omp.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace matmul {
namespace {

// These OpenMP study variants are intentionally specialized for one fixed
// workload so the code can focus on the optimization idea itself.
constexpr int kMatrixSize = 2048;
constexpr int kTileSize = 128;
constexpr int kRegisterRows = 4;
constexpr int kRegisterCols = 8;

static_assert(kMatrixSize % kTileSize == 0);
static_assert(kTileSize % kRegisterRows == 0);
static_assert(kTileSize % kRegisterCols == 0);

using TileKernel = void (*)(const double *A, const double *B, double *C,
                            int row_block, int col_block, int inner_block);
using PackedKernel = void (*)(const double *a_packed, const double *b_packed,
                              double *c_block);

inline void CheckMatrixSize(int rows, int columns, int inners) {
  assert(rows == kMatrixSize);
  assert(columns == kMatrixSize);
  assert(inners == kMatrixSize);
}

inline void MultiplyTileScalar(const double *A, const double *B, double *C,
                               int row_block, int col_block, int inner_block) {
  for (int row = 0; row < kTileSize; ++row) {
    double *c_row = &C[(row_block + row) * kMatrixSize + col_block];
    for (int inner = 0; inner < kTileSize; ++inner) {
      const double a_value =
          A[(row_block + row) * kMatrixSize + inner_block + inner];
      const double *b_row = &B[(inner_block + inner) * kMatrixSize + col_block];
      for (int col = 0; col < kTileSize; ++col) {
        c_row[col] += a_value * b_row[col];
      }
    }
  }
}

inline void MultiplyTileSimd(const double *A, const double *B, double *C,
                             int row_block, int col_block, int inner_block) {
  for (int row = 0; row < kTileSize; ++row) {
    double *c_row = &C[(row_block + row) * kMatrixSize + col_block];
    for (int inner = 0; inner < kTileSize; ++inner) {
      const double a_value =
          A[(row_block + row) * kMatrixSize + inner_block + inner];
      const double *b_row = &B[(inner_block + inner) * kMatrixSize + col_block];
#if defined(__ARM_NEON)
      const float64x2_t a_vec = vmovq_n_f64(a_value);
      for (int col = 0; col < kTileSize; col += 2) {
        float64x2_t b_vec = vld1q_f64(&b_row[col]);
        float64x2_t c_vec = vld1q_f64(&c_row[col]);
        c_vec = vmlaq_f64(c_vec, a_vec, b_vec);
        vst1q_f64(&c_row[col], c_vec);
      }
#else
      for (int col = 0; col < kTileSize; ++col) {
        c_row[col] += a_value * b_row[col];
      }
#endif
    }
  }
}

inline void PackATile(double *dest, const double *src, int row_block,
                      int inner_block) {
  for (int row = 0; row < kTileSize; ++row) {
    std::memcpy(&dest[row * kTileSize],
                &src[(row_block + row) * kMatrixSize + inner_block],
                kTileSize * sizeof(double));
  }
}

inline void PackBTile(double *dest, const double *src, int inner_block,
                      int col_block) {
  for (int inner = 0; inner < kTileSize; ++inner) {
    std::memcpy(&dest[inner * kTileSize],
                &src[(inner_block + inner) * kMatrixSize + col_block],
                kTileSize * sizeof(double));
  }
}

inline void MultiplyPackedTileScalar(const double *a_packed,
                                     const double *b_packed, double *c_block) {
  for (int row = 0; row < kTileSize; ++row) {
    const double *a_row = &a_packed[row * kTileSize];
    double *c_row = &c_block[row * kMatrixSize];
    for (int inner = 0; inner < kTileSize; ++inner) {
      const double a_value = a_row[inner];
      const double *b_row = &b_packed[inner * kTileSize];
      for (int col = 0; col < kTileSize; ++col) {
        c_row[col] += a_value * b_row[col];
      }
    }
  }
}

inline void MultiplyPackedTileSimd(const double *a_packed,
                                   const double *b_packed, double *c_block) {
  a_packed =
      static_cast<const double *>(__builtin_assume_aligned(a_packed, 64));
  b_packed =
      static_cast<const double *>(__builtin_assume_aligned(b_packed, 64));

  for (int row = 0; row < kTileSize; ++row) {
    const double *a_row = &a_packed[row * kTileSize];
    double *c_row = &c_block[row * kMatrixSize];
    for (int inner = 0; inner < kTileSize; ++inner) {
      const double a_value = a_row[inner];
      const double *b_row = &b_packed[inner * kTileSize];
#if defined(__ARM_NEON)
      const float64x2_t a_vec = vmovq_n_f64(a_value);
      for (int col = 0; col < kTileSize; col += 2) {
        float64x2_t b_vec = vld1q_f64(&b_row[col]);
        float64x2_t c_vec = vld1q_f64(&c_row[col]);
        c_vec = vmlaq_f64(c_vec, a_vec, b_vec);
        vst1q_f64(&c_row[col], c_vec);
      }
#else
      for (int col = 0; col < kTileSize; ++col) {
        c_row[col] += a_value * b_row[col];
      }
#endif
    }
  }
}

inline void PackARegisterPanel(double *dest, const double *src, int row_block,
                               int inner_block) {
  for (int row = 0; row < kRegisterRows; ++row) {
    std::memcpy(&dest[row * kTileSize],
                &src[(row_block + row) * kMatrixSize + inner_block],
                kTileSize * sizeof(double));
  }
}

inline void PackBRegisterPanel(double *dest, const double *src, int inner_block,
                               int col_block) {
  for (int inner = 0; inner < kTileSize; ++inner) {
    std::memcpy(&dest[inner * kRegisterCols],
                &src[(inner_block + inner) * kMatrixSize + col_block],
                kRegisterCols * sizeof(double));
  }
}

[[maybe_unused]] inline void
MultiplyPackedRegister4x8Scalar(const double *a_panel, const double *b_panel,
                                double *c_block) {
  for (int row = 0; row < kRegisterRows; ++row) {
    const double *a_row = &a_panel[row * kTileSize];
    double *c_row = &c_block[row * kMatrixSize];
    for (int inner = 0; inner < kTileSize; ++inner) {
      const double a_value = a_row[inner];
      const double *b_row = &b_panel[inner * kRegisterCols];
      for (int col = 0; col < kRegisterCols; ++col) {
        c_row[col] += a_value * b_row[col];
      }
    }
  }
}

inline void MultiplyPackedRegister4x8(const double *a_panel,
                                      const double *b_panel, double *c_block) {
  a_panel = static_cast<const double *>(__builtin_assume_aligned(a_panel, 64));
  b_panel = static_cast<const double *>(__builtin_assume_aligned(b_panel, 64));

#if defined(__ARM_NEON)
  const double *a_row0 = &a_panel[0];
  const double *a_row1 = &a_panel[kTileSize];
  const double *a_row2 = &a_panel[2 * kTileSize];
  const double *a_row3 = &a_panel[3 * kTileSize];

  float64x2_t c00 = vld1q_f64(&c_block[0]);
  float64x2_t c01 = vld1q_f64(&c_block[2]);
  float64x2_t c02 = vld1q_f64(&c_block[4]);
  float64x2_t c03 = vld1q_f64(&c_block[6]);
  float64x2_t c10 = vld1q_f64(&c_block[kMatrixSize + 0]);
  float64x2_t c11 = vld1q_f64(&c_block[kMatrixSize + 2]);
  float64x2_t c12 = vld1q_f64(&c_block[kMatrixSize + 4]);
  float64x2_t c13 = vld1q_f64(&c_block[kMatrixSize + 6]);
  float64x2_t c20 = vld1q_f64(&c_block[2 * kMatrixSize + 0]);
  float64x2_t c21 = vld1q_f64(&c_block[2 * kMatrixSize + 2]);
  float64x2_t c22 = vld1q_f64(&c_block[2 * kMatrixSize + 4]);
  float64x2_t c23 = vld1q_f64(&c_block[2 * kMatrixSize + 6]);
  float64x2_t c30 = vld1q_f64(&c_block[3 * kMatrixSize + 0]);
  float64x2_t c31 = vld1q_f64(&c_block[3 * kMatrixSize + 2]);
  float64x2_t c32 = vld1q_f64(&c_block[3 * kMatrixSize + 4]);
  float64x2_t c33 = vld1q_f64(&c_block[3 * kMatrixSize + 6]);

  for (int inner = 0; inner < kTileSize; ++inner) {
    const double *b_row = &b_panel[inner * kRegisterCols];
    const float64x2_t b0 = vld1q_f64(&b_row[0]);
    const float64x2_t b1 = vld1q_f64(&b_row[2]);
    const float64x2_t b2 = vld1q_f64(&b_row[4]);
    const float64x2_t b3 = vld1q_f64(&b_row[6]);

    const float64x2_t a0 = vmovq_n_f64(a_row0[inner]);
    const float64x2_t a1 = vmovq_n_f64(a_row1[inner]);
    const float64x2_t a2 = vmovq_n_f64(a_row2[inner]);
    const float64x2_t a3 = vmovq_n_f64(a_row3[inner]);

    c00 = vmlaq_f64(c00, a0, b0);
    c01 = vmlaq_f64(c01, a0, b1);
    c02 = vmlaq_f64(c02, a0, b2);
    c03 = vmlaq_f64(c03, a0, b3);
    c10 = vmlaq_f64(c10, a1, b0);
    c11 = vmlaq_f64(c11, a1, b1);
    c12 = vmlaq_f64(c12, a1, b2);
    c13 = vmlaq_f64(c13, a1, b3);
    c20 = vmlaq_f64(c20, a2, b0);
    c21 = vmlaq_f64(c21, a2, b1);
    c22 = vmlaq_f64(c22, a2, b2);
    c23 = vmlaq_f64(c23, a2, b3);
    c30 = vmlaq_f64(c30, a3, b0);
    c31 = vmlaq_f64(c31, a3, b1);
    c32 = vmlaq_f64(c32, a3, b2);
    c33 = vmlaq_f64(c33, a3, b3);
  }

  vst1q_f64(&c_block[0], c00);
  vst1q_f64(&c_block[2], c01);
  vst1q_f64(&c_block[4], c02);
  vst1q_f64(&c_block[6], c03);
  vst1q_f64(&c_block[kMatrixSize + 0], c10);
  vst1q_f64(&c_block[kMatrixSize + 2], c11);
  vst1q_f64(&c_block[kMatrixSize + 4], c12);
  vst1q_f64(&c_block[kMatrixSize + 6], c13);
  vst1q_f64(&c_block[2 * kMatrixSize + 0], c20);
  vst1q_f64(&c_block[2 * kMatrixSize + 2], c21);
  vst1q_f64(&c_block[2 * kMatrixSize + 4], c22);
  vst1q_f64(&c_block[2 * kMatrixSize + 6], c23);
  vst1q_f64(&c_block[3 * kMatrixSize + 0], c30);
  vst1q_f64(&c_block[3 * kMatrixSize + 2], c31);
  vst1q_f64(&c_block[3 * kMatrixSize + 4], c32);
  vst1q_f64(&c_block[3 * kMatrixSize + 6], c33);
#else
  MultiplyPackedRegister4x8Scalar(a_panel, b_panel, c_block);
#endif
}

void RunOmpTiled(const double *A, const double *B, double *C,
                 TileKernel kernel) {
#pragma omp parallel for default(none) shared(A, B, C, kernel) collapse(2)     \
    schedule(static)
  for (int row_block = 0; row_block < kMatrixSize; row_block += kTileSize) {
    for (int col_block = 0; col_block < kMatrixSize; col_block += kTileSize) {
      for (int inner_block = 0; inner_block < kMatrixSize;
           inner_block += kTileSize) {
        kernel(A, B, C, row_block, col_block, inner_block);
      }
    }
  }
}

void RunOmpPacked(const double *A, const double *B, double *C,
                  PackedKernel kernel) {
#pragma omp parallel default(none) shared(A, B, C, kernel)
  {
    alignas(64) double a_packed[kTileSize * kTileSize];
    alignas(64) double b_packed[kTileSize * kTileSize];

#pragma omp for collapse(2) schedule(static)
    for (int row_block = 0; row_block < kMatrixSize; row_block += kTileSize) {
      for (int col_block = 0; col_block < kMatrixSize; col_block += kTileSize) {
        double *c_block = &C[row_block * kMatrixSize + col_block];
        for (int inner_block = 0; inner_block < kMatrixSize;
             inner_block += kTileSize) {
          PackATile(a_packed, A, row_block, inner_block);
          PackBTile(b_packed, B, inner_block, col_block);
          kernel(a_packed, b_packed, c_block);
        }
      }
    }
  }
}

void RunOmpPackedRow(const double *A, const double *B, double *C,
                     PackedKernel kernel) {
#pragma omp parallel default(none) shared(A, B, C, kernel)
  {
    alignas(64) double a_packed[kTileSize * kTileSize];
    alignas(64) double b_packed[kTileSize * kTileSize];

    // collapse(2) 대신 row_block만 병렬화
#pragma omp for schedule(static)
    for (int row_block = 0; row_block < kMatrixSize; row_block += kTileSize) {
      for (int col_block = 0; col_block < kMatrixSize; col_block += kTileSize) {
        double *c_block = &C[row_block * kMatrixSize + col_block];
        for (int inner_block = 0; inner_block < kMatrixSize;
             inner_block += kTileSize) {
          PackATile(a_packed, A, row_block, inner_block);
          PackBTile(b_packed, B, inner_block, col_block);
          kernel(a_packed, b_packed, c_block);
        }
      }
    }
  }
}

void RunOmpPackedRegister(const double *A, const double *B, double *C) {
#pragma omp parallel default(none) shared(A, B, C)
  {
    alignas(64) double a_panel[kRegisterRows * kTileSize];
    alignas(64) double b_panel[kTileSize * kRegisterCols];

#pragma omp for collapse(2) schedule(static)
    for (int row_block = 0; row_block < kMatrixSize; row_block += kTileSize) {
      for (int col_block = 0; col_block < kMatrixSize; col_block += kTileSize) {
        for (int inner_block = 0; inner_block < kMatrixSize;
             inner_block += kTileSize) {
          for (int col_micro = col_block; col_micro < col_block + kTileSize;
               col_micro += kRegisterCols) {
            PackBRegisterPanel(b_panel, B, inner_block, col_micro);
            for (int row_micro = row_block; row_micro < row_block + kTileSize;
                 row_micro += kRegisterRows) {
              PackARegisterPanel(a_panel, A, row_micro, inner_block);
              MultiplyPackedRegister4x8(
                  a_panel, b_panel, &C[row_micro * kMatrixSize + col_micro]);
            }
          }
        }
      }
    }
  }
}

} // namespace

void OmpThread(const double *A, const double *B, double *C, int rows,
               int columns, int inners) {
  CheckMatrixSize(rows, columns, inners);
  RunOmpTiled(A, B, C, MultiplyTileScalar);
}

void OmpThreadSimd(const double *A, const double *B, double *C, int rows,
                   int columns, int inners) {
  CheckMatrixSize(rows, columns, inners);
  RunOmpTiled(A, B, C, MultiplyTileSimd);
}

void OmpThreadPacked(const double *A, const double *B, double *C, int rows,
                     int columns, int inners) {
  CheckMatrixSize(rows, columns, inners);
  RunOmpPacked(A, B, C, MultiplyPackedTileScalar);
}

void OmpThreadPackedSimd(const double *A, const double *B, double *C, int rows,
                         int columns, int inners) {
  CheckMatrixSize(rows, columns, inners);
  RunOmpPacked(A, B, C, MultiplyPackedTileSimd);
}

void OmpThreadPackedRow(const double *A, const double *B, double *C, int rows,
                        int columns, int inners) {
  CheckMatrixSize(rows, columns, inners);
  RunOmpPackedRow(A, B, C, MultiplyPackedTileSimd);
}

void OmpThreadPackedRegister(const double *A, const double *B, double *C,
                             int rows, int columns, int inners) {
  CheckMatrixSize(rows, columns, inners);
  RunOmpPackedRegister(A, B, C);
}

} // namespace matmul
