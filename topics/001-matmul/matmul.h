#ifndef MATMUL_H
#define MATMUL_H

namespace matmul {

/**
 * @brief Common interface for Matrix Multiplication: C = A * B
 *
 * This header defines the standard interface for various matrix multiplication
 * implementations. Matrix multiplication is a fundamental operation in machine
 * learning and high-performance computing.
 *
 * Matrix Details:
 * - A: rows x inners matrix (stored in row-major order)
 * - B: inners x columns matrix (stored in row-major order)
 * - C: rows x columns matrix (stored in row-major order)
 *
 * Row-major order means that elements of a row are stored in contiguous memory
 * locations. For example, A[row][inner] is accessed as
 * A[row * inners + inner].
 */
typedef void (*MatmulFunc)(const double *A, const double *B, double *C,
                           int rows, int columns, int inners);

// Common configuration for tile-based optimizations
constexpr int kDefaultTileSize = 64;

// Implementations of matrix multiplication with various optimization techniques
void Naive(const double *A, const double *B, double *C, int rows, int columns,
           int inners);
void NaiveRegisterAcc(const double *A, const double *B, double *C, int rows,
                      int columns, int inners);
void LoopReorder(const double *A, const double *B, double *C, int rows,
                 int columns, int inners);
void Tiled1D(const double *A, const double *B, double *C, int rows, int columns,
             int inners);
void TiledMD(const double *A, const double *B, double *C, int rows, int columns,
             int inners);
void Simd(const double *A, const double *B, double *C, int rows, int columns,
          int inners);
void Packed(const double *A, const double *B, double *C, int rows, int columns,
            int inners);
void OmpThread(const double *A, const double *B, double *C, int rows,
               int columns, int inners);
void OmpThreadSimd(const double *A, const double *B, double *C, int rows,
                   int columns, int inners);
void OmpThreadPacked(const double *A, const double *B, double *C, int rows,
                     int columns, int inners);
void OmpThreadPackedSimd(const double *A, const double *B, double *C, int rows,
                         int columns, int inners);
void OmpThreadPackedRow(const double *A, const double *B, double *C, int rows,
                        int columns, int inners);
void OmpThreadPackedRegister(const double *A, const double *B, double *C,
                             int rows, int columns, int inners);
void Reference(const double *A, const double *B, double *C, int rows,
               int columns, int inners);

} // namespace matmul

#endif // MATMUL_H
