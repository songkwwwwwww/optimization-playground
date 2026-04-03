#include "matmul.h"
#include <cassert>

namespace matmul {

namespace {

inline void MultiplyTile(const double *A, const double *B, double *C, int size,
                         int row_block, int col_block, int inner_block) {
  for (int row = 0; row < kDefaultTileSize; ++row) {
    const double *a_row = &A[(row_block + row) * size + inner_block];
    double *c_row = &C[(row_block + row) * size + col_block];
    for (int inner = 0; inner < kDefaultTileSize; ++inner) {
      const double a_value = a_row[inner];
      const double *b_row = &B[(inner_block + inner) * size + col_block];
      for (int col = 0; col < kDefaultTileSize; ++col) {
        c_row[col] += a_value * b_row[col];
      }
    }
  }
}

} // namespace

/**
 * @brief Tiled (Blocked) Matrix Multiplication
 *
 * Tiling or "Loop Blocking" improves Temporal Locality (data reuse in the
 * cache). Instead of computing the entire matrix at once, we compute it in
 * small blocks (tiles) that fit entirely within the CPU's fast L1 or L2 cache.
 *
 * Concept:
 * - When matrices are too large, elements fetched into the cache are evicted
 *   before they can be reused.
 * - By dividing matrices into TILE_SIZE x TILE_SIZE blocks, we ensure that
 *   the working set fits in the cache. The CPU can perform many arithmetic
 *   operations on the cached block before fetching the next one.
 *
 * Note: TILE_SIZE should ideally be tuned based on the target machine's L1/L2
 * cache size.
 */
void TiledMD(const double *A, const double *B, double *C, int rows, int columns,
             int inners) {
  assert(rows == columns);
  assert(columns == inners);
  const int size = rows;
  assert(size % kDefaultTileSize == 0);

  for (int row_block = 0; row_block < size; row_block += kDefaultTileSize) {
    for (int inner_block = 0; inner_block < size;
         inner_block += kDefaultTileSize) {
      for (int col_block = 0; col_block < size; col_block += kDefaultTileSize) {
        MultiplyTile(A, B, C, size, row_block, col_block, inner_block);
      }
    }
  }
}

} // namespace matmul
