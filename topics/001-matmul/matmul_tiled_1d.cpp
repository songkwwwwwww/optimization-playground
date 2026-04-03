#include "matmul.h"
#include <algorithm>
#include <cassert>

namespace matmul {

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
void Tiled1D(const double *A, const double *B, double *C, int rows, int columns,
             int inners) {
  assert(rows == columns);
  assert(columns == inners);

  for (int inner_tile = 0; inner_tile < inners;
       inner_tile += kDefaultTileSize) {
    for (int row = 0; row < rows; row++) {
      int inner_tile_end = std::min(inners, inner_tile + kDefaultTileSize);
      for (int inner = inner_tile; inner < inner_tile_end; inner++) {
        double a = A[row * inners + inner];
        for (int column = 0; column < columns; column++) {
          C[row * columns + column] += a * B[inner * columns + column];
        }
      }
    }
  }
}

} // namespace matmul
