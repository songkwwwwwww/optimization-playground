#include "matmul.h"
#include "matrix_utils.h"
#include <algorithm>
#include <gtest/gtest.h>

using namespace matmul;

/**
 * @brief Test fixture for Matrix Multiplication implementations
 *
 * Sets up the required matrices before each test and cleans them up afterwards.
 * Uses a reference BLAS implementation to verify the correctness of our custom
 * matrix multiplication algorithms.
 */
class MatmulTest : public ::testing::Test {
protected:
  void SetUp() override {
    // 256 = 64 * 4, so the simplified tile-based study variants can stay free
    // of remainder-handling code while the test still runs quickly.
    rows = 256;
    columns = 256;
    inners = 256;

    // Allocate memory with proper alignment for SIMD testing
    A = AllocateAligned(rows * inners);
    B = AllocateAligned(inners * columns);
    C = AllocateAligned(rows * columns);
    refC = AllocateAligned(rows * columns);

    // Initialize inputs with random values
    InitializeRandom(A, rows * inners);
    InitializeRandom(B, inners * columns);

    // Generate the "golden" reference results using BLAS
    Reference(A, B, refC, rows, columns, inners);
  }

  void TearDown() override {
    FreeAligned(A);
    FreeAligned(B);
    FreeAligned(C);
    FreeAligned(refC);
  }

  int rows, columns, inners;
  double *A, *B, *C, *refC;
};

// Test definitions for each optimization technique

TEST_F(MatmulTest, NaiveCorrectness) {
  Naive(A, B, C, rows, columns, inners);
  EXPECT_TRUE(VerifyResults(C, refC, rows * columns));
}

TEST_F(MatmulTest, NaiveRegisterAccCorrectness) {
  NaiveRegisterAcc(A, B, C, rows, columns, inners);
  EXPECT_TRUE(VerifyResults(C, refC, rows * columns));
}

TEST_F(MatmulTest, LoopReorderCorrectness) {
  LoopReorder(A, B, C, rows, columns, inners);
  EXPECT_TRUE(VerifyResults(C, refC, rows * columns));
}

TEST_F(MatmulTest, Tiled1DCorrectness) {
  Tiled1D(A, B, C, rows, columns, inners);
  EXPECT_TRUE(VerifyResults(C, refC, rows * columns));
}

TEST_F(MatmulTest, TiledMDCorrectness) {
  TiledMD(A, B, C, rows, columns, inners);
  EXPECT_TRUE(VerifyResults(C, refC, rows * columns));
}

TEST_F(MatmulTest, SIMDCorrectness) {
  Simd(A, B, C, rows, columns, inners);
  EXPECT_TRUE(VerifyResults(C, refC, rows * columns));
}

TEST_F(MatmulTest, PackedCorrectness) {
  Packed(A, B, C, rows, columns, inners);
  EXPECT_TRUE(VerifyResults(C, refC, rows * columns));
}

TEST(MatmulOpenMpStudyTest, Fixed2048SquareMatrixCorrectness) {
  constexpr int rows = 2048;
  constexpr int columns = 2048;
  constexpr int inners = 2048;

  double *A = AllocateAligned(rows * inners);
  double *B = AllocateAligned(inners * columns);
  double *C = AllocateAligned(rows * columns);
  double *refC = AllocateAligned(rows * columns);

  InitializeRandom(A, rows * inners);
  InitializeRandom(B, inners * columns);
  Reference(A, B, refC, rows, columns, inners);

  struct Variant {
    const char *name;
    MatmulFunc func;
  };

  const Variant variants[] = {
      {"OmpThread", OmpThread},
      {"OmpThreadSimd", OmpThreadSimd},
      {"OmpThreadPacked", OmpThreadPacked},
      {"OmpThreadPackedSimd", OmpThreadPackedSimd},
      {"OmpThreadPackedRow", OmpThreadPackedRow},
      {"OmpThreadPackedRegister", OmpThreadPackedRegister},
  };

  for (const Variant &variant : variants) {
    SCOPED_TRACE(variant.name);
    std::fill_n(C, rows * columns, 0.0);
    variant.func(A, B, C, rows, columns, inners);
    EXPECT_TRUE(VerifyResults(C, refC, rows * columns));
  }

  FreeAligned(A);
  FreeAligned(B);
  FreeAligned(C);
  FreeAligned(refC);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
