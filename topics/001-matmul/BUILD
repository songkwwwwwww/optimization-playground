load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@rules_python//python:defs.bzl", "py_binary")

cc_library(
    name = "matmul_lib",
    srcs = [
        "matmul_loop_reorder.cpp",
        "matmul_naive.cpp",
        "matmul_naive_register_acc.cpp",
        "matmul_omp.cpp",
        "matmul_packed.cpp",
        "matmul_reference.cpp",
        "matmul_simd.cpp",
        "matmul_tiled_1d.cpp",
        "matmul_tiled_md.cpp",
        "matrix_utils.cpp",
    ],
    hdrs = [
        "matmul.h",
        "matrix_utils.h",
    ],
    copts = [
        "-Xpreprocessor",
        "-fopenmp",
    ],
    linkopts = select({
        "@platforms//os:macos": ["-framework Accelerate"],
        "//conditions:default": ["-lopenblas"],
    }),
    deps = ["@libomp//:omp"],
)

cc_test(
    name = "matmul_test",
    srcs = ["matmul_test.cpp"],
    deps = [
        ":matmul_lib",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "matmul_bench",
    srcs = ["matmul_bench.cpp"],
    deps = [
        ":matmul_lib",
        "@google_benchmark//:benchmark",
    ],
)

cc_binary(
    name = "matmul_prof",
    srcs = ["main.cpp"],
    deps = [":matmul_lib"],
)
