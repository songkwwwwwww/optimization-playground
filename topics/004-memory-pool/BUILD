load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

COMMON_COPTS = [
    "-std=c++23",
    "-faligned-allocation",
    "-mmacosx-version-min=11.0",
]

cc_library(
    name = "object_pool_lib",
    hdrs = ["object_pool.h"],
    copts = COMMON_COPTS,
)

cc_test(
    name = "object_pool_test",
    srcs = ["object_pool_test.cpp"],
    copts = COMMON_COPTS,
    deps = [
        ":object_pool_lib",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "object_pool_bench",
    srcs = ["object_pool_bench.cpp"],
    copts = COMMON_COPTS,
    deps = [
        ":object_pool_lib",
        "@google_benchmark//:benchmark",
    ],
)
