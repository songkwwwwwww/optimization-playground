load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

COMMON_COPTS = [
    "-std=c++23",
    "-fno-builtin",
]

cc_library(
    name = "scan_lib",
    srcs = ["scan.cpp"],
    hdrs = ["scan.h"],
    copts = COMMON_COPTS,
)

cc_test(
    name = "scan_test",
    srcs = ["scan_test.cpp"],
    copts = COMMON_COPTS,
    deps = [
        ":scan_lib",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "scan_bench",
    srcs = ["scan_bench.cpp"],
    copts = COMMON_COPTS,
    deps = [
        ":scan_lib",
        "@google_benchmark//:benchmark",
    ],
)
