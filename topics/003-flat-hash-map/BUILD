load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

COMMON_COPTS = [
    "-std=c++23",
]

cc_library(
    name = "hash_table_workloads",
    hdrs = ["hash_table_workloads.h"],
    copts = COMMON_COPTS,
)

cc_test(
    name = "flat_hash_map_test",
    srcs = ["flat_hash_map_test.cpp"],
    copts = COMMON_COPTS,
    deps = [
        ":hash_table_workloads",
        "@com_google_absl//absl/container:flat_hash_map",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "flat_hash_map_bench",
    srcs = ["flat_hash_map_bench.cpp"],
    copts = COMMON_COPTS,
    deps = [
        ":hash_table_workloads",
        "@com_google_absl//absl/container:flat_hash_map",
        "@google_benchmark//:benchmark",
    ],
)
