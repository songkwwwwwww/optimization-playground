load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

COMMON_COPTS = [
    "-std=c++23",
    "-fno-builtin",
]

cc_library(
    name = "count_even_lib",
    srcs = ["count_even.cpp"],
    hdrs = ["count_even.h"],
    copts = COMMON_COPTS,
)

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

cc_test(
    name = "count_even_test",
    srcs = ["count_even_test.cpp"],
    copts = COMMON_COPTS,
    deps = [
        ":count_even_lib",
        "@googletest//:gtest_main",
    ],
)

test_suite(
    name = "experiment1_search_test",
    tests = [":scan_test"],
)

test_suite(
    name = "experiment2_count_test",
    tests = [":count_even_test"],
)

test_suite(
    name = "all_tests",
    tests = [
        ":experiment1_search_test",
        ":experiment2_count_test",
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

alias(
    name = "experiment1_search_bench",
    actual = ":scan_bench",
)

cc_binary(
    name = "count_even_bench",
    srcs = ["count_even_bench.cpp"],
    copts = COMMON_COPTS,
    deps = [
        ":count_even_lib",
        "@google_benchmark//:benchmark",
    ],
)

alias(
    name = "experiment2_count_bench",
    actual = ":count_even_bench",
)
