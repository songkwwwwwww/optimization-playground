load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

cc_library(
    name = "spsc_queue_lib",
    hdrs = ["spsc_queue.h"],
    copts = [
        "-std=c++23",
        "-faligned-allocation",
        "-mmacosx-version-min=11.0",
    ],
)

cc_test(
    name = "spsc_queue_test",
    srcs = ["spsc_queue_test.cpp"],
    copts = [
        "-std=c++23",
        "-faligned-allocation",
        "-mmacosx-version-min=11.0",
    ],
    deps = [
        ":spsc_queue_lib",
        "@googletest//:gtest_main",
    ],
)

cc_binary(
    name = "spsc_queue_bench",
    srcs = ["spsc_queue_bench.cpp"],
    copts = [
        "-std=c++23",
        "-faligned-allocation",
        "-mmacosx-version-min=11.0",
    ],
    deps = [
        ":spsc_queue_lib",
        "@google_benchmark//:benchmark",
    ],
)
