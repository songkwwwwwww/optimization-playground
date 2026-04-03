def _libomp_repo_rule_impl(repository_ctx):
    repository_ctx.symlink("/opt/homebrew/opt/libomp", "libomp_dir")
    repository_ctx.file("BUILD.bazel", """
cc_library(
    name = "omp",
    hdrs = glob(["libomp_dir/include/**/*.h"]),
    includes = ["libomp_dir/include"],
    srcs = ["libomp_dir/lib/libomp.dylib"],
    visibility = ["//visibility:public"],
)
""")
    repository_ctx.file("WORKSPACE", "")

libomp_repo_rule = repository_rule(
    implementation = _libomp_repo_rule_impl,
    local = True,
)

def _libomp_ext_impl(module_ctx):
    libomp_repo_rule(name = "libomp")

libomp_ext = module_extension(
    implementation = _libomp_ext_impl,
)
