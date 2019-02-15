licenses(["notice"])  # Apache 2.0

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

package(
    default_visibility = [
        "//tensorflow_model_analysis/frontend:__subpackages__",
    ],
)

web_library(
    name = "lib",
    srcs = [
        "tfma-array-value.html",
        "tfma-array-value.js",
    ],
    path = "/tfma-array-value",
    deps = [
        "@org_polymer",
        "@org_polymer_paper_tooltip",
    ],
)
