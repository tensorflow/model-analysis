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
        "tfma-value-at-cutoffs.html",
        "tfma-value-at-cutoffs.js",
    ],
    path = "/tfma-value-at-cutoffs",
    deps = [
        "@org_polymer",
    ],
)
