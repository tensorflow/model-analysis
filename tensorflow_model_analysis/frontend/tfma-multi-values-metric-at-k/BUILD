licenses(["notice"])  # Apache 2.0

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

package(
    default_visibility = ["//tensorflow_model_analysis:__subpackages__"],
)

web_library(
    name = "lib",
    srcs = [
        "tfma-multi-values-metric-at-k.html",
        "tfma-multi-values-metric-at-k.js",
    ],
    path = "/tfma-multi-values-metric-at-k",
    deps = [
        "@org_polymer",
    ],
)
