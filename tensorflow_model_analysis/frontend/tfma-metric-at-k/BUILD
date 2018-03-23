licenses(["notice"])  # Apache 2.0

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

package(
    default_visibility = ["//tensorflow_model_analysis:__subpackages__"],
)

web_library(
    name = "lib",
    srcs = [
        "tfma-metric-at-k.html",
        "tfma-metric-at-k.js",
    ],
    path = "/tfma-metric-at-k",
    deps = [
        "@org_polymer",
    ],
)
