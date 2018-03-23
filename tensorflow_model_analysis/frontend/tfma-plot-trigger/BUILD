licenses(["notice"])  # Apache 2.0

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

package(
    default_visibility = ["//tensorflow_model_analysis:__subpackages__"],
)

web_library(
    name = "lib",
    srcs = [
        "tfma-plot-trigger.html",
        "tfma-plot-trigger.js",
    ],
    path = "/tfma-plot-trigger",
    deps = [
        "@org_polymer",
    ],
)
