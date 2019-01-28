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
        "tfma-nb-slicing-metrics.html",
    ],
    path = "/tfma-nb-slicing-metrics",
    deps = [
        "//tensorflow_model_analysis/frontend/tfma-slicing-metrics-browser:lib",
        "@org_polymer",
    ],
)
