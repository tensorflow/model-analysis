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
        "tfma-nb-time-series.html",
    ],
    path = "/tfma-nb-time-series",
    deps = [
        "//tensorflow_model_analysis/frontend/tfma-time-series-browser:lib",
        "@org_polymer",
    ],
)
