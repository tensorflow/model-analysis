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
        "tfma-nb-plot.html",
        "tfma-nb-plot.js",
    ],
    path = "/tfma-nb-plot",
    deps = [
        "//tensorflow_model_analysis/frontend/tfma-plot:lib",
        "@org_polymer",
    ],
)
