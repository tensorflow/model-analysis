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
        "tfma-precision-recall-curve.html",
        "tfma-precision-recall-curve.js",
    ],
    path = "/tfma-precision-recall-curve",
    deps = [
        "@org_googlewebcomponents_google_chart",
        "@org_polymer",
    ],
)
