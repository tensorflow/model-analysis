licenses(["notice"])  # Apache 2.0

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

package(
    default_visibility = ["//tensorflow_model_analysis:__subpackages__"],
)

web_library(
    name = "lib",
    srcs = [
        "tfma-roc-curve.html",
        "tfma-roc-curve.js",
    ],
    path = "/tfma-roc-curve",
    deps = [
        "@org_googlewebcomponents_google_chart",
        "@org_polymer",
    ],
)
