licenses(["notice"])  # Apache 2.0

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

package(
    default_visibility = ["//tensorflow_model_analysis:__subpackages__"],
)

web_library(
    name = "lib",
    srcs = [
        "tfma-prediction-distribution.html",
        "tfma-prediction-distribution.js",
    ],
    path = "/tfma-prediction-distribution",
    deps = [
        "@org_googlewebcomponents_google_chart",
        "@org_polymer",
    ],
)
