licenses(["notice"])  # Apache 2.0

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

package(
    default_visibility = ["//tensorflow_model_analysis:__subpackages__"],
)

web_library(
    name = "lib",
    srcs = [
        "tfma-calibration-plot.html",
        "tfma-calibration-plot.js",
    ],
    path = "/tfma-calibration-plot",
    deps = [
        "@org_googlewebcomponents_google_chart",
        "@org_polymer",
    ],
)
