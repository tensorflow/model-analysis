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
        "tfma-time-series-browser.html",
        "tfma-time-series-browser.js",
    ],
    path = "/tfma-time-series-browser",
    deps = [
        "//tensorflow_model_analysis/frontend/tfma-line-chart-grid:lib",
        "//tensorflow_model_analysis/frontend/tfma-metrics-table:lib",
        "@org_googlewebcomponents_google_chart",
        "@org_polymer",
    ],
)
