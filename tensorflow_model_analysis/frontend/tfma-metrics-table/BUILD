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
        "tfma-metrics-table.html",
        "tfma-metrics-table.js",
    ],
    path = "/tfma-metrics-table",
    deps = [
        "//tensorflow_model_analysis/frontend/tfma-array-value:lib",
        "//tensorflow_model_analysis/frontend/tfma-bounded-value:lib",
        "//tensorflow_model_analysis/frontend/tfma-confusion-matrix-at-thresholds:lib",
        "//tensorflow_model_analysis/frontend/tfma-int64:lib",
        "//tensorflow_model_analysis/frontend/tfma-multi-class-confusion-matrix:lib",
        "//tensorflow_model_analysis/frontend/tfma-value-at-cutoffs:lib",
        "@org_googlewebcomponents_google_chart",
        "@org_polymer",
        "@org_polymer_iron_icons",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_styles",
    ],
)
