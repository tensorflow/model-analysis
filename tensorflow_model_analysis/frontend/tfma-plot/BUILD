licenses(["notice"])  # Apache 2.0

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

package(
    default_visibility = ["//tensorflow_model_analysis:__subpackages__"],
)

web_library(
    name = "lib",
    srcs = [
        "tfma-plot.html",
        "tfma-plot.js",
    ],
    path = "/tfma-plot",
    deps = [
        "//tensorflow_model_analysis/frontend/tfma-calibration-plot:lib",
        "//tensorflow_model_analysis/frontend/tfma-precision-recall-curve:lib",
        "//tensorflow_model_analysis/frontend/tfma-prediction-distribution:lib",
        "//tensorflow_model_analysis/frontend/tfma-roc-curve:lib",
        "//tensorflow_model_analysis/frontend/tfma-shared-styles:lib",
        "@org_polymer",
        "@org_polymer_iron_pages",
        "@org_polymer_paper_button",
        "@org_polymer_paper_spinner",
        "@org_polymer_paper_tabs",
    ],
)
