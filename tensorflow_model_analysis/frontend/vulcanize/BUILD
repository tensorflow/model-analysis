package(default_visibility = ["//tensorflow_model_analysis:__subpackages__"])

licenses(["notice"])  # Apache 2.0

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("@org_tensorflow_tensorboard//tensorboard/defs:vulcanize.bzl", "tensorboard_html_binary")

web_library(
    name = "notebook",
    srcs = [
        "notebook.html",
        "//tensorflow_model_analysis/frontend/lib:js-lib-bin",
        "@org_d3js//:d3.min.js",
    ],
    path = "/tfma-notebook",
    deps = [
        "//tensorflow_model_analysis/frontend/tfma-nb-plot:lib",
        "//tensorflow_model_analysis/frontend/tfma-nb-slicing-metrics:lib",
        "//tensorflow_model_analysis/frontend/tfma-nb-time-series:lib",
    ],
)

tensorboard_html_binary(
    name = "vulcanized_template",
    compile = True,
    input_path = "/tfma-notebook/notebook.html",
    output_path = "/notebook.html",
    deps = [
        ":notebook",
        "//tensorflow_model_analysis/frontend/lib:externs",
    ],
)
