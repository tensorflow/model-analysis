package(default_visibility = ["//tensorflow_model_analysis:__subpackages__"])

licenses(["notice"])  # Apache 2.0

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("@org_tensorflow_tensorboard//tensorboard/defs:vulcanize.bzl", "tensorboard_html_binary")

web_library(
    name = "index",
    srcs = [
        "index.html",
        "index.js",
        "//tensorflow_model_analysis/frontend/lib:js-lib-bin",
    ],
    path = "/tfma-metrics-table",
    deps = [
        "//tensorflow_model_analysis/frontend/tfma-metrics-table:lib",
    ],
)

tensorboard_html_binary(
    name = "demo",
    testonly = 1,
    input_path = "/tfma-metrics-table/index.html",
    output_path = "/index.html",
    deps = [
        ":index",
        "//tensorflow_model_analysis/frontend/lib:externs",
    ],
)
