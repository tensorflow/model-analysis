package(default_visibility = ["//tensorflow_model_analysis:__subpackages__"])

licenses(["notice"])  # Apache 2.0

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("@org_tensorflow_tensorboard//tensorboard/defs:vulcanize.bzl", "tensorboard_html_binary")

web_library(
    name = "index",
    srcs = [
        "index.html",
        "//tensorflow_model_analysis/frontend/lib:js-lib-bin",
    ],
    path = "/tfma-int64",
    deps = [
        "//tensorflow_model_analysis/frontend/tfma-int64:lib",
    ],
)

tensorboard_html_binary(
    name = "demo",
    testonly = 1,
    input_path = "/tfma-int64/index.html",
    output_path = "/index.html",
    deps = [
        ":index",
        "//tensorflow_model_analysis/frontend/lib:externs",
    ],
)
