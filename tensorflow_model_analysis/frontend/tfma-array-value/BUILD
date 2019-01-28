load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")
load("@org_tensorflow_tensorboard//tensorboard/defs:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = ["//tensorflow_model_analysis:__subpackages__"],
)

web_library(
    name = "lib",
    srcs = [
        "tfma-array-value.html",
        "tfma-array-value.js",
    ],
    path = "/tfma-array-value",
    deps = [
        "@org_polymer",
        "@org_polymer_paper_tooltip",
    ],
)

tensorboard_webcomponent_library(
    name = "internal-only-lib",
    srcs = [
        "tfma-array-value.html",
        "tfma-array-value.js",
    ],
    destdir = "tfma-array-value",
    deps = [
        "//third_party/javascript/google_chart/v1:lib",
        "//third_party/javascript/polymer/v1/paper-tooltip:lib",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)
