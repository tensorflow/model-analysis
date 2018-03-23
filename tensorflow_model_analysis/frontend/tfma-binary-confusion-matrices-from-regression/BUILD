licenses(["notice"])  # Apache 2.0

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

package(
    default_visibility = ["//tensorflow_model_analysis:__subpackages__"],
)

web_library(
    name = "lib",
    srcs = [
        "tfma-binary-confusion-matrices-from-regression.html",
        "tfma-binary-confusion-matrices-from-regression.js",
    ],
    path = "/tfma-binary-confusion-matrices-from-regression",
    deps = [
        "@org_polymer",
    ],
)
