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
        "tfma-bounded-value.html",
        "tfma-bounded-value.js",
    ],
    path = "/tfma-bounded-value",
    deps = [
        "@org_polymer",
    ],
)
