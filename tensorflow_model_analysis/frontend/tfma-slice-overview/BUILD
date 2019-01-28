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
        "tfma-slice-overview.html",
        "tfma-slice-overview.js",
    ],
    path = "/tfma-slice-overview",
    deps = [
        "@org_googlewebcomponents_google_chart",
        "@org_polymer",
        "@org_polymer_paper_dropdown_menu",
        "@org_polymer_paper_item",
        "@org_polymer_paper_listbox",
    ],
)
