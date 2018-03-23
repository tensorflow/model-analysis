licenses(["notice"])  # Apache 2.0

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

package(
    default_visibility = ["//tensorflow_model_analysis:__subpackages__"],
)

web_library(
    name = "lib",
    srcs = [
        "tfma-graph-data-filter.html",
        "tfma-graph-data-filter.js",
    ],
    path = "/tfma-graph-data-filter",
    deps = [
        "//tensorflow_model_analysis/frontend/tfma-metrics-histogram:lib",
        "//tensorflow_model_analysis/frontend/tfma-slice-overview:lib",
        "@org_polymer",
        "@org_polymer_iron_pages",
        "@org_polymer_paper_dropdown_menu",
        "@org_polymer_paper_input",
        "@org_polymer_paper_item",
        "@org_polymer_paper_listbox",
        "@org_polymer_paper_slider",
    ],
)
