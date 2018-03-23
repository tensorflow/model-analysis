licenses(["notice"])  # Apache 2.0

load("@io_bazel_rules_closure//closure:defs.bzl", "web_library")

package(
    default_visibility = ["//tensorflow_model_analysis:__subpackages__"],
)

web_library(
    name = "lib",
    srcs = [
        "tfma-line-chart-grid.html",
        "tfma-line-chart-grid.js",
    ],
    path = "/tfma-line-chart-grid",
    deps = [
        "//tensorflow_model_analysis/frontend/tfma-line-chart:lib",
        "@org_polymer",
        "@org_polymer_iron_icons",
        "@org_polymer_paper_card",
        "@org_polymer_paper_dropdown_menu",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_item",
        "@org_polymer_paper_listbox",
    ],
)
