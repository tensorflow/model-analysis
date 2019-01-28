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
        "tfma-metrics-histogram.html",
        "tfma-metrics-histogram.js",
    ],
    path = "/tfma-metrics-histogram",
    deps = [
        "@org_googlewebcomponents_google_chart",
        "@org_polymer",
        "@org_polymer_iron_icons",
        "@org_polymer_iron_resizable_behavior",
        "@org_polymer_paper_button",
        "@org_polymer_paper_card",
        "@org_polymer_paper_dialog",
        "@org_polymer_paper_dropdown_menu",
        "@org_polymer_paper_icon_button",
        "@org_polymer_paper_input",
        "@org_polymer_paper_item",
        "@org_polymer_paper_listbox",
        "@org_polymer_paper_slider",
        "@org_polymer_paper_toggle_button",
    ],
)
