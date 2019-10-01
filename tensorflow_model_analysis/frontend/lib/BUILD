licenses(["notice"])  # Apache 2.0

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_js_binary", "closure_js_library", "closure_js_test")

package(
    default_visibility = ["//tensorflow_model_analysis:__subpackages__"],
)

closure_js_library(
    name = "buckets-wrapper",
    srcs = [
        "buckets-wrapper.js",
    ],
    suppress = [
        "JSC_UNKNOWN_EXPR_TYPE",
        "JSC_UNRECOGNIZED_TYPE_ERROR",
        "JSC_UNUSED_LOCAL_ASSIGNMENT",
    ],
    deps = [
        ":constants",
        ":least-square-fit",
    ],
)

closure_js_library(
    name = "cell-renderer",
    srcs = [
        "cell-renderer.js",
    ],
    suppress = [
        "JSC_IMPLICITLY_NULLABLE_JSDOC",
        "JSC_NULLABLE_RETURN_WITH_NAME",
        "JSC_UNKNOWN_EXPR_TYPE",
        "JSC_UNRECOGNIZED_TYPE_ERROR",
        "JSC_UNUSED_LOCAL_ASSIGNMENT",
    ],
    deps = [
        ":constants",
        ":table-provider",
        "@io_bazel_rules_closure//closure/library",
    ],
)

closure_js_library(
    name = "constants",
    srcs = [
        "constants.js",
    ],
    suppress = [
        "JSC_UNKNOWN_EXPR_TYPE",
        "JSC_UNRECOGNIZED_TYPE_ERROR",
        "JSC_UNUSED_LOCAL_ASSIGNMENT",
    ],
)

closure_js_library(
    name = "data",
    srcs = [
        "data.js",
    ],
    suppress = [
        "JSC_MISSING_CONST_PROPERTY",
        "JSC_UNKNOWN_EXPR_TYPE",
        "JSC_UNRECOGNIZED_TYPE_ERROR",
        "JSC_UNUSED_LOCAL_ASSIGNMENT",
    ],
    deps = [
        ":cell-renderer",
        ":constants",
        ":table-provider",
    ],
)

closure_js_library(
    name = "graph-data",
    srcs = [
        "graph-data.js",
    ],
    suppress = [
        "JSC_IMPLICITLY_NULLABLE_JSDOC",
        "JSC_UNKNOWN_EXPR_TYPE",
        "JSC_UNRECOGNIZED_TYPE_ERROR",
        "JSC_UNUSED_LOCAL_ASSIGNMENT",
    ],
    deps = [":table-provider"],
)

closure_js_library(
    name = "least-square-fit",
    srcs = [
        "least-square-fit.js",
    ],
    suppress = [
        "JSC_UNKNOWN_EXPR_TYPE",
        "JSC_UNRECOGNIZED_TYPE_ERROR",
        "JSC_UNUSED_LOCAL_ASSIGNMENT",
    ],
)

closure_js_library(
    name = "line-chart-provider",
    srcs = [
        "line-chart-provider.js",
    ],
    suppress = [
        "JSC_UNKNOWN_EXPR_TYPE",
        "JSC_UNRECOGNIZED_TYPE_ERROR",
        "JSC_UNUSED_LOCAL_ASSIGNMENT",
    ],
)

closure_js_library(
    name = "series-data",
    srcs = [
        "series-data.js",
    ],
    suppress = [
        "JSC_MISSING_CONST_PROPERTY",
        "JSC_UNKNOWN_EXPR_TYPE",
        "JSC_UNRECOGNIZED_TYPE_ERROR",
        "JSC_UNUSED_LOCAL_ASSIGNMENT",
    ],
    deps = [
        ":line-chart-provider",
        ":series-data-helper",
        ":table-provider",
    ],
)

closure_js_library(
    name = "series-data-helper",
    srcs = [
        "series-data-helper.js",
    ],
    suppress = [
        "JSC_UNKNOWN_EXPR_TYPE",
        "JSC_UNRECOGNIZED_TYPE_ERROR",
        "JSC_UNUSED_LOCAL_ASSIGNMENT",
    ],
)

closure_js_library(
    name = "single-series-graph-data",
    srcs = [
        "single-series-graph-data.js",
    ],
    suppress = [
        "JSC_MISSING_CONST_PROPERTY",
        "JSC_UNKNOWN_EXPR_TYPE",
        "JSC_UNRECOGNIZED_TYPE_ERROR",
        "JSC_UNUSED_LOCAL_ASSIGNMENT",
    ],
    deps = [
        ":constants",
        ":data",
        ":graph-data",
    ],
)

closure_js_library(
    name = "table-provider",
    srcs = [
        "table-provider.js",
    ],
    suppress = [
        "JSC_UNKNOWN_EXPR_TYPE",
        "JSC_UNRECOGNIZED_TYPE_ERROR",
        "JSC_UNUSED_LOCAL_ASSIGNMENT",
    ],
)

closure_js_library(
    name = "test-util",
    srcs = [
        "test-util.js",
    ],
    suppress = [
        "JSC_UNKNOWN_EXPR_TYPE",
        "JSC_UNRECOGNIZED_TYPE_ERROR",
        "JSC_UNUSED_LOCAL_ASSIGNMENT",
    ],
)

JS_LIB_EXTERNS = [
    "externs.js",
]

ALL_EXTERNS = [
    "svg-externs.js",
    "d3-externs.js",
    "d3-tip-externs.js",
] + JS_LIB_EXTERNS

filegroup(
    name = "externs-list",
    srcs = JS_LIB_EXTERNS,
)

filegroup(
    name = "all-externs-list",
    srcs = ALL_EXTERNS,
)

# The externs needed to build the plugin dashboard and component demos.
closure_js_library(
    name = "externs",
    srcs = ALL_EXTERNS,
)

# The extenrs needed to build the js support library.
closure_js_library(
    name = "js-lib-externs",
    srcs = JS_LIB_EXTERNS,
)

closure_js_binary(
    name = "js-lib-bin",
    defs = [
        # Wrap the compiled js in an anonymous function to avoid name collision.
        "--output_wrapper=\"(function() {%output%})();\"",
    ],
    entry_points = [
        "tfma.Data",
        "tfma.BucketsWrapper",
        "tfma.SeriesData",
        "tfma.SingleSeriesGraphData",
    ],
    deps = [
        ":buckets-wrapper",
        ":data",
        ":graph-data",
        ":js-lib-externs",
        ":series-data",
        ":single-series-graph-data",
    ],
)

closure_js_test(
    name = "buckets-wrapper-test",
    srcs = [
        "buckets-wrapper_test.js",
    ],
    compilation_level = "WHITESPACE_ONLY",
    entry_points = ["tfma.tests.BucketsWrapperTest"],
    deps = [
        ":buckets-wrapper",
        ":constants",
        "@io_bazel_rules_closure//closure/library",
        "@io_bazel_rules_closure//closure/library:testing",
    ],
)

closure_js_test(
    name = "cell-renderer-test",
    srcs = [
        "cell-renderer_test.js",
    ],
    compilation_level = "WHITESPACE_ONLY",
    entry_points = ["tfma.tests.CellRendererTest"],
    deps = [
        ":cell-renderer",
        ":constants",
        "@io_bazel_rules_closure//closure/library",
        "@io_bazel_rules_closure//closure/library:testing",
    ],
)

closure_js_test(
    name = "data-test",
    srcs = [
        "data_test.js",
    ],
    compilation_level = "WHITESPACE_ONLY",
    entry_points = ["tfma.tests.DataTest"],
    deps = [
        ":constants",
        ":data",
        "@io_bazel_rules_closure//closure/library",
        "@io_bazel_rules_closure//closure/library:testing",
    ],
)

closure_js_test(
    name = "least-square-fit-test",
    srcs = [
        "least-square-fit_test.js",
    ],
    compilation_level = "WHITESPACE_ONLY",
    entry_points = ["tfma.tests.LeastSquareFitTest"],
    deps = [
        ":constants",
        ":least-square-fit",
        "@io_bazel_rules_closure//closure/library",
        "@io_bazel_rules_closure//closure/library:testing",
    ],
)

closure_js_test(
    name = "series-data-test",
    srcs = [
        "series-data_test.js",
    ],
    compilation_level = "WHITESPACE_ONLY",
    entry_points = ["tfma.tests.SeriesDataTest"],
    deps = [
        ":constants",
        ":data",
        ":series-data",
        "@io_bazel_rules_closure//closure/library",
        "@io_bazel_rules_closure//closure/library:testing",
    ],
)

closure_js_test(
    name = "single-series-graph-data-test",
    srcs = [
        "single-series-graph-data_test.js",
    ],
    compilation_level = "WHITESPACE_ONLY",
    entry_points = ["tfma.tests.SingleSeriesGraphDataTest"],
    deps = [
        ":constants",
        ":single-series-graph-data",
        "@io_bazel_rules_closure//closure/library",
        "@io_bazel_rules_closure//closure/library:testing",
    ],
)
