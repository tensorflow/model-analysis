licenses(["notice"])  # Apache 2.0

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//tensorflow_model_analysis/api/impl:api_types",
        "//tensorflow_model_analysis/eval_saved_model/post_export_metrics:metric_keys",
        "//tensorflow_model_analysis/slicer",
    ],
)

py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    deps = [
        ":util",
        "//tensorflow_model_analysis:constants",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis/api/impl:api_types",
        "//tensorflow_model_analysis/eval_saved_model:testutil",
        "//tensorflow_model_analysis/slicer",
    ],
)
