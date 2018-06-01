licenses(["notice"])  # Apache 2.0

py_library(
    name = "post_export_metrics",
    srcs = ["post_export_metrics.py"],
    deps = [
        ":metric_keys",
        ":metrics",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis:types",
        "//tensorflow_model_analysis/eval_saved_model:encoding",
        "//tensorflow_model_analysis/proto:metrics_for_slice_py_pb2",
    ],
)

py_library(
    name = "metric_keys",
    srcs = ["metric_keys.py"],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    deps = [
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis:types",
    ],
)

py_test(
    name = "post_export_metrics_test",
    srcs = ["post_export_metrics_test.py"],
    deps = [
        ":post_export_metrics",
        "//tensorflow_model_analysis:expect_apache_beam_installed",  # b/73825929
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis/api/impl:evaluate",
        "//tensorflow_model_analysis/eval_saved_model:testutil",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:fixed_prediction_estimator",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:fixed_prediction_estimator_extra_fields",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:linear_classifier",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:linear_regressor",
    ],
)
