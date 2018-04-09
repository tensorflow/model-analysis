licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "evaluate",
    srcs = ["evaluate.py"],
    deps = [
        "//tensorflow_model_analysis:expect_apache_beam_installed",  # b/73825929
        "//tensorflow_model_analysis:expect_six_installed",
        "//tensorflow_model_analysis:expect_tensorflow_transform_installed",
        "//tensorflow_model_analysis:types",
        "//tensorflow_model_analysis/eval_saved_model:load",
        "//tensorflow_model_analysis/eval_saved_model:util",
        "//tensorflow_model_analysis/eval_saved_model/post_export_metrics:metric_keys",
        "//tensorflow_model_analysis/slicer",
    ],
)

py_test(
    name = "evaluate_test",
    srcs = ["evaluate_test.py"],
    deps = [
        ":evaluate",
        "//tensorflow_model_analysis:expect_apache_beam_installed",  # b/73825929
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis/eval_saved_model:testutil",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:fixed_prediction_estimator",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:linear_classifier",
        "//tensorflow_model_analysis/eval_saved_model/post_export_metrics",
        "//tensorflow_model_analysis/eval_saved_model/post_export_metrics:metric_keys",
        "//third_party/tensorflow/core:protos_all_py_pb2",
    ],
)

py_library(
    name = "serialization",
    srcs = ["serialization.py"],
    deps = [
        ":api_types",
        "//tensorflow_model_analysis:expect_apache_beam_installed",  # b/73825929
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis:version",
        "//tensorflow_model_analysis/slicer",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "serialization_test",
    srcs = ["serialization_test.py"],
    deps = [
        ":serialization",
        "//tensorflow_model_analysis/eval_saved_model:testutil",
    ],
)

py_library(
    name = "api_types",
    srcs = ["api_types.py"],
    deps = [
        "//tensorflow_model_analysis:constants",
        "//tensorflow_model_analysis/slicer",
    ],
)
