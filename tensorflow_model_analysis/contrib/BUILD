licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "model_eval_lib",
    srcs = ["model_eval_lib.py"],
    deps = [
        "//tensorflow_model_analysis:constants",
        "//tensorflow_model_analysis:expect_apache_beam_installed",  # b/73825929
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis/api/impl:api_types",
        "//tensorflow_model_analysis/api/impl:evaluate",
    ],
)

py_test(
    name = "model_eval_lib_test",
    srcs = ["model_eval_lib_test.py"],
    deps = [
        ":model_eval_lib",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis/eval_saved_model:testutil",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:csv_linear_classifier",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:custom_estimator",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:fixed_prediction_estimator",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:linear_classifier",
        "//tensorflow_model_analysis/eval_saved_model/post_export_metrics",
        "//tensorflow_model_analysis/slicer",
    ],
)
