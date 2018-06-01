licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "feature_extractor",
    srcs = ["feature_extractor.py"],
    deps = [
        "//tensorflow_model_analysis:constants",
        "//tensorflow_model_analysis:expect_apache_beam_installed",  # b/73825929
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis/api/impl:api_types",
        "//tensorflow_model_analysis/eval_saved_model:load",
    ],
)

py_test(
    name = "feature_extractor_test",
    srcs = ["feature_extractor_test.py"],
    deps = [
        ":feature_extractor",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis/eval_saved_model:testutil",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:csv_linear_classifier",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:custom_estimator",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:fixed_prediction_estimator",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:linear_classifier",
    ],
)
