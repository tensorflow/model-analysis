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
        "//tensorflow_model_analysis/api/impl:serialization",
        "//tensorflow_model_analysis/eval_saved_model/post_export_metrics",
        "//tensorflow_model_analysis/slicer",
    ],
)

py_test(
    name = "model_eval_lib_test",
    srcs = ["model_eval_lib_test.py"],
    shard_count = 3,
    deps = [
        ":model_eval_lib",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis/eval_saved_model:testutil",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:csv_linear_classifier",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:custom_estimator",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:fixed_prediction_estimator",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:linear_classifier",
        "//tensorflow_model_analysis/eval_saved_model/post_export_metrics",
        "//tensorflow_model_analysis/proto:metrics_for_slice_py_pb2",
        "//tensorflow_model_analysis/slicer",
    ],
)

py_library(
    name = "tfma_unit_non_testonly",
    srcs = ["tfma_unit.py"],
    deps = [
        "//tensorflow_model_analysis:expect_apache_beam_installed",
        "//tensorflow_model_analysis/api/impl:evaluate",
        "//tensorflow_model_analysis/eval_saved_model:load",
        "//tensorflow_model_analysis/eval_saved_model:testutil_non_testonly",
        "//tensorflow_model_analysis/slicer",
    ],
)

py_library(
    name = "tfma_unit",
    testonly = 1,
    deps = ["tfma_unit_non_testonly"],
)

py_test(
    name = "tfma_unit_test",
    srcs = ["tfma_unit_test.py"],
    deps = [
        ":tfma_unit",
        "//tensorflow_model_analysis:expect_apache_beam_installed",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:fixed_prediction_estimator",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:fixed_prediction_estimator_extra_fields",
        "//tensorflow_model_analysis/eval_saved_model/post_export_metrics",
    ],
)
