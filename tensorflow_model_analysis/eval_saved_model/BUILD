licenses(["notice"])  # Apache 2.0

py_library(
    name = "constants",
    srcs = ["constants.py"],
    deps = [],
)

py_library(
    name = "export",
    srcs = ["export.py"],
    deps = [
        ":constants",
        ":encoding",
        ":util",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis:types",
        "//tensorflow_model_analysis:version",
    ],
)

py_library(
    name = "exporter",
    srcs = ["exporter.py"],
    deps = [
        ":export",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
    ],
)

py_library(
    name = "encoding",
    srcs = ["encoding.py"],
    deps = [
        "//google/protobuf:any_py_pb2",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis:types",
        "//third_party/tensorflow/core:protos_all_py_pb2",
    ],
)

py_test(
    name = "encoding_test",
    srcs = ["encoding_test.py"],
    deps = [
        ":encoding",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
    ],
)

py_library(
    name = "load",
    srcs = ["load.py"],
    deps = [
        ":constants",
        ":encoding",
        ":graph_ref",
        ":util",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis:types",
        "//tensorflow_model_analysis:util",
    ],
)

py_library(
    name = "graph_ref",
    srcs = ["graph_ref.py"],
    deps = [
        ":encoding",
        "//google/protobuf:any_py_pb2",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis:types",
        "//third_party/tensorflow/core:protos_all_py_pb2",
    ],
)

py_test(
    name = "graph_ref_test",
    srcs = ["graph_ref_test.py"],
    deps = [
        ":graph_ref",
        "//third_party/tensorflow/core:protos_all_py_pb2",
    ],
)

py_test(
    name = "integration_test",
    srcs = ["integration_test.py"],
    shard_count = 6,
    deps = [
        ":encoding",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis/eval_saved_model:load",
        "//tensorflow_model_analysis/eval_saved_model:testutil",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:csv_linear_classifier",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:custom_estimator",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:dnn_classifier",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:fake_sequence_to_prediction",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:fixed_prediction_classifier",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:linear_classifier",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:linear_classifier_multivalent",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:multi_head",
        "//tensorflow_model_analysis/eval_saved_model/post_export_metrics:metrics",
        "//third_party/tensorflow/core:protos_all_py_pb2",
    ],
)

py_library(
    name = "testutil",
    testonly = 1,
    deps = [":testutil_non_testonly"],
)

py_library(
    name = "testutil_non_testonly",
    srcs = ["testutil.py"],
    deps = [
        ":util",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis:types",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    deps = [
        ":testutil",
        ":util",
        "//third_party/tensorflow/core:protos_all_py_pb2",
    ],
)
