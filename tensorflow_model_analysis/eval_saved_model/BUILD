licenses(["notice"])  # Apache 2.0

py_library(
    name = "export",
    srcs = ["export.py"],
    deps = [
        ":encoding",
        ":util",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis:types",
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
        ":encoding",
        ":graph_ref",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis:types",
        "//third_party/py/numpy",
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
    shard_count = 4,
    deps = [
        ":encoding",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis/eval_saved_model:load",
        "//tensorflow_model_analysis/eval_saved_model:testutil",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:csv_linear_classifier",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:custom_estimator",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:linear_classifier",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:linear_classifier_multivalent",
        "//tensorflow_model_analysis/eval_saved_model/example_trainers:multi_head",
        "//third_party/tensorflow/core:protos_all_py_pb2",
    ],
)

py_library(
    name = "testutil",
    testonly = 1,
    srcs = ["testutil.py"],
    deps = [
        "//tensorflow_model_analysis:expect_tensorflow_installed",
    ],
)

py_test(
    name = "testutil_test",
    srcs = ["testutil_test.py"],
    deps = [
        ":testutil",
        "//third_party/tensorflow/core:protos_all_py_pb2",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis:types",
    ],
)
