# Description:
#   Python SDK for the TensorFlow Model Analysis API.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "constants",
    srcs = ["constants.py"],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    deps = [
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "types_compat",
    srcs = ["types_compat.py"],
    deps = [
        "//tensorflow_model_analysis:expect_apache_beam_installed",  # b/73825929
    ],
)

py_library(
    name = "version",
    srcs = ["version.py"],
    deps = [],
)

py_library(
    name = "expect_tensorflow_installed",
    # This is a dummy rule used as a TensorFlow dependency in open-source.
    # We expect TensorFlow to already be installed on the system, e.g. via
    # `pip install tensorflow`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_tensorflow_transform_installed",
    # This is a dummy rule used as a tf.Transform dependency in open-source.
    # We expect tf.Transform to already be installed on the system, e.g. via
    # `pip install tensorflow-transform`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_apache_beam_installed",
    # This is a dummy rule used as a apache-beam dependency in open-source.
    # We expect Apache Beam to already be installed on the system, e.g. via
    # `pip install apache-beam`
    visibility = ["//visibility:public"],
)

py_library(
    name = "expect_six_installed",
    # This is a dummy rule used as a six dependency in open-source.
    # We expect six to already be installed on the system, e.g. via
    # `pip install six`
    visibility = ["//visibility:public"],
)
