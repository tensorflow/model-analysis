licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@protobuf_bzl//:protobuf.bzl", "py_proto_library")

py_proto_library(
    name = "config_py_pb2",
    srcs = [
        "config.proto",
    ],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    deps = [
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_proto_library(
    name = "metrics_for_slice_py_pb2",
    srcs = [
        "metrics_for_slice.proto",
    ],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    deps = [
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_proto_library(
    name = "validation_result",
    srcs = [
        "validation_result.proto",
    ],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    deps = [
        ":config_py_pb2",
        ":metrics_for_slice_py_pb2",
        "@com_google_protobuf//:protobuf_python",
    ],
)
