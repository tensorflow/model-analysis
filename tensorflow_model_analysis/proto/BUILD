load("@protobuf_bzl//:protobuf.bzl", "py_proto_library")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "proto_files",
    src = glob(["*.proto"]),
)

py_proto_library(
    name = "protos_py_pb2",
    srcs = glob(["*.proto"]),
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
    deps = [
        "__init__.py",
        "@com_google_protobuf//:protobuf_python",
    ],
)
