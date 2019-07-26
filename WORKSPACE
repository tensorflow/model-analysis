workspace(name = "org_tensorflow_model_analysis")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# We have to import zlib directly ourselves, because protobuf_deps.bzl isn't
# part of the protobuf release yet
# (https://github.com/protocolbuffers/protobuf/issues/5918).
http_archive(
    name = "net_zlib",
    build_file = "@com_google_protobuf//:third_party/zlib.BUILD",
    sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
    strip_prefix = "zlib-1.2.11",
    urls = ["https://zlib.net/zlib-1.2.11.tar.gz"],
)

bind(
    name = "zlib",
    actual = "@net_zlib//:zlib",
)

http_archive(
    name = "io_bazel_rules_webtesting",
    sha256 = "5ed12bcfa923c94fb0d0654cf7ca3939491fd1513b1bdbe39eaed566e478e3a3",
    strip_prefix = "rules_webtesting-afa8c4435ed8fd832046dab807ef998a26779ecb",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_webtesting/archive/afa8c4435ed8fd832046dab807ef998a26779ecb.zip",
        "https://github.com/bazelbuild/rules_webtesting/archive/afa8c4435ed8fd832046dab807ef998a26779ecb.zip",  # 0.3.1
    ],
)

load("@io_bazel_rules_webtesting//web:repositories.bzl", "web_test_repositories")

web_test_repositories()

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "e0a111000aeed2051f29fcc7a3f83be3ad8c6c93c186e64beb1ad313f0c7f9f9",
    strip_prefix = "rules_closure-cf1e44edb908e9616030cc83d085989b8e6cd6df",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",  # 2019-04-04
    ],
)

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")

closure_repositories()

http_archive(
    name = "org_tensorflow_tensorboard",
    sha256 = "5a2cdb8cfef775e226aacac9b631b567cb994261c25f370e33854f043d6f7354",
    strip_prefix = "tensorboard-5fc3c8cea4b5f79c738345686a218f089b58ddba",
    urls = ["https://github.com/tensorflow/tensorboard/archive/5fc3c8cea4b5f79c738345686a218f089b58ddba.zip"],  # 1.13
)

load("@org_tensorflow_tensorboard//third_party:workspace.bzl", "tensorboard_workspace")

tensorboard_workspace()

load("//third_party:workspace.bzl", "tensorflow_model_analysis_workspace")

# Please add all new dependencies in workspace.bzl.
tensorflow_model_analysis_workspace()
