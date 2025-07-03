workspace(name = "org_tensorflow_model_analysis")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# TF 2.17.1
# LINT.IfChange(tf_commit)
_TENSORFLOW_GIT_COMMIT = "3c92ac03cab816044f7b18a86eb86aa01a294d95"

# LINT.ThenChange(:io_bazel_rules_closure)
http_archive(
    name = "org_tensorflow",
    sha256 = "317dd95c4830a408b14f3e802698eb68d70d81c7c7cfcd3d28b0ba023fe84a68",
    strip_prefix = "tensorflow-%s" % _TENSORFLOW_GIT_COMMIT,
    urls = [
        "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
        "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
    ],
)

# Needed by tensorboard. Because these http_archives do not handle transitive
# dependencies, we need to unroll them here.
http_archive(
    name = "rules_rust",
    sha256 = "08109dccfa5bbf674ff4dba82b15d40d85b07436b02e62ab27e0b894f45bb4a3",
    strip_prefix = "rules_rust-d5ab4143245af8b33d1947813d411a6cae838409",
    urls = [
        # Master branch as of 2022-01-31
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_rust/archive/d5ab4143245af8b33d1947813d411a6cae838409.tar.gz",
        "https://github.com/bazelbuild/rules_rust/archive/d5ab4143245af8b33d1947813d411a6cae838409.tar.gz",
    ],
)

load("@rules_rust//rust:repositories.bzl", "rust_repositories")

rust_repositories()

http_archive(
    name = "io_bazel_rules_webtesting",
    sha256 = "9bb461d5ef08e850025480bab185fd269242d4e533bca75bfb748001ceb343c3",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_webtesting/releases/download/0.3.3/rules_webtesting.tar.gz",
        "https://github.com/bazelbuild/rules_webtesting/releases/download/0.3.3/rules_webtesting.tar.gz",
    ],
)

load("@io_bazel_rules_webtesting//web:repositories.bzl", "web_test_repositories")

web_test_repositories()

# LINT.IfChange(io_bazel_rules_closure)
_RULES_CLOSURE_GIT_COMMIT = "db4683a2a1836ac8e265804ca5fa31852395185b"

# LINT.ThenChange(:tf_commit)
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "6a900831c1eb8dbfc9d6879b5820fd614d4ea1db180eb5ff8aedcb75ee747c1f",
    strip_prefix = "rules_closure-%s" % _RULES_CLOSURE_GIT_COMMIT,
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/%s.tar.gz" % _RULES_CLOSURE_GIT_COMMIT,
        "https://github.com/bazelbuild/rules_closure/archive/%s.tar.gz" % _RULES_CLOSURE_GIT_COMMIT,  # 2020-01-15
    ],
)

load("@io_bazel_rules_closure//closure:repositories.bzl", "rules_closure_dependencies", "rules_closure_toolchains")

rules_closure_dependencies()

rules_closure_toolchains()

http_archive(
    name = "org_tensorflow_tensorboard",
    sha256 = "60f98f6321f3851725ca73bf94ac994d88ff6d1f8a8332a16f49cefffeabcca3",
    strip_prefix = "tensorboard-2.8.0",
    urls = ["https://github.com/tensorflow/tensorboard/archive/refs/tags/2.8.0.zip"],
)

load("@org_tensorflow_tensorboard//third_party:workspace.bzl", "tensorboard_workspace")

_PROTOBUF_COMMIT = "4.25.6"  # 4.25.6

http_archive(
    name = "com_google_protobuf",
    sha256 = "ff6e9c3db65f985461d200c96c771328b6186ee0b10bc7cb2bbc87cf02ebd864",
    strip_prefix = "protobuf-%s" % _PROTOBUF_COMMIT,
    urls = [
        "https://github.com/protocolbuffers/protobuf/archive/v4.25.6.zip",
    ],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

tensorboard_workspace()

load("//third_party:workspace.bzl", "tensorflow_model_analysis_workspace")

# Please add all new dependencies in workspace.bzl.
tensorflow_model_analysis_workspace()

load("@bazel_skylib//lib:versions.bzl", "versions")

versions.check("6.5.0")
