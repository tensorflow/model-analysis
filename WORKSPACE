workspace(name = "org_tensorflow_model_analysis")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# TF 1.15.2
# LINT.IfChange(tf_commit)
_TENSORFLOW_GIT_COMMIT = "5d80e1e8e6ee999be7db39461e0e79c90403a2e4"
# LINT.ThenChange(:io_bazel_rules_clousure)
http_archive(
    name = "org_tensorflow",
    sha256 = "7e3c893995c221276e17ddbd3a1ff177593d00fc57805da56dcc30fdc4299632",
    urls = [
      "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
      "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
    ],
    strip_prefix = "tensorflow-%s" % _TENSORFLOW_GIT_COMMIT,
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
    sha256 = "7d206c2383811f378a5ef03f4aacbcf5f47fd8650f6abbc3fa89f3a27dd8b176",
    strip_prefix = "rules_closure-0.10.0",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/0.10.0.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/0.10.0.tar.gz",
    ],
)

load("@io_bazel_rules_closure//closure:repositories.bzl", "rules_closure_dependencies", "rules_closure_toolchains")
rules_closure_dependencies()
rules_closure_toolchains()

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

# Specify the minimum required bazel version.
load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")

check_bazel_version_at_least("3.7.2")
