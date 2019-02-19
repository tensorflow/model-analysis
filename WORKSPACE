workspace(name = "org_tensorflow_model_analysis")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "424dce95fddfea8dcf8012a4749fd7166a3009d8d7c32942f19dff12d0bbc2e8",
    strip_prefix = "rules_closure-35ffe0eec59ce21ea6b04a8e3345cfcfcf20f5ed",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/35ffe0eec59ce21ea6b04a8e3345cfcfcf20f5ed.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/35ffe0eec59ce21ea6b04a8e3345cfcfcf20f5ed.tar.gz",  # 2018-04-12
    ],
)

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")

closure_repositories()

http_archive(
    name = "org_tensorflow_tensorboard",
    sha256 = "bd5ce4676158c8e00de43e763e1e6e699b97cade0ce841d5e4b896e3e733dbec",
    strip_prefix = "tensorboard-270d34d1e2b0fc9401db0549004e4a0f0f1ffd2d",
    urls = ["https://github.com/tensorflow/tensorboard/archive/270d34d1e2b0fc9401db0549004e4a0f0f1ffd2d.zip"],  # 2018-04-17
)

load("@org_tensorflow_tensorboard//third_party:workspace.bzl", "tensorboard_workspace")

tensorboard_workspace()

load("//third_party:workspace.bzl", "tensorflow_model_analysis_workspace")

# Please add all new dependencies in workspace.bzl.
tensorflow_model_analysis_workspace()
