workspace(name = "org_tensorflow_model_analysis")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "6691c58a2cd30a86776dd9bb34898b041e37136f2dc7e24cadaeaf599c95c657",
    strip_prefix = "rules_closure-08039ba8ca59f64248bb3b6ae016460fe9c9914f",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/08039ba8ca59f64248bb3b6ae016460fe9c9914f.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/08039ba8ca59f64248bb3b6ae016460fe9c9914f.tar.gz",  # 2018-01-16
    ],
)

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")
closure_repositories()

http_archive(
    name = "org_tensorflow_tensorboard",
    sha256 = "a943c0242a07da4d445135ffc9a7c7cb987d9bd948ae733695bc16095dceec20",
    strip_prefix = "tensorboard-2fdb2199553729a6c5b42b7eb0305a101b454add",
    urls = ["https://github.com/tensorflow/tensorboard/archive/2fdb2199553729a6c5b42b7eb0305a101b454add.zip"],
)

load("@org_tensorflow_tensorboard//third_party:workspace.bzl", "tensorboard_workspace")
tensorboard_workspace()

load("//third_party:workspace.bzl", "tensorflow_model_analysis_workspace")
# Please add all new dependencies in workspace.bzl.
tensorflow_model_analysis_workspace()
