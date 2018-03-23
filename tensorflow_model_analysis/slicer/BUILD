licenses(["notice"])  # Apache 2.0

py_library(
    name = "slicer",
    srcs = ["slicer.py"],
    visibility = ["//tensorflow_model_analysis:__subpackages__"],
    deps = [
        ":slice_accessor",
        "//tensorflow_model_analysis:expect_six_installed",
        "//tensorflow_model_analysis:types",
        "//third_party/py/typing",
    ],
)

py_test(
    name = "slicer_test",
    srcs = ["slicer_test.py"],
    deps = [
        ":slicer",
        "//tensorflow_model_analysis:expect_six_installed",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "slice_accessor",
    srcs = ["slice_accessor.py"],
    deps = [
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis:types",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "slice_accessor_test",
    srcs = ["slice_accessor_test.py"],
    deps = [
        ":slice_accessor",
        "//tensorflow_model_analysis:expect_tensorflow_installed",
        "//tensorflow_model_analysis/eval_saved_model:encoding",
        "//third_party/py/numpy",
    ],
)
