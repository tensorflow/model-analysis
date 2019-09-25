licenses(["notice"])

package(default_visibility = [
    "//cloud/ml/explainability:__subpackages__",
    "//tensorflow_model_analysis:__subpackages__",
])

py_binary(
    name = "build_docs",
    srcs = ["build_docs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//tensorflow_model_analysis",
        "//third_party/py/absl:app",
        "//third_party/py/tensorflow_docs/api_generator:generate_lib",
    ],
)
