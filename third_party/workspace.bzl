"""TensorBoard external dependencies that can be loaded in WORKSPACE files."""

load("//third_party:polymer.bzl", "tensorflow_model_analysis_polymer_workspace")

def tensorflow_model_analysis_workspace():
    """Download TensorFlow Model Analysis build dependencies."""
    tensorflow_model_analysis_polymer_workspace()
