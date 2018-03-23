# TensorFlow Model Analysis

**TensorFlow Model Analysis (TFMA)** is a library for evaluating TensorFlow
models. It allows users to evaluate their models on large amounts of data in a
distributed fashion, using the same metrics defined in their trainer. These
metrics can also be computed over different slices of data, and the results can
be visualised in Jupyter Notebooks.

**TFMA may introduce backwards incompatible changes before version 1.0**.

## Installation and Dependencies

The easiest and recommended way to install TFMA is with the PyPI package.

`pip install tensorflow-model-analysis`

Currently TFMA requires that TensorFlow be installed but does not have an
explicit dependency on TensorFlow as a package. See [TensorFlow
documentation](https://www.tensorflow.org/install/) for more information on
installing TensorFlow.

To enable TFMA visualization in Jupyter Notebook, run<sup>1</sup>:

```
jupyter nbextension enable --py widgetsnbextension
jupyter nbextension install --py --symlink tensorflow_model_analysis
jupyter nbextension enable --py tensorflow_model_analysis
```

TFMA requires [Apache Beam](https://beam.apache.org/) to run distributed
pipelines. Apache Beam runs in local mode by default, and can also run in
distributed mode using
[Google Cloud Dataflow](https://cloud.google.com/dataflow/). TFMA is designed to
be extensible to other Apache Beam runners.

### Compatible Versions

This is a table of versions known to be compatible with each other, based on
our testing framework. Other combinations may also work, but are untested.

|tensorflow-model-analysis  |tensorflow    |apache-beam[gcp]|
|---------------------------|--------------|----------------|
|GitHub master              |1.6           |2.4.0           |

## Getting Started

For instructions on using TFMA, see the [getting started
guide](docs/getting_started.md).

<sup>1</sup> If Jupyter is installed in your home directory, add `--user` for
    all commands; if Jupyter is installed in root or virtualenv is used,
    `--sys-prefix` might be needed.
