# TensorFlow Model Analysis

*TensorFlow Model Analysis* (TFMA) is a library for evaluating TensorFlow
models. It allows users to evaluate their models on large amounts of data in a
distributed manner, using the same metrics defined in their trainer. These
metrics can be computed over different slices of data and visualized in Jupyter
notebooks.

![TFMA Slicing Metrics Browser](./images/tfma-slicing-metrics-browser.gif)

Caution: TFMA may introduce backwards incompatible changes before version 1.0.

## Installation

The recommended way to install TFMA is using the
[PyPI package](https://pypi.org/project/tensorflow-model-analysis/):

<pre class="devsite-terminal devsite-click-to-copy">
pip install tensorflow-model-analysis
</pre>

### Build TFMA from source

To build from source follow the following steps:

Install the protoc as per the link mentioned:
[protoc](https://grpc.io/docs/protoc-installation/#install-pre-compiled-binaries-any-os)

Create a virtual environment by running the commands

```
python3 -m venv <virtualenv_name>
source <virtualenv_name>/bin/activate
pip3 install setuptools wheel
git clone https://github.com/tensorflow/model-analysis.git
cd model-analysis
python3 setup.py bdist_wheel
```

This will build the TFMA wheel in the dist directory. To install the wheel from
dist directory run the commands

```
cd dist
pip3 install tensorflow_model_analysis-<version>-py3-none-any.whl
```

### Nightly Packages

TFMA also hosts nightly packages at https://pypi-nightly.tensorflow.org on
Google Cloud. To install the latest nightly package, please use the following
command:

<pre class="devsite-terminal devsite-click-to-copy">
pip install -i https://pypi-nightly.tensorflow.org/simple tensorflow-model-analysis
</pre>

This will install the nightly packages for the major dependencies of TFMA such
as TensorFlow Metadata (TFMD), TFX Basic Shared Libraries (TFX-BSL).

Note: These nightly packages are unstable and breakages are likely to happen.
The fix could often take a week or more depending on the complexity involved.

Currently, TFMA requires that TensorFlow is installed but does not have an
explicit dependency on the TensorFlow PyPI package. See the
[TensorFlow install guides](https://www.tensorflow.org/install) for
instructions.

To enable TFMA visualization in Jupyter Notebook:

<pre class="prettyprint">
  <code class="devsite-terminal">jupyter nbextension enable --py widgetsnbextension</code>
  <code class="devsite-terminal">jupyter nbextension enable --py tensorflow_model_analysis</code>
</pre>

Note: If Jupyter notebook is already installed in your home directory, add
`--user` to these commands. If Jupyter is installed as root, or using a virtual
environment, the parameter `--sys-prefix` might be required.

### Jupyter Lab

As of writing, because of https://github.com/pypa/pip/issues/9187, `pip install`
might never finish. In that case, you should revert pip to version 19 instead of
20: `pip install "pip<20"`.

Using a JupyterLab extension requires installing dependencies on the command
line. You can do this within the console in the JupyterLab UI or on the command
line. This includes separately installing any pip package dependencies and
JupyterLab labextension plugin dependencies, and the version numbers must be
compatible.

The examples below use 0.27.0. Check available [versions](#compatible-versions)
below to use the latest.

#### Jupyter Lab 1.2.x

```posix-terminal
pip install tensorflow_model_analysis==0.27.0

jupyter labextension install tensorflow_model_analysis@0.27.0

jupyter labextension install @jupyter-widgets/jupyterlab-manager@1.1
```

#### Jupyter Lab 2

```posix-terminal
pip install tensorflow_model_analysis==0.27.0

jupyter labextension install tensorflow_model_analysis@0.27.0

jupyter labextension install @jupyter-widgets/jupyterlab-manager@2
```

#### Troubleshooting

Check pip packages:

```posix-terminal
pip list
```

Check extensions:

```posix-terminal
jupyter labextension list
```

### Notable Dependencies

TensorFlow is required.

[Apache Beam](https://beam.apache.org/) is required; it's the way that efficient
distributed computation is supported. By default, Apache Beam runs in local mode
but can also run in distributed mode using
[Google Cloud Dataflow](https://cloud.google.com/dataflow/) and other Apache
Beam
[runners](https://beam.apache.org/documentation/runners/capability-matrix/).

[Apache Arrow](https://arrow.apache.org/) is also required. TFMA uses Arrow to
represent data internally in order to make use of vectorized numpy functions.

## Getting Started

For instructions on using TFMA, see the [get started guide](./get_started).

## Compatible Versions

The following table is the TFMA package versions that are compatible with each
other. This is determined by our testing framework, but other *untested*
combinations may also work.

tensorflow-model-analysis                                                            | apache-beam[gcp] | pyarrow | tensorflow        | tensorflow-metadata | tfx-bsl
------------------------------------------------------------------------------------ | ---------------- | ------- | ----------------- | ------------------- | -------
[GitHub master](https://github.com/tensorflow/model-analysis/blob/master/RELEASE.md) | 2.28.0           | 2.0.0   | nightly (1.x/2.x) | 0.30.0              | 0.30.0
[0.30.0](https://github.com/tensorflow/model-analysis/blob/v0.30.0/RELEASE.md)       | 2.28.0           | 2.0.0   | 1.15 / 2.4        | 0.30.0              | 0.30.0
[0.29.0](https://github.com/tensorflow/model-analysis/blob/v0.29.0/RELEASE.md)       | 2.28.0           | 2.0.0   | 1.15 / 2.4        | 0.29.0              | 0.29.0
[0.28.0](https://github.com/tensorflow/model-analysis/blob/v0.28.0/RELEASE.md)       | 2.28.0           | 2.0.0   | 1.15 / 2.4        | 0.28.0              | 0.28.0
[0.27.0](https://github.com/tensorflow/model-analysis/blob/v0.27.0/RELEASE.md)       | 2.27.0           | 2.0.0   | 1.15 / 2.4        | 0.27.0              | 0.27.0
[0.26.0](https://github.com/tensorflow/model-analysis/blob/v0.26.0/RELEASE.md)       | 2.25.0           | 0.17.0  | 1.15 / 2.3        | 0.26.0              | 0.26.0
[0.25.0](https://github.com/tensorflow/model-analysis/blob/v0.25.0/RELEASE.md)       | 2.25.0           | 0.17.0  | 1.15 / 2.3        | 0.25.0              | 0.25.0
[0.24.3](https://github.com/tensorflow/model-analysis/blob/v0.24.3/RELEASE.md)       | 2.24.0           | 0.17.0  | 1.15 / 2.3        | 0.24.0              | 0.24.1
[0.24.2](https://github.com/tensorflow/model-analysis/blob/v0.24.2/RELEASE.md)       | 2.23.0           | 0.17.0  | 1.15 / 2.3        | 0.24.0              | 0.24.0
[0.24.1](https://github.com/tensorflow/model-analysis/blob/v0.24.1/RELEASE.md)       | 2.23.0           | 0.17.0  | 1.15 / 2.3        | 0.24.0              | 0.24.0
[0.24.0](https://github.com/tensorflow/model-analysis/blob/v0.24.0/RELEASE.md)       | 2.23.0           | 0.17.0  | 1.15 / 2.3        | 0.24.0              | 0.24.0
[0.23.0](https://github.com/tensorflow/model-analysis/blob/v0.23.0/RELEASE.md)       | 2.23.0           | 0.17.0  | 1.15 / 2.3        | 0.23.0              | 0.23.0
[0.22.2](https://github.com/tensorflow/model-analysis/blob/v0.22.2/RELEASE.md)       | 2.20.0           | 0.16.0  | 1.15 / 2.2        | 0.22.2              | 0.22.0
[0.22.1](https://github.com/tensorflow/model-analysis/blob/v0.22.1/RELEASE.md)       | 2.20.0           | 0.16.0  | 1.15 / 2.2        | 0.22.2              | 0.22.0
[0.22.0](https://github.com/tensorflow/model-analysis/blob/v0.22.0/RELEASE.md)       | 2.20.0           | 0.16.0  | 1.15 / 2.2        | 0.22.0              | 0.22.0
[0.21.6](https://github.com/tensorflow/model-analysis/blob/v0.21.6/RELEASE.md)       | 2.19.0           | 0.15.0  | 1.15 / 2.1        | 0.21.0              | 0.21.3
[0.21.5](https://github.com/tensorflow/model-analysis/blob/v0.21.5/RELEASE.md)       | 2.19.0           | 0.15.0  | 1.15 / 2.1        | 0.21.0              | 0.21.3
[0.21.4](https://github.com/tensorflow/model-analysis/blob/v0.21.4/RELEASE.md)       | 2.19.0           | 0.15.0  | 1.15 / 2.1        | 0.21.0              | 0.21.3
[0.21.3](https://github.com/tensorflow/model-analysis/blob/v0.21.3/RELEASE.md)       | 2.17.0           | 0.15.0  | 1.15 / 2.1        | 0.21.0              | 0.21.0
[0.21.2](https://github.com/tensorflow/model-analysis/blob/v0.21.2/RELEASE.md)       | 2.17.0           | 0.15.0  | 1.15 / 2.1        | 0.21.0              | 0.21.0
[0.21.1](https://github.com/tensorflow/model-analysis/blob/v0.21.1/RELEASE.md)       | 2.17.0           | 0.15.0  | 1.15 / 2.1        | 0.21.0              | 0.21.0
[0.21.0](https://github.com/tensorflow/model-analysis/blob/v0.21.0/RELEASE.md)       | 2.17.0           | 0.15.0  | 1.15 / 2.1        | 0.21.0              | 0.21.0
[0.15.4](https://github.com/tensorflow/model-analysis/blob/v0.15.4/RELEASE.md)       | 2.16.0           | 0.15.0  | 1.15 / 2.0        | n/a                 | 0.15.1
[0.15.3](https://github.com/tensorflow/model-analysis/blob/v0.15.3/RELEASE.md)       | 2.16.0           | 0.15.0  | 1.15 / 2.0        | n/a                 | 0.15.1
[0.15.2](https://github.com/tensorflow/model-analysis/blob/v0.15.2/RELEASE.md)       | 2.16.0           | 0.15.0  | 1.15 / 2.0        | n/a                 | 0.15.1
[0.15.1](https://github.com/tensorflow/model-analysis/blob/v0.15.1/RELEASE.md)       | 2.16.0           | 0.15.0  | 1.15 / 2.0        | n/a                 | 0.15.0
[0.15.0](https://github.com/tensorflow/model-analysis/blob/v0.15.0/RELEASE.md)       | 2.16.0           | 0.15.0  | 1.15              | n/a                 | n/a
[0.14.0](https://github.com/tensorflow/model-analysis/blob/v0.14.0/RELEASE.md)       | 2.14.0           | n/a     | 1.14              | n/a                 | n/a
[0.13.1](https://github.com/tensorflow/model-analysis/blob/v0.13.1/RELEASE.md)       | 2.11.0           | n/a     | 1.13              | n/a                 | n/a
[0.13.0](https://github.com/tensorflow/model-analysis/blob/v0.13.0/RELEASE.md)       | 2.11.0           | n/a     | 1.13              | n/a                 | n/a
[0.12.1](https://github.com/tensorflow/model-analysis/blob/v0.12.1/RELEASE.md)       | 2.10.0           | n/a     | 1.12              | n/a                 | n/a
[0.12.0](https://github.com/tensorflow/model-analysis/blob/v0.12.0/RELEASE.md)       | 2.10.0           | n/a     | 1.12              | n/a                 | n/a
[0.11.0](https://github.com/tensorflow/model-analysis/blob/v0.11.0/RELEASE.md)       | 2.8.0            | n/a     | 1.11              | n/a                 | n/a
[0.9.2](https://github.com/tensorflow/model-analysis/blob/v0.9.2/RELEASE.md)         | 2.6.0            | n/a     | 1.9               | n/a                 | n/a
[0.9.1](https://github.com/tensorflow/model-analysis/blob/v0.9.1/RELEASE.md)         | 2.6.0            | n/a     | 1.10              | n/a                 | n/a
[0.9.0](https://github.com/tensorflow/model-analysis/blob/v0.9.0/RELEASE.md)         | 2.5.0            | n/a     | 1.9               | n/a                 | n/a
[0.6.0](https://github.com/tensorflow/model-analysis/blob/v0.6.0/RELEASE.md)         | 2.4.0            | n/a     | 1.6               | n/a                 | n/a

## Questions

Please direct any questions about working with TFMA to
[Stack Overflow](https://stackoverflow.com) using the
[tensorflow-model-analysis](https://stackoverflow.com/questions/tagged/tensorflow-model-analysis)
tag.
