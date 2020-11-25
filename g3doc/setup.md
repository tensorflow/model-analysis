# Tensorflow Model Analysis Setup

## Configuration

TFMA stores its configuration in a
[proto](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/proto/config.proto)
that is serialized to JSON. This proto consolidates the configuration required
for input data, output data, model specifications, metric specifications, and
slicing specifications.

All TFMA pipelines are associated with a baseline (primary) model and zero or
more candidate (secondary) models. The baseline and candidate model are defined
by the user at the start of the pipeline and each require a unique name. The
following are examples of typical configuration setups a user may use:

*   Single model evaluation:
    *   N/A (i.e. no name)
*   Validation-based evaluation:
    *   `baseline`
    *   `candidate`
*   Model comparison evaluation:
    *   `my_model_a`
    *   `my_model_b`

### Model Specs

Model specs are of type `tfma.ModelSpec` and are used to define the location of
a model as well as other model specific parameters. For example the following
are typical settings that would need to be configured prior to running an
evaluation:

*   `name` - name of model (if multiple models used)
*   `signature_name` - name of signature used for predictions (default is
    `serving_default`). Use `eval` if using an EvalSavedModel.
*   `label_key` - name of the feature associated with the label.
*   `example_weight_key` - name of the feature assocated with the example
    weight.

### Metrics Specs

Metrics specs are of type `tfma.MetricsSpec` and are used to configure the
metrics that will be calculated as part of the evaluation. Different machine
learning problems use different types of metrics and TFMA offers a lot of
options for configuring and customizing the metrics that are computed. Since
metrics are a very large part of TFMA, they are discussed in detail separately
in [metrics](metrics.md).

### Slicing Specs

Slicing specs are of type `tfma.SlicingSpec` and are used to configure the
slices criteria that will be used during the evaluation. Slicing can be done
either by `feature_keys`, `feature_values`, or both. Some examples of slicing
specs are as follows:

*   `{}`
    *   Slice consisting of overall data.
*   `{ feature_keys: ["country"] }`
    *   Slices for all values in feature "country". For example, we might get
        slices "country:us", "country:jp", etc.
*   `{ feature_values: [{key: "country", value: "us"}] }`
    *   Slice consisting of "country:us".
*   `{ feature_keys: ["country", "city"] }`
    *   Slices for all values in feature "country" crossed with all values in
        feature "city" (note this may be expensive).
*   `{ feature_keys: ["country"] feature_values: [{key: "age", value: "20"}] }`
    *   Slices for all values in feature "country" crossed with value "age:20"

Note that feature keys may be either transformed features or raw input features.
See `tfma.SlicingSpec` for more information.

## EvalSharedModel

In addition to the configuration settings, TFMA also requires that an instance
of a `tfma.EvalSharedModel` be created for sharing a model between multiple
threads in the same process. The shared model instance includes information
about the type of model (keras, etc) and how to load and configure the model
from its saved location on disk (e.g. tags, etc). The
`tfma.default_eval_shared_model` API can be used to create a default instance
given a path and set of tags.
