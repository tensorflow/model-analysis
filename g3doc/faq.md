# Tensorflow Model Analysis Frequently Asked Questions

[TOC]

## General

### Is an EvalSavedModel still required?

Previously TFMA required all metrics to be stored within a tensorflow graph
using a special `EvalSavedModel`. Now, metrics can be computed outside of the TF
graph using `beam.CombineFn` implementations.

Some of the main differences are:

*   An `EvalSavedModel` requires a special export from the trainer whereas a
    serving model can be used without any changes required to the training code.
*   When an `EvalSavedModel` is used, any metrics added at training time are
    automatically available at evaluation time. Without an `EvalSavedModel`
    these metrics must be re-added.
    *   The exception to this rule is if a keras model is used the metrics can
        also be added automatically because keras saves the metric information
        along side of the saved model.

NOTE: There are some metrics that are only supported using combiners (e.g.
multi-class/multi-label plots, aggregated multi-clas/multi-label metrics, etc).

### Can TFMA work with both in-graph metrics and external metrics?

TFMA allows a hybrid approach to be used where some metrics can be computed
in-graph where as others can be computed outside. If you currently have an
`EvalSavedModel` then you can continue to use it.

There are two cases:

1.  Use TFMA `EvalSavedModel` for both feature extraction and metric
    computations but also add additional combiner-based metrics. In this case
    you would get all the in-graph metrics from the `EvalSavedModel` along with
    any additional metrics from the combiner-based that might not have been
    previously supported.
2.  Use TFMA `EvalSavedModel` for feature/prediction extraction but use
    combiner-based metrics for all metrics computations. This mode is useful if
    there are feature transformations present in the `EvalSavedModel` that you
    would like to use for slicing, but prefer to perform all metric computations
    outside the graph.

## Setup

### What model types are supported?

TFMA supports keras models, models based on generic TF2 signature APIs, as well
TF estimator based models (although depending on the use case the estimator
based models may require an `EvalSavedModel` to be used).

See [get_started](get_started.md) guide for the full list of model types
supported and any restrictions.

### How do I setup TFMA to work with a native keras based model? {#keras}

The following is an example config for a keras model based on the following
assumptions:

*   Saved model is for serving and uses the signature name `serving_default`
    (this can be changed using `model_specs[0].signature_name`).
*   Built in metrics from `model.compile(...)` should be evaluated (this can be
    disabled via `options.include_default_metric` within the [tfma.EvalConfig](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/proto/config.proto)).

```python
from google.protobuf import text_format

config = text_format.Parse("""
  model_specs {
    label_key: "<label-key>"
    example_weight_key: "<example-weight-key>"
  }
  metrics_specs {
    # Add metrics here. For example:
    #  metrics { class_name: "ConfusionMatrixPlot" }
    #  metrics { class_name: "CalibrationPlot" }
  }
  slicing_specs {}
""", tfma.EvalConfig())
```

See [metrics](metrics.md) for more information about other types of metrics that
can be configured.

### How do I setup TFMA to work with a generic TF2 signatures based model? {#generic-tf2}

The following is an example config for a generic TF2 model. Below,
`signature_name` is the name of the specific signature that should be used for
evaluation.

```python
from google.protobuf import text_format

config = text_format.Parse("""
  model_specs {
    signature_name: "<signature-name>"
    label_key: "<label-key>"
    example_weight_key: "<example-weight-key>"
  }
  metrics_specs {
    # Add metrics here. For example:
    #  metrics { class_name: "BinaryCrossentropy" }
    #  metrics { class_name: "ConfusionMatrixPlot" }
    #  metrics { class_name: "CalibrationPlot" }
  }
  slicing_specs {}
""", tfma.EvalConfig())
```

See [metrics](metrics.md) for more information about other types of metrics that
can be configured.

### How do I setup TFMA to work with an estimator based model? {#estimator}

In this case there are three choices:

<style>
.wide-first-col tr td:first-child {
  width: 50%;
}
</style>

| Option                               | Configuration                         | {.wide-first-col}
| ------------------------------------ | ------------------------------------- |
| 1) Use serving model \               | * By default uses `serving_default`   |
: \                                    :   for the for the signature name. \   :
: If this option is used then any      : * Must specify `label_key` and        :
: metrics added during training will   :   `example_weight` key.               :
: NOT be included in the evaluation.   :                                       :
:                                      :                                       :
| 2) Use `EvalSavedModel` for both     | * Use `eval` for the signature name.  |
: feature / prediction extraction and  :                                       :
: evaluation and also add additional   :                                       :
: combiner based metrics.              :                                       :
:                                      :                                       :
| 3) Use `EvalSavedModel` but only for | * Use `eval` for the signature        |
: feature / prediction extraction. \   :   name. \                             :
: \                                    : * Disable `include_default_metrics`.  :
: This option is useful if only        :                                       :
: external metrics are desired, but    :                                       :
: there are feature transformations    :                                       :
: that you would like to slice on.     :                                       :
: Similar to option (1) any metrics    :                                       :
: added during training will NOT be    :                                       :
: included in the evaluation.          :                                       :

NOTE: If using an `EvalSavedModel`, see [EvalSavedModel](eval_saved_model.md)
for more information about setup.

**Option1: Use Serving Model**

The following is an example config for option 1 assuming `serving_default` is
the signature name used:

```python
from google.protobuf import text_format

config = text_format.Parse("""
  model_specs {
    label_key: "<label-key>"
    example_weight_key: "<example-weight-key>"
  }
  metrics_specs {
    # Add metrics here.
  }
  slicing_specs {}
""", tfma.EvalConfig())
```

See [metrics](metrics.md) for more information about other types of metrics that
can be configured.

**Option2: Use EvalSavedModel along with additional combiner-based metrics**

The following is an example config for option 2:

```python
from google.protobuf import text_format

config = text_format.Parse("""
  model_specs {
    signature_name: "eval"
  }
  metrics_specs {
    # Add metrics here.
  }
  slicing_specs {}
""", tfma.EvalConfig())
```

See [metrics](metrics.md) for more information about other types of metrics that
can be configured.

**Option3: Use EvalSavedModel Model only for Feature / Prediction Extraction**

In this case the config is the same as above only `include_default_metrics` is
disabled.

```python
from google.protobuf import text_format

config = text_format.Parse("""
  model_specs {
    signature_name: "eval"
  }
  metrics_specs {
    # Add metrics here.
  }
  slicing_specs {}
  options {
    include_default_metrics { value: false }
  }
""", tfma.EvalConfig())
```

See [metrics](metrics.md) for more information about other types of metrics that
can be configured.


### How do I setup TFMA to work with a keras model-to-estimator based model? {#model-to-estmator}

The keras `model_to_estimator` setup is similar to the [estimator](#estimator)
confiugration. However there are a few differences specific to how model to
estimator works. In particular, the model-to-esimtator returns its outputs in
the form of a dict where the dict key is the name of the last output layer in
the associated keras model (if no name is provided, keras will choose a default
name for you such as `dense_1` or `output_1`). From a TFMA perspective, this
behavior is similar to what would be output for a multi-output model even though
the model to estimator may only be for a single model. To account for this
difference, an additional step is required to setup the output name. However,
the same three options apply as [above](#estimator).

The following is an example of the changes required to an estimator based
config:

```python
from google.protobuf import text_format

config = text_format.Parse("""
  ... as for estimator ...
  metrics_specs {
    output_names: ["<keras-output-layer>"]
    # Add metrics here.
  }
  ... as for estimator ...
""", tfma.EvalConfig())
```

## Metrics

### What types of metrics are supported?

TFMA supports a wide variety of matrics including:

  * [regression metrics](metrics.md#regression-metrics)
  * [binary classification metrics](metrics.md#binary-classification-metrics)
  * [multi-class/multi-label classification metrics](metrics.md#multi-classmulti-label-classification-metrics)
  * [micro average / macro average metrics](metrics.md#multi-classmulti-label-aggregate-metrics)
  * [query / ranking based metrics](metrics.md#query-ranking-based-metrics)

### Are metrics from multi-output models supported?

Yes. See [metrics](metrics.md#multi-output-model-metrics) guide for more
details.

### Are metrics from multiple-models supported?

Yes. See [metrics](metrics.md#multi-model-evaluation-metrics) guide for more
details.

### Can the metric settings (name, etc) be customized?

Yes. Metrics settings can be customized (e.g. setting specific thresholds, etc)
by adding `config` settings to the metric configuration. See
[metrics](metrics.md) guide has more details.

### Are custom metrics supported?

Yes. Either by writing a custom `tf.keras.metrics.Metric` implementation or by
writing a custom `beam.CombineFn` implementation. The
[metrics](metrics.md#customizing-metric-settings) guide has more details.

### What types of metrics are not supported?

As long as your metric can be calculated using a `beam.CombineFn`, there are no
restrictions on the types of metrics that can be computed based on
`tfma.metrics.Metric`. If working with a metric derived from
`tf.keras.metrics.Metric` then the following criteria must be satisfied:

*   It should be possible to compute sufficient statistics for the metric on
    each example independently, then combine these sufficient statistics by
    adding them across all the examples, and determine the metric value solely
    from these sufficient statistics.
*   For example, for accuracy the sufficient statistics are "total correct" and
    "total examples". It’s possible to compute these two numbers for individual
    examples, and add them up for a group of examples to get the right values
    for those examples. The final accuracy can be computed used "total correct /
    total examples".
*   Formally:
    *   Let $${e_1, e_2, \ldots, e_n}$$ be a set of $$n$$ examples.
    *   Let $$M$$ be your metric function, such that $$M(e_1, e_2, \ldots,
        e_n)$$ = metric value over the $$n$$ examples
    *   Let $$\sigma$$ be your function for computing sufficient statistics for
        your metric, such that $$\sigma(e_1, e_2, \ldots, e_n)$$ = sufficient
        statistics over $$n$$ examples
    *   Let $$\mu$$ be your function for computing your metric from the
        sufficient statistics. Your metric must be such that $$M(e_1, e_2,
        \ldots, e_n) = \mu(\sigma(e_1, e_2, \ldots, e_n))$$
    *   Additionally, TFMA requires that: $$\sigma(e_1, e_2, \ldots, e_n) =
        \sigma(e_1) + \sigma(e_2) + \ldots + \sigma(e_n)$$

## Add-ons

### Can I use TFMA to evaluate fairness or bias in my model?

TFMA includes a
[FairnessIndicators](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/addons/fairness/post_export_metrics/fairness_indicators.py)
add-on that provides post-export metrics for evaluating the effects of
unintended bias in classification models.

## Customization

### What if I need more customization?

TFMA is very flexibile and allows you to customize almost all parts of the
pipeline using custom `Extractors`, `Evaluators`, and/or `Writers`. These
abstrations are discusssed in more detail in the [architecture](architecture.md)
document.

## Troubleshooting, debugging, and getting help

### Why do I get an error about prediction key not found?

Some model's output their prediction in the form of a dictionary. For example, a
TF estimator for binary classification problem outputs a dictionary containing
`probabilities`, `class_ids`, etc. In most cases TFMA has defaults for finding
commomly used key names such as `predictions`, `probabilities`, etc. However, if
your model is very customized it may output keys under names not known by TFMA.
In theses cases a `prediciton_key` setting must be added to the `tfma.ModelSpec`
to identify the name of the key the output is stored under.
