# Tensorflow Model Analysis Metrics and Plots

## Overview

TFMA supports the following metrics and plots:

*   Standard keras metrics
    ([`tf.keras.metrics.*`](https://www.tensorflow.org/api_docs/python/tf/keras/metrics))
    *   Note that you do not need a keras model to use keras metrics. Metrics
        are computed outside of the graph in beam using the metrics classes
        directly.
*   Standard TFMA metrics and plots
    ([`tfma.metrics.*`](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/metrics))

*   Custom keras metrics (metrics derived from
    [`tf.keras.metrics.Metric`](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric))

*   Custom TFMA metrics (metrics derived from
    [`tfma.metrics.Metric`](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/metrics/Metric))
    using custom beam combiners or metrics derived from other metrics).

NOTE: In TFMA, plots and metrics are both defined under the metrics library. By
convention the classes related to plots end in `Plot`.

TFMA also provides built-in support for converting binary classification metrics
for use with multi-class/multi-label problems:

*   Binarization based on class ID, top K, etc.
*   Aggregated metrics based on micro averaging, macro averaging, etc.

TFMA also provides built-in support for query/ranking based metrics where the
examples are grouped by a query key automatically in the pipeline.

Combined there are over 50+ standard metrics and plots available for a variety
of problems including regression, binary classification, multi-class/multi-label
classification, ranking, etc.

## Configuration

There are two ways to configure metrics in TFMA: (1) using the
[`tfma.MetricsSpec`](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/api/MetricsSpec)
or (2) by creating instances of `tf.keras.metrics.*` and/or `tfma.metrics.*`
classes in python and using
[`tfma.metrics.specs_from_metrics`](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/api/metrics/specs_from_metrics)
to convert them to a list of `tfma.MetricsSpec`.

The following sections describe example configurations for different types of
machine learning problems.

### Regression Metrics

The following is an example configuration setup for a regression problem.
Consult the `tf.keras.metrics.*` and `tfma.metrics.*` modules for possible
additional metrics supported.

```python
from google.protobuf import text_format

metrics_specs = text_format.Parse("""
  metrics_specs {
    metrics { class_name: "ExampleCount" }
    metrics { class_name: "MeanSquaredError" }
    metrics { class_name: "Accuracy" }
    metrics { class_name: "MeanLabel" }
    metrics { class_name: "MeanPrediction" }
    metrics { class_name: "Calibration" }
    metrics {
      class_name: "CalibrationPlot"
      config: '"min_value": 0, "max_value": 10'
    }
  }
""", tfma.EvalConfig()).metrics_specs
```

This same setup can be created using the following python code:

```python
metrics = [
    tfma.metrics.ExampleCount(name='example_count'),
    tf.keras.metrics.MeanSquaredError(name='mse'),
    tf.keras.metrics.Accuracy(name='accuracy'),
    tfma.metrics.MeanLabel(name='mean_label'),
    tfma.metrics.MeanPrediction(name='mean_prediction'),
    tfma.metrics.Calibration(name='calibration'),
    tfma.metrics.CalibrationPlot(
        name='calibration', min_value=0, max_value=10)
]
metrics_specs = tfma.metrics.specs_from_metrics(metrics)
```

Note that this setup is also avaliable by calling
`tfma.metrics.default_regression_specs`.

### Binary Classification Metrics

The following is an example configuration setup for a binary classification
problem. Consult the `tf.keras.metrics.*` and `tfma.metrics.*` modules for
possible additional metrics supported.

```python
from google.protobuf import text_format

metrics_specs = text_format.Parse("""
  metrics_specs {
    metrics { class_name: "ExampleCount" }
    metrics { class_name: "BinaryCrossentropy" }
    metrics { class_name: "BinaryAccuracy" }
    metrics { class_name: "AUC" }
    metrics { class_name: "AUCPrecisionRecall" }
    metrics { class_name: "MeanLabel" }
    metrics { class_name: "MeanPrediction" }
    metrics { class_name: "Calibration" }
    metrics { class_name: "ConfusionMatrixPlot" }
    metrics { class_name: "CalibrationPlot" }
  }
""", tfma.EvalConfig()).metrics_specs
```

This same setup can be created using the following python code:

```python
metrics = [
    tfma.metrics.ExampleCount(name='example_count'),
    tf.keras.metrics.BinaryCrossentropy(name='binary_crossentropy'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.AUC(name='auc', num_thresholds=10000),
    tf.keras.metrics.AUC(
        name='auc_precision_recall', curve='PR', num_thresholds=10000),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tfma.metrics.MeanLabel(name='mean_label'),
    tfma.metrics.MeanPrediction(name='mean_prediction'),
    tfma.metrics.Calibration(name='calibration'),
    tfma.metrics.ConfusionMatrixPlot(name='confusion_matrix_plot'),
    tfma.metrics.CalibrationPlot(name='calibration_plot')
]
metrics_specs = tfma.metrics.specs_from_metrics(metrics)
```

Note that this setup is also avaliable by calling
`tfma.metrics.default_binary_classification_specs`.

### Multi-class/Multi-label Classification Metrics

The following is an example configuration setup for a multi-class classification
problem. Consult the `tf.keras.metrics.*` and `tfma.metrics.*` modules for
possible additional metrics supported.

```python
from google.protobuf import text_format

metrics_specs = text_format.Parse("""
  metrics_specs {
    metrics { class_name: "ExampleCount" }
    metrics { class_name: "SparseCategoricalCrossentropy" }
    metrics { class_name: "SparseCategoricalAccuracy" }
    metrics { class_name: "Precision" config: '"top_k": 1' }
    metrics { class_name: "Precision" config: '"top_k": 3' }
    metrics { class_name: "Recall" config: '"top_k": 1' }
    metrics { class_name: "Recall" config: '"top_k": 3' }
    metrics { class_name: "MultiClassConfusionMatrixPlot" }
  }
""", tfma.EvalConfig()).metrics_specs
```

Note: For multi-label there is `MultiLabelConfusionMatrixPlot` instead of
`MultiClassConfusionMatrixPlot`

This same setup can be created using the following python code:

```python
metrics = [
    tfma.metrics.ExampleCount(name='example_count'),
    tf.keras.metrics.SparseCategoricalCrossentropy(
        name='sparse_categorical_crossentropy'),
    tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision', top_k=1),
    tf.keras.metrics.Precision(name='precision', top_k=3),
    tf.keras.metrics.Recall(name='recall', top_k=1),
    tf.keras.metrics.Recall(name='recall', top_k=3),
    tfma.metrics.MultiClassConfusionMatrixPlot(
        name='multi_class_confusion_matrix_plot'),
]
metrics_specs = tfma.metrics.specs_from_metrics(metrics)
```

Note that this setup is also avaliable by calling
`tfma.metrics.default_multi_class_classification_specs`.

### Multi-class/Multi-label Binarized Metrics

Multi-class/multi-label metrics can be binarized to produce metrics per class,
per top_k, etc using the `tfma.BinarizationOptions`. For example:

```python
from google.protobuf import text_format

metrics_specs = text_format.Parse("""
  metrics_specs {
    binarize: { class_ids: { values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] } }
    // Metrics to binarize
    metrics { class_name: "AUC" }
    ...
  }
""", tfma.EvalConfig()).metrics_specs
```

This same setup can be created using the following python code:

```python
metrics = [
    // Metrics to binarize
    tf.keras.metrics.AUC(name='auc', num_thresholds=10000),
    ...
]
metrics_specs = tfma.metrics.specs_from_metrics(
    metrics, binarize=tfma.BinarizationOptions(
        class_ids={'values': [0,1,2,3,4,5,6,7,8,9]}))
```

### Multi-class/Multi-label Aggregate Metrics

Multi-class/multi-label metrics can be aggregated to produce a single aggregated
value for a binary classification metric by using `tfma.AggregationOptions`.

Note that aggregation settings are independent of binarization settings so you
can use both `tfma.AggregationOptions` and `tfma.BinarizationOptions` at the
same time.

#### Micro Average

Micro averaging can be performed by using the `micro_average` option within
`tfma.AggregationOptions`. For example:

```python
from google.protobuf import text_format

metrics_specs = text_format.Parse("""
  metrics_specs {
    aggregate: { micro_average: true }
    // Metrics to aggregate
    metrics { class_name: "AUC" }
    ...
  }
""", tfma.EvalConfig()).metrics_specs
```

This same setup can be created using the following python code:

```python
metrics = [
    // Metrics to aggregate
    tf.keras.metrics.AUC(name='auc', num_thresholds=10000),
    ...
]
metrics_specs = tfma.metrics.specs_from_metrics(
    metrics, aggregate=tfma.AggregationOptions(micro_average=True))
```

Micro averaging also supports setting `top_k` where only the top k values are
used in the computation. For example:

```python
from google.protobuf import text_format

metrics_specs = text_format.Parse("""
  metrics_specs {
    aggregate: {
      micro_average: true
      top_k_list: { values: [1, 3] }
    }
    // Metrics to aggregate
    metrics { class_name: "AUC" }
    ...
  }
""", tfma.EvalConfig()).metrics_specs
```

This same setup can be created using the following python code:

```python
metrics = [
    // Metrics to aggregate
    tf.keras.metrics.AUC(name='auc', num_thresholds=10000),
    ...
]
metrics_specs = tfma.metrics.specs_from_metrics(
    metrics,
    aggregate=tfma.AggregationOptions(micro_average=True,
                                      top_k_list={'values': [1, 3]}))
```

#### Macro / Weighted Macro Average

Macro averaging can be performed by using the `macro_average` or
`weighted_macro_average` options within `tfma.AggregationOptions`. Unless
`top_k` settings are used, macro requires setting the `class_weights` in order
to know which classes to compute the average for. If a `class_weight` is not
provided then 0.0 is assumed. For example:

```python
from google.protobuf import text_format

metrics_specs = text_format.Parse("""
  metrics_specs {
    aggregate: {
      macro_average: true
      class_weights: { key: 0 value: 1.0 }
      class_weights: { key: 1 value: 1.0 }
      class_weights: { key: 2 value: 1.0 }
      class_weights: { key: 3 value: 1.0 }
      class_weights: { key: 4 value: 1.0 }
      class_weights: { key: 5 value: 1.0 }
      class_weights: { key: 6 value: 1.0 }
      class_weights: { key: 7 value: 1.0 }
      class_weights: { key: 8 value: 1.0 }
      class_weights: { key: 9 value: 1.0 }
    }
    // Metrics to aggregate
    metrics { class_name: "AUC" }
    ...
  }
""", tfma.EvalConfig()).metrics_specs
```

This same setup can be created using the following python code:

```python
metrics = [
    // Metrics to aggregate
    tf.keras.metrics.AUC(name='auc', num_thresholds=10000),
    ...
]
metrics_specs = tfma.metrics.specs_from_metrics(
    metrics,
    aggregate=tfma.AggregationOptions(
        macro_average=True, class_weights={i: 1.0 for i in range(10)}))
```

Like micro averaging, macro averaging also supports setting `top_k` where only
the top k values are used in the computation. For example:

```python
from google.protobuf import text_format

metrics_specs = text_format.Parse("""
  metrics_specs {
    aggregate: {
      macro_average: true
      top_k_list: { values: [1, 3] }
    }
    // Metrics to aggregate
    metrics { class_name: "AUC" }
    ...
  }
""", tfma.EvalConfig()).metrics_specs
```

This same setup can be created using the following python code:

```python
metrics = [
    // Metrics to aggregate
    tf.keras.metrics.AUC(name='auc', num_thresholds=10000),
    ...
]
metrics_specs = tfma.metrics.specs_from_metrics(
    metrics,
    aggregate=tfma.AggregationOptions(macro_average=True,
                                      top_k_list={'values': [1, 3]}))
```

### Query / Ranking Based Metrics

Query/ranking based metrics are enabled by specifying the `query_key` option in
the metrics specs. For example:

```python
from google.protobuf import text_format

metrics_specs = text_format.Parse("""
  metrics_specs {
    query_key: "doc_id"
    metrics {
      class_name: "NDCG"
      config: '"gain_key": "gain", "top_k_list": [1, 2]'
    }
    metrics { class_name: "MinLabelPosition" }
  }
""", tfma.EvalConfig()).metrics_specs
```

This same setup can be created using the following python code:

```python
metrics = [
    tfma.metrics.NDCG(name='ndcg', gain_key='gain', top_k_list=[1, 2]),
    tfma.metrics.MinLabelPosition(name='min_label_position')
]
metrics_specs = tfma.metrics.specs_from_metrics(metrics, query_key='doc_id')
```

### Multi-model Evaluation Metrics

TFMA supports evaluating multiple models at the same time. When multi-model
evaluation is performed, metrics will be calculated for each model. For example:

```python
from google.protobuf import text_format

metrics_specs = text_format.Parse("""
  metrics_specs {
    # no model_names means all models
    ...
  }
""", tfma.EvalConfig()).metrics_specs
```

If metrics need to be computed for a subset of models, set `model_names` in the
`metric_specs`. For example:

```python
from google.protobuf import text_format

metrics_specs = text_format.Parse("""
  metrics_specs {
    model_names: ["my-model1"]
    ...
  }
""", tfma.EvalConfig()).metrics_specs
```

The `specs_from_metrics` API also supports passing model names:

```python
metrics = [
    ...
]
metrics_specs = tfma.metrics.specs_from_metrics(
    metrics, model_names=['my-model1'])
```

### Model Comparison Metrics

TFMA supports evaluating comparison metrics for a candidate model against a
baseline model. A simple way to setup the candidate and baseline model pair is
to pass along a eval_shared_model with the proper model names (tfma.BASELINE_KEY
and tfma.CANDIDATE_KEY):

```python

eval_config = text_format.Parse("""
  model_specs {
    # ... model_spec without names ...
  }
  metrics_spec {
    # ... metrics ...
  }
""", tfma.EvalConfig())

eval_shared_models = [
  tfma.default_eval_shared_model(
      model_name=tfma.CANDIDATE_KEY,
      eval_saved_model_path='/path/to/saved/candidate/model',
      eval_config=eval_config),
  tfma.default_eval_shared_model(
      model_name=tfma.BASELINE_KEY,
      eval_saved_model_path='/path/to/saved/baseline/model',
      eval_config=eval_config),
]

eval_result = tfma.run_model_analysis(
    eval_shared_models,
    eval_config=eval_config,
    # This assumes your data is a TFRecords file containing records in the
    # tf.train.Example format.
    data_location="/path/to/file/containing/tfrecords",
    output_path="/path/for/output")
```

Comparison metrics are computed automatically for all of the diff-able metrics
(currently only scalar value metrics such as accuracy and AUC).

### Multi-output Model Metrics

TFMA supports evaluating metrics on models that have different outputs.
Multi-output models store their output predictions in the form of a dict keyed
by output name. When multi-output model's are used, the names of the outputs
associated with a set of metrics must be specified in the `output_names` section
of the MetricsSpec. For example:

```python
from google.protobuf import text_format

metrics_specs = text_format.Parse("""
  metrics_specs {
    output_names: ["my-output"]
    ...
  }
""", tfma.EvalConfig()).metrics_specs
```

The `specs_from_metrics` API also supports passing output names:

```python
metrics = [
    ...
]
metrics_specs = tfma.metrics.specs_from_metrics(
    metrics, output_names=['my-output'])
```

### Customizing Metric Settings

TFMA allows customizing of the settings that are used with different metrics.
For example you might want to change the name, set thresholds, etc. This is done
by adding a `config` section to the metric config. The config is specified using
the JSON string version of the parameters that would be passed to the metrics
`__init__` method (for ease of use the leading and trailing '{' and '}' brackets
may be omitted). For example:

```python
from google.protobuf import text_format

metrics_specs = text_format.Parse("""
  metrics_specs {
    metrics {
      class_name: "ConfusionMatrixAtThresholds"
      config: '"thresholds": [0.3, 0.5, 0.8]'
    }
  }
""", tfma.MetricsSpec()).metrics_specs
```

This customization is of course also supported directly:

```python
metrics = [
   tfma.metrics.ConfusionMatrixAtThresholds(thresholds=[0.3, 0.5, 0.8]),
]
metrics_specs = tfma.metrics.specs_from_metrics(metrics)
```

NOTE: It is advisable to set the default number of thresholds used with AUC, etc
to 10000 because this is the default value used by the underlying histogram
calcuation which is shared between multiple metric implementations.

## Outputs

The output of a metric evaluation is a series of metric keys/values and/or plot
keys/values based on the configuration used.

### Metric Keys

[MetricKeys](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/proto/metrics_for_slice.proto)
are defined using a structured key type. This key uniquely identifies each of
the following aspects of a metric:

*   Metric name (`auc`, `mean_label`, etc)
*   Model name (only used if multi-model evaluation)
*   Output name (only used if multi-output models are evaluated)
*   Sub key (e.g. class ID if multi-class model is binarized)

### Metric Value

[MetricValues](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/proto/metrics_for_slice.proto)
are defined using a proto that encapulates the different value types supported
by the different metrics (e.g. `double`, `ConfusionMatrixAtThresholds`, etc).

Below are the supported metric value types:

*   [`double_value`](https://developers.google.com/protocol-buffers/docs/reference/csharp/class/google/protobuf/well-known-types/double-value) -
    A wrapper for a double type.
*   [`bytes_value`](https://developers.google.com/protocol-buffers/docs/proto3) -
    A bytes value.
*   `bounded_value` - Represents a real value which could be a pointwise
    estimate, optionally with approximate bounds of some sort. Has properties
    `value`, `lower_bound`, and `upper_bound`.
*   `value_at_cutoffs` - Value at cutoffs (e.g. precision@K, recall@K). Has
    property `values`, each of which has properties `cutoff` and `value`.
*   `confusion_matrix_at_thresholds` - Confusion matrix at thresholds. Has
    property `matrices`, each of which has properties for `threshold`,
    `precision`, `recall`, and confusion matrix values such as
    `false_negatives`.
*   `array_value` - For metrics which return an array of values.

### Plot Keys

[PlotKeys](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/proto/metrics_for_slice.proto)
are similar to metric keys except that for historical reasons all the plots
values are stored in a single proto so the plot key does not have a name.

### Plot Values

All the supported plots are stored in a single proto called
[PlotData](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/proto/metrics_for_slice.proto).

### EvalResult

The return from an evaluation run is an
[`tfma.EvalResult`](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/EvalResult).
This record contains `slicing_metrics` that encode the metric key as a
multi-level dict where the levels correspond to output name, class ID, metric
name, and metric value respectively. This is intended to be used for UI display
in a Jupiter notebook. If access to the underlying data is needed the `metrics`
result file should be used instead (see
[metrics_for_slice.proto](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/proto/metrics_for_slice.proto)).

## Customization

In addition to custom metrics that are added as part of a saved keras (or legacy
EvalSavedModel). There are two ways to customize metrics in TFMA post saving:
(1) by defining a custom keras metric class and (2) by defining a custom TFMA
metrics class backed by a beam combiner.

In both cases, the metrics are configured by specifying the name of the metric
class and associated module. For example:

```python
from google.protobuf import text_format

metrics_specs = text_format.Parse("""
  metrics_specs {
    metrics { class_name: "MyMetric" module: "my.module"}
  }
""", tfma.EvalConfig()).metrics_specs
```

NOTE: When customizing metrics you must ensure that the module is available to
beam.

### Custom Keras Metrics

To create a custom keras metric, users need to extend `tf.keras.metrics.Metric`
with their implementation and then make sure the metric's module is available at
evaluation time.

Note that for metrics added post model save, TFMA only supports metrics that
take label (i.e. y_true), prediction (y_pred), and example weight
(sample_weight) as parameters to the `update_state` method.

#### Keras Metric Example

The following is an example of a custom keras metric:

```python
class MyMetric(tf.keras.metrics.Mean):

  def __init__(self, name='my_metric', dtype=None):
    super(MyMetric, self).__init__(name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super(MyMetric, self).update_state(
        y_pred, sample_weight=sample_weight)
```

### Custom TFMA Metrics

To create a custom TFMA metric, users need to extend `tfma.metrics.Metric` with
their implementation and then make sure the metric's module is available at
evaluation time.

#### Metric

A `tfma.metrics.Metric` implementation is made up of a set of kwargs that define
the metrics configuration along with a function for creating the computations
(possibly multiple) needed to calcuate the metrics value. There are two main
computation types that can be used: `tfma.metrics.MetricComputation` and
`tfma.metrics.DerivedMetricComputation` that are described in the sections
below. The function that creates these computations will be passed the following
parameters as input:

*   `eval_config: tfam.EvalConfig`
    *   The eval config passed to the evaluator (useful for looking up model
        spec settings such as prediction key to use, etc).
*   `model_names: List[Text]`
    *   List of model names to compute metrics for (None if single-model)
*   `output_names: List[Text]`.
    *   List of output names to compute metrics for (None if single-model)
*   `sub_keys: List[tfma.SubKey]`.
    *   List of sub keys (class ID, top K, etc) to compute metrics for (or None)
*   `aggregation_type: tfma.AggregationType`
    *   Type of aggregation if computing an aggregation metric.
*   `class_weights: Dict[int, float]`.
    *   Class weights to use if computing an aggregation metric.
*   `query_key: Text`
    *   Query key used if computing a query/ranking based metric.

If a metric is not associated with one or more of these settings then it may
leave those parameters out of its signature definition.

If a metric is computed the same way for each model, output, and sub key, then
the utility `tfma.metrics.merge_per_key_computations` can be used to perform the
same computations for each of these inputs separately.

#### MetricComputation

A `MetricComputation` is made up of a combination of `preprocessors` and a
`combiner`. The `preprocessors` is a list of `preprocessor`, which is a
`beam.DoFn` that takes extracts as its input and outputs the initial state that
will be used by the combiner (see [architecture](architecture.md) for more info
on what are extracts). All preprocessors will be executed sequentially in the
order of the list. If the `preprocessors` is empty, then the combiner will be
passed
[StandardMetricInputs](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/metrics/StandardMetricInputs)
(standard metric inputs contains labels, predictions, and example_weights). The
`combiner` is a `beam.CombineFn` that takes a tuple of (slice key, preprocessor
output) as its input and outputs a tuple of (slice_key, metric results dict) as
its result.

Note that slicing happens between the `preprocessors` and `combiner`.

Note that if a metric computation wants to make use of both the standard metric
inputs, but augment it with a few of the features from the `features` extracts,
then the special
[FeaturePreprocessor](https://www.tensorflow.org/tfx/model_analysis/api_docs/python/tfma/metrics/FeaturePreprocessor)
can be used which will merge the requested features from multiple combiners into
a single shared StandardMetricsInputs value that is passed to all the combiners
(the combiners are responsible for reading the features they are interested in
and ignoring the rest).

##### Example

The following is a very simple example of TFMA metric definition for computing
the ExampleCount:

```python
class ExampleCount(tfma.metrics.Metric):

  def __init__(self, name: Text = 'example_count'):
    super(ExampleCount, self).__init__(_example_count, name=name)


def _example_count(
    name: Text = 'example_count') -> tfma.metrics.MetricComputations:
  key = tfma.metrics.MetricKey(name=name)
  return [
      tfma.metrics.MetricComputation(
          keys=[key],
          preprocessors=[_ExampleCountPreprocessor()],
          combiner=_ExampleCountCombiner(key))
  ]


class ExampleCountTest(tfma.test.testutil.TensorflowModelAnalysisTest):

  def testExampleCount(self):
    metric = ExampleCount()
    computations = metric.computations(example_weighted=False)
    computation = computations[0]

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create([...])  # Add inputs
          | 'PreProcess' >> beam.ParDo(computation.preprocessors[0])
          | 'Process' >> beam.Map(tfma.metrics.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(computation.combiner)
      )

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = computation.keys[0]
          self.assertIn(key, got_metrics)
          self.assertAlmostEqual(got_metrics[key], expected_value, places=5)
        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

class _ExampleCountPreprocessor(beam.DoFn):

  def process(self, extracts: tfma.Extracts) -> Iterable[int]:
    yield 1


class _ExampleCountPreprocessorTest(unittest.TestCase):

  def testExampleCountPreprocessor(self):
    ...  # Init the test case here
    with beam.Pipeline() as pipeline:
      updated_pcoll = (
          pipeline
          | 'Create' >> beam.Create([...])  # Add inputs
          | 'Preprocess'
          >> beam.ParDo(
              _ExampleCountPreprocessor()
          )
      )

      beam_testing_util.assert_that(
          updated_pcoll,
          lambda result: ...,  # Assert the test case
      )


class _ExampleCountCombiner(beam.CombineFn):

  def __init__(self, metric_key: tfma.metrics.MetricKey):
    self._metric_key = metric_key

  def create_accumulator(self) -> int:
    return 0

  def add_input(self, accumulator: int, state: int) -> int:
    return accumulator + state

  def merge_accumulators(self, accumulators: Iterable[int]) -> int:
    accumulators = iter(accumulators)
    result = next(accumulator)
    for accumulator in accumulators:
      result += accumulator
    return result

  def extract_output(self,
                     accumulator: int) -> Dict[tfma.metrics.MetricKey, int]:
    return {self._metric_key: accumulator}
```

#### DerivedMetricComputation

A `DerivedMetricComputation` is made up of a result function that is used to
calculate metric values based on the output of other metric computations. The
result function takes a dict of computed values as its input and outputs a dict
of additional metric results.

Note that it is acceptable (recommended) to include the computations that a
derived computation depends on in the list of computations created by a metric.
This avoid having to pre-create and pass computations that are shared between
multiple metrics. The evaluator will automatically de-dup computations that have
the same definition so ony one computation is actually run.

##### Example

The
[TJUR metrics](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/metrics/tjur_discrimination.py)
provides a good example of derived metrics.
