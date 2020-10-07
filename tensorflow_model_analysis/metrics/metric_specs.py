# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Specifications for common metrics."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import collections
import copy
import importlib
import json
import re

from typing import Any, Dict, FrozenSet, Iterator, Iterable, List, NamedTuple, Optional, Text, Type, Union, Tuple

import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis.metrics import aggregation
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import calibration
from tensorflow_model_analysis.metrics import calibration_plot
from tensorflow_model_analysis.metrics import confusion_matrix_plot
from tensorflow_model_analysis.metrics import example_count
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import multi_class_confusion_matrix_plot
from tensorflow_model_analysis.metrics import tf_metric_wrapper
from tensorflow_model_analysis.metrics import weighted_example_count
from tensorflow_metadata.proto.v0 import schema_pb2

_TF_LOSSES_MODULE = tf.keras.losses.Loss().__class__.__module__

_TFOrTFMAMetric = Union[tf.keras.metrics.Metric, tf.keras.losses.Loss,
                        metric_types.Metric]
_TFMetricOrLoss = Union[tf.keras.metrics.Metric, tf.keras.losses.Loss]

# TF configs that should be treated special by either modifying the class names
# used or updating the default config settings.
_TF_CONFIG_DEFAULTS = {
    'AUC': {
        'class_name': 'AUC',
        'config': {
            'num_thresholds': binary_confusion_matrices.DEFAULT_NUM_THRESHOLDS
        }
    },
    'AUCPrecisionRecall': {
        'class_name': 'AUC',
        'config': {
            'name': 'auc_precision_recall',
            'num_thresholds': binary_confusion_matrices.DEFAULT_NUM_THRESHOLDS,
            'curve': 'PR'
        }
    }
}


def specs_from_metrics(
    metrics: Union[List[_TFOrTFMAMetric], Dict[Text, List[_TFOrTFMAMetric]]],
    model_names: Optional[List[Text]] = None,
    output_names: Optional[List[Text]] = None,
    binarize: Optional[config.BinarizationOptions] = None,
    aggregate: Optional[config.AggregationOptions] = None,
    query_key: Optional[Text] = None,
    include_example_count: Optional[bool] = None,
    include_weighted_example_count: Optional[bool] = None
) -> List[config.MetricsSpec]:
  """Returns specs for tf.keras.metrics/losses or tfma.metrics classes.

  Examples:

    metrics_specs = specs_from_metrics([
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.AUC(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tfma.metrics.MeanLabel(),
        tfma.metrics.MeanPrediction()
        ...
    ])

    metrics_specs = specs_from_metrics({
      'output1': [
          tf.keras.metrics.BinaryAccuracy(),
          tf.keras.metrics.AUC(),
          tfma.metrics.MeanLabel(),
          tfma.metrics.MeanPrediction()
          ...
      ],
      'output2': [
          tf.keras.metrics.Precision(),
          tf.keras.metrics.Recall(),
      ]
    })

  Args:
    metrics: List of tf.keras.metrics.Metric, tf.keras.losses.Loss, or
      tfma.metrics.Metric. For multi-output models a dict of dicts may be passed
      where the first dict is indexed by the output_name.
    model_names: Optional model names (if multi-model evaluation).
    output_names: Optional output names (if multi-output models). If the metrics
      are a dict this should not be set.
    binarize: Optional settings for binarizing multi-class/multi-label metrics.
    aggregate: Optional settings for aggregating multi-class/multi-label
      metrics.
    query_key: Optional query key for query/ranking based metrics.
    include_example_count: True to add example_count metric. Default is True.
    include_weighted_example_count: True to add weighted_example_count metric.
      Default is True. A weighted example count will be added per output for
      multi-output models.
  """
  if isinstance(metrics, dict) and output_names:
    raise ValueError('metrics cannot be a dict when output_names is used: '
                     'metrics={}, output_names={}'.format(
                         metrics, output_names))
  if isinstance(metrics, dict):
    specs = []
    for output_name in sorted(metrics.keys()):
      specs.extend(
          specs_from_metrics(
              metrics[output_name],
              model_names=model_names,
              output_names=[output_name],
              binarize=binarize,
              aggregate=aggregate,
              include_example_count=include_example_count,
              include_weighted_example_count=include_weighted_example_count))
      include_example_count = False
    return specs

  if include_example_count is None:
    include_example_count = True
  if include_weighted_example_count is None:
    include_weighted_example_count = True

  # Add the computations for the example counts and weights since they are
  # independent of the model and class ID.
  specs = example_count_specs(
      model_names=model_names,
      output_names=output_names,
      include_example_count=include_example_count,
      include_weighted_example_count=include_weighted_example_count)

  metric_configs = []
  for metric in metrics:
    if isinstance(metric, tf.keras.metrics.Metric):
      metric_configs.append(_serialize_tf_metric(metric))
    elif isinstance(metric, tf.keras.losses.Loss):
      metric_configs.append(_serialize_tf_loss(metric))
    elif isinstance(metric, metric_types.Metric):
      metric_configs.append(_serialize_tfma_metric(metric))
    else:
      raise NotImplementedError('unknown metric type {}: metric={}'.format(
          type(metric), metric))
  specs.append(
      config.MetricsSpec(
          metrics=metric_configs,
          model_names=model_names,
          output_names=output_names,
          binarize=binarize,
          aggregate=aggregate,
          query_key=query_key))

  return specs


def example_count_specs(
    model_names: Optional[List[Text]] = None,
    output_names: Optional[List[Text]] = None,
    include_example_count: bool = True,
    include_weighted_example_count: bool = True) -> List[config.MetricsSpec]:
  """Returns metric specs for example count and weighted example counts.

  Args:
    model_names: Optional list of model names (if multi-model evaluation).
    output_names: Optional list of output names (if multi-output model).
    include_example_count: True to add example_count metric.
    include_weighted_example_count: True to add weighted_example_count metric. A
      weighted example count will be added per output for multi-output models.
  """
  specs = []
  if include_example_count:
    metric_config = _serialize_tfma_metric(example_count.ExampleCount())
    specs.append(
        config.MetricsSpec(metrics=[metric_config], model_names=model_names))
  if include_weighted_example_count:
    metric_config = _serialize_tfma_metric(
        weighted_example_count.WeightedExampleCount())
    specs.append(
        config.MetricsSpec(
            metrics=[metric_config],
            model_names=model_names,
            output_names=output_names))
  return specs


def default_regression_specs(
    model_names: Optional[List[Text]] = None,
    output_names: Optional[List[Text]] = None,
    loss_functions: Optional[List[Union[tf.keras.metrics.Metric,
                                        tf.keras.losses.Loss]]] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None) -> List[config.MetricsSpec]:
  """Returns default metric specs for for regression problems.

  Args:
    model_names: Optional model names (if multi-model evaluation).
    output_names: Optional list of output names (if multi-output model).
    loss_functions: Loss functions to use (if None MSE is used).
    min_value: Min value for calibration plot (if None no plot will be created).
    max_value: Max value for calibration plot (if None no plot will be created).
  """

  if loss_functions is None:
    loss_functions = [tf.keras.metrics.MeanSquaredError(name='mse')]

  metrics = [
      tf.keras.metrics.Accuracy(name='accuracy'),
      calibration.MeanLabel(name='mean_label'),
      calibration.MeanPrediction(name='mean_prediction'),
      calibration.Calibration(name='calibration'),
  ]
  for fn in loss_functions:
    metrics.append(fn)
  if min_value is not None and max_value is not None:
    metrics.append(
        calibration_plot.CalibrationPlot(
            name='calibration_plot', left=min_value, right=max_value))

  return specs_from_metrics(
      metrics, model_names=model_names, output_names=output_names)


def default_binary_classification_specs(
    model_names: Optional[List[Text]] = None,
    output_names: Optional[List[Text]] = None,
    binarize: Optional[config.BinarizationOptions] = None,
    aggregate: Optional[config.AggregationOptions] = None,
    include_loss: bool = True) -> List[config.MetricsSpec]:
  """Returns default metric specs for binary classification problems.

  Args:
    model_names: Optional model names (if multi-model evaluation).
    output_names: Optional list of output names (if multi-output model).
    binarize: Optional settings for binarizing multi-class/multi-label metrics.
    aggregate: Optional settings for aggregating multi-class/multi-label
      metrics.
    include_loss: True to include loss.
  """

  metrics = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.AUC(
          name='auc',
          num_thresholds=binary_confusion_matrices.DEFAULT_NUM_THRESHOLDS),
      tf.keras.metrics.AUC(
          name='auc_precison_recall',  # Matches default name used by estimator.
          curve='PR',
          num_thresholds=binary_confusion_matrices.DEFAULT_NUM_THRESHOLDS),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      calibration.MeanLabel(name='mean_label'),
      calibration.MeanPrediction(name='mean_prediction'),
      calibration.Calibration(name='calibration'),
      confusion_matrix_plot.ConfusionMatrixPlot(name='confusion_matrix_plot'),
      calibration_plot.CalibrationPlot(name='calibration_plot')
  ]
  if include_loss:
    metrics.append(tf.keras.metrics.BinaryCrossentropy(name='loss'))

  return specs_from_metrics(
      metrics,
      model_names=model_names,
      output_names=output_names,
      binarize=binarize,
      aggregate=aggregate)


def default_multi_class_classification_specs(
    model_names: Optional[List[Text]] = None,
    output_names: Optional[List[Text]] = None,
    binarize: Optional[config.BinarizationOptions] = None,
    aggregate: Optional[config.AggregationOptions] = None,
    sparse: bool = True) -> List[config.MetricsSpec]:
  """Returns default metric specs for multi-class classification problems.

  Args:
    model_names: Optional model names if multi-model evaluation.
    output_names: Optional list of output names (if multi-output model).
    binarize: Optional settings for binarizing multi-class/multi-label metrics.
    aggregate: Optional settings for aggregating multi-class/multi-label
      metrics.
    sparse: True if the labels are sparse.
  """

  if sparse:
    metrics = [
        tf.keras.metrics.SparseCategoricalCrossentropy(name='loss'),
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    ]
  else:
    metrics = [
        tf.keras.metrics.CategoricalCrossentropy(name='loss'),
        tf.keras.metrics.CategoricalAccuracy(name='accuracy')
    ]
  metrics.append(
      multi_class_confusion_matrix_plot.MultiClassConfusionMatrixPlot())
  if binarize is not None:
    for top_k in binarize.top_k_list.values:
      metrics.extend([
          tf.keras.metrics.Precision(name='precision', top_k=top_k),
          tf.keras.metrics.Recall(name='recall', top_k=top_k)
      ])
    binarize_without_top_k = config.BinarizationOptions()
    binarize_without_top_k.CopyFrom(binarize)
    binarize_without_top_k.ClearField('top_k_list')
    binarize = binarize_without_top_k
  multi_class_metrics = specs_from_metrics(
      metrics, model_names=model_names, output_names=output_names)
  if aggregate is None:
    aggregate = config.AggregationOptions(micro_average=True)
  multi_class_metrics.extend(
      default_binary_classification_specs(
          model_names=model_names,
          output_names=output_names,
          binarize=binarize,
          aggregate=aggregate))
  return multi_class_metrics


def _keys_for_metric(
    metric_name: Text, spec: config.MetricsSpec,
    aggregation_type: Optional[metric_types.AggregationType],
    sub_keys: List[Optional[metric_types.SubKey]]
) -> Iterator[metric_types.MetricKey]:
  """Yields all non-diff keys for a specific metric name."""
  for model_name in spec.model_names or ['']:
    for output_name in spec.output_names or ['']:
      for sub_key in sub_keys:
        key = metric_types.MetricKey(
            name=metric_name,
            model_name=model_name,
            output_name=output_name,
            sub_key=sub_key,
            aggregation_type=aggregation_type)
        yield key


def _keys_and_metrics_from_specs(
    metrics_specs: Iterable[config.MetricsSpec]
) -> Iterator[Tuple[metric_types.MetricKey, config.MetricConfig,
                    metric_types.Metric]]:
  """Yields key, config, instance tuples for each non-diff metric in specs."""
  tfma_metric_classes = metric_types.registered_metrics()
  for spec in metrics_specs:
    for aggregation_type, sub_keys in _create_sub_keys(spec).items():
      for metric_config in spec.metrics:
        if metric_config.class_name in tfma_metric_classes:
          instance = _deserialize_tfma_metric(metric_config,
                                              tfma_metric_classes)
        elif not metric_config.module:
          instance = _deserialize_tf_metric(metric_config, {})
        else:
          cls = getattr(
              importlib.import_module(metric_config.module),
              metric_config.class_name)
          if issubclass(cls, tf.keras.metrics.Metric):
            instance = _deserialize_tf_metric(metric_config,
                                              {metric_config.class_name: cls})
          elif issubclass(cls, tf.keras.losses.Loss):
            instance = _deserialize_tf_loss(metric_config,
                                            {metric_config.class_name: cls})
          elif issubclass(cls, metric_types.Metric):
            instance = _deserialize_tfma_metric(metric_config,
                                                {metric_config.class_name: cls})
          else:
            raise NotImplementedError(
                'unknown metric type {}: metric={}'.format(cls, metric_config))

        for key in _keys_for_metric(instance.name, spec, aggregation_type,
                                    sub_keys):
          yield key, metric_config, instance


def metric_keys_to_skip_for_confidence_intervals(
    metrics_specs: Iterable[config.MetricsSpec]
) -> FrozenSet[metric_types.MetricKey]:
  """Returns metric keys not to be displayed with confidence intervals."""
  skipped_keys = []
  for key, _, instance in _keys_and_metrics_from_specs(metrics_specs):
    # if metric does not implement compute_confidence_interval, do not skip
    if not getattr(instance, 'compute_confidence_interval', True):
      skipped_keys.append(key)
  return frozenset(skipped_keys)


# Optional slice and associated threshold setting. If slice is not set it
# matches all slices.
_SliceAndThreshold = Tuple[Optional[Union[config.SlicingSpec,
                                          config.CrossSlicingSpec]],
                           Union[config.GenericChangeThreshold,
                                 config.GenericValueThreshold]]


def metric_thresholds_from_metrics_specs(
    metrics_specs: Iterable[config.MetricsSpec]
) -> Dict[metric_types.MetricKey, Iterable[_SliceAndThreshold]]:
  """Returns thresholds associated with given metrics specs."""
  result = collections.defaultdict(list)
  existing = collections.defaultdict(dict)

  def add_if_not_exists(key: metric_types.MetricKey,
                        slice_spec: Optional[Union[config.SlicingSpec,
                                                   config.CrossSlicingSpec]],
                        threshold: Union[config.GenericChangeThreshold,
                                         config.GenericValueThreshold]):
    """Adds value to results if it doesn't already exist."""
    # Note that hashing by SerializeToString() is only safe if used within the
    # same process.
    slice_hash = slice_spec.SerializeToString() if slice_spec else None
    threshold_hash = threshold.SerializeToString()
    if (not (key in existing and slice_hash in existing[key] and
             threshold_hash in existing[key][slice_hash])):
      if slice_hash not in existing[key]:
        existing[key][slice_hash] = {}
      existing[key][slice_hash][threshold_hash] = True
      result[key].append((slice_spec, threshold))

  def add_threshold(key: metric_types.MetricKey,
                    slice_spec: Union[Optional[config.SlicingSpec],
                                      Optional[config.CrossSlicingSpec]],
                    threshold: config.MetricThreshold):
    """Adds thresholds to results."""
    if threshold.HasField('value_threshold'):
      add_if_not_exists(key, slice_spec, threshold.value_threshold)
    if threshold.HasField('change_threshold'):
      key = key.make_diff_key()
      add_if_not_exists(key, slice_spec, threshold.change_threshold)

  for spec in metrics_specs:
    for aggregation_type, sub_keys in _create_sub_keys(spec).items():
      # Add thresholds for metrics computed in-graph.
      for metric_name, threshold in spec.thresholds.items():
        for key in _keys_for_metric(metric_name, spec, aggregation_type,
                                    sub_keys):
          add_threshold(key, None, threshold)
      for metric_name, per_slice_thresholds in spec.per_slice_thresholds.items(
      ):
        for key in _keys_for_metric(metric_name, spec, aggregation_type,
                                    sub_keys):
          for per_slice_threshold in per_slice_thresholds.thresholds:
            for slice_spec in per_slice_threshold.slicing_specs:
              add_threshold(key, slice_spec, per_slice_threshold.threshold)
      for metric_name, cross_slice_thresholds in (
          spec.cross_slice_thresholds.items()):
        for key in _keys_for_metric(metric_name, spec, aggregation_type,
                                    sub_keys):
          for cross_slice_threshold in cross_slice_thresholds.thresholds:
            for cross_slice_spec in cross_slice_threshold.cross_slicing_specs:
              add_threshold(key, cross_slice_spec,
                            cross_slice_threshold.threshold)

  # Add thresholds for post export metrics defined in MetricConfigs.
  for key, metric_config, _ in _keys_and_metrics_from_specs(metrics_specs):
    if metric_config.HasField('threshold'):
      add_threshold(key, None, metric_config.threshold)
    for per_slice_threshold in metric_config.per_slice_thresholds:
      for slice_spec in per_slice_threshold.slicing_specs:
        add_threshold(key, slice_spec, per_slice_threshold.threshold)
    for cross_slice_threshold in metric_config.cross_slice_thresholds:
      for cross_slice_spec in cross_slice_threshold.cross_slicing_specs:
        add_threshold(key, cross_slice_spec, cross_slice_threshold.threshold)

  return result


def to_computations(
    metrics_specs: List[config.MetricsSpec],
    eval_config: Optional[config.EvalConfig] = None,
    schema: Optional[schema_pb2.Schema] = None
) -> metric_types.MetricComputations:
  """Returns computations associated with given metrics specs."""
  computations = []

  #
  # Split into TF metrics and TFMA metrics
  #

  # Dict[Text, Type[tf.keras.metrics.Metric]]
  tf_metric_classes = {}  # class_name -> class
  # Dict[Text, Type[tf.keras.losses.Loss]]
  tf_loss_classes = {}  # class_name -> class
  # List[metric_types.MetricsSpec]
  tf_metrics_specs = []
  # Dict[Text, Type[metric_types.Metric]]
  tfma_metric_classes = metric_types.registered_metrics()  # class_name -> class
  # List[metric_types.MetricsSpec]
  tfma_metrics_specs = []
  #
  # Note: Lists are used instead of Dicts for the following items because
  # protos are are no hashable.
  #
  # List[List[_TFOrTFMAMetric]] (offsets align with metrics_specs).
  per_spec_metric_instances = []
  # List[List[_TFMetricOrLoss]] (offsets align with tf_metrics_specs).
  per_tf_spec_metric_instances = []
  # List[List[metric_types.Metric]]] (offsets align with tfma_metrics_specs).
  per_tfma_spec_metric_instances = []
  for spec in metrics_specs:
    tf_spec = config.MetricsSpec()
    tf_spec.CopyFrom(spec)
    del tf_spec.metrics[:]
    tfma_spec = config.MetricsSpec()
    tfma_spec.CopyFrom(spec)
    del tfma_spec.metrics[:]
    for metric in spec.metrics:
      if metric.class_name in tfma_metric_classes:
        tfma_spec.metrics.append(metric)
      elif not metric.module:
        tf_spec.metrics.append(metric)
      else:
        cls = getattr(importlib.import_module(metric.module), metric.class_name)
        if issubclass(cls, tf.keras.metrics.Metric):
          tf_metric_classes[metric.class_name] = cls
          tf_spec.metrics.append(metric)
        elif issubclass(cls, tf.keras.losses.Loss):
          tf_loss_classes[metric.class_name] = cls
          tf_spec.metrics.append(metric)
        else:
          tfma_metric_classes[metric.class_name] = cls
          tfma_spec.metrics.append(metric)

    metric_instances = []
    if tf_spec.metrics:
      tf_metrics_specs.append(tf_spec)
      tf_metric_instances = []
      for m in tf_spec.metrics:
        # To distinguish losses from metrics, losses are required to set the
        # module name.
        if m.module == _TF_LOSSES_MODULE:
          tf_metric_instances.append(_deserialize_tf_loss(m, tf_loss_classes))
        else:
          tf_metric_instances.append(
              _deserialize_tf_metric(m, tf_metric_classes))
      per_tf_spec_metric_instances.append(tf_metric_instances)
      metric_instances.extend(tf_metric_instances)
    if tfma_spec.metrics:
      tfma_metrics_specs.append(tfma_spec)
      tfma_metric_instances = [
          _deserialize_tfma_metric(m, tfma_metric_classes)
          for m in tfma_spec.metrics
      ]
      per_tfma_spec_metric_instances.append(tfma_metric_instances)
      metric_instances.extend(tfma_metric_instances)
    per_spec_metric_instances.append(metric_instances)

  # Process TF specs
  computations.extend(
      _process_tf_metrics_specs(tf_metrics_specs, per_tf_spec_metric_instances,
                                eval_config))

  # Process TFMA specs
  computations.extend(
      _process_tfma_metrics_specs(tfma_metrics_specs,
                                  per_tfma_spec_metric_instances, eval_config,
                                  schema))

  # Process macro averaging metrics (note that processing of TF and TFMA specs
  # were setup to create the binarized metrics that macro averaging depends on).
  for i, spec in enumerate(metrics_specs):
    for aggregation_type, sub_keys in _create_sub_keys(spec).items():
      if not (aggregation_type and (aggregation_type.macro_average or
                                    aggregation_type.weighted_macro_average)):
        continue
      class_weights = _class_weights(spec) or {}
      for model_name in spec.model_names or ['']:
        for output_name in spec.output_names or ['']:
          for sub_key in sub_keys:
            for metric in per_spec_metric_instances[i]:
              sub_keys = _macro_average_sub_keys(sub_key, class_weights)
              if aggregation_type.macro_average:
                computations.extend(
                    aggregation.macro_average(
                        metric.get_config()['name'],
                        sub_keys=sub_keys,
                        eval_config=eval_config,
                        model_name=model_name,
                        output_name=output_name,
                        sub_key=sub_key,
                        class_weights=class_weights))
              elif aggregation_type.weighted_macro_average:
                computations.extend(
                    aggregation.weighted_macro_average(
                        metric.get_config()['name'],
                        sub_keys=sub_keys,
                        eval_config=eval_config,
                        model_name=model_name,
                        output_name=output_name,
                        sub_key=sub_key,
                        class_weights=class_weights))

  return computations


def _process_tf_metrics_specs(
    tf_metrics_specs: List[config.MetricsSpec],
    per_tf_spec_metric_instances: List[List[_TFMetricOrLoss]],
    eval_config: config.EvalConfig) -> metric_types.MetricComputations:
  """Processes list of TF MetricsSpecs to create computations."""

  # Wrap args into structure that is hashable so we can track unique arg sets.
  class UniqueArgs(
      NamedTuple('UniqueArgs',
                 [('model_name', Text),
                  ('sub_key', Optional[metric_types.SubKey]),
                  ('aggregation_type', Optional[metric_types.AggregationType]),
                  ('class_weights', Tuple[Tuple[int, float], ...])])):
    pass

  def _create_private_tf_metrics(
      metrics: List[_TFMetricOrLoss]) -> List[_TFMetricOrLoss]:
    """Creates private versions of TF metrics."""
    result = []
    for m in metrics:
      if isinstance(m, tf.keras.metrics.Metric):
        result.append(_private_tf_metric(m))
      else:
        result.append(_private_tf_loss(m))
    return result

  #
  # Group TF metrics by the subkeys, models and outputs. This is done in reverse
  # because model and subkey processing is done outside of TF and so each unique
  # sub key combination needs to be run through a separate model instance. Note
  # that output_names are handled by the tf_metric_computation since all the
  # outputs are batch calculated in a single model evaluation call.
  #

  # UniqueArgs -> output_name -> [_TFMetricOrLoss]
  metrics_by_unique_args = collections.defaultdict(dict)
  for i, spec in enumerate(tf_metrics_specs):
    metrics = per_tf_spec_metric_instances[i]
    sub_keys_by_aggregation_type = _create_sub_keys(spec)
    # Keep track of metrics that can be shared between macro averaging and
    # binarization. For example, if macro averaging is being performed over 10
    # classes and 5 of the classes are also being binarized, then those 5
    # classes can be re-used by the macro averaging calculation. The remaining
    # 5 classes need to be added as private metrics since those classes were
    # not requested but are still needed for the macro averaging calculation.
    if None in sub_keys_by_aggregation_type:
      shared_sub_keys = set(sub_keys_by_aggregation_type[None])
    else:
      shared_sub_keys = set()
    for aggregation_type, sub_keys in sub_keys_by_aggregation_type.items():
      if aggregation_type:
        class_weights = tuple(sorted((_class_weights(spec) or {}).items()))
      else:
        class_weights = ()
      is_macro = (
          aggregation_type and (aggregation_type.macro_average or
                                aggregation_type.weighted_macro_average))
      for parent_sub_key in sub_keys:
        if is_macro:
          child_sub_keys = _macro_average_sub_keys(parent_sub_key,
                                                   _class_weights(spec))
        else:
          child_sub_keys = [parent_sub_key]
        for output_name in spec.output_names or ['']:
          for sub_key in child_sub_keys:
            if is_macro and sub_key not in shared_sub_keys:
              # Create private metrics for all non-shared metrics.
              instances = _create_private_tf_metrics(metrics)
            else:
              instances = metrics
            for model_name in spec.model_names or ['']:
              unique_args = UniqueArgs(
                  model_name, sub_key,
                  aggregation_type if not is_macro else None,
                  class_weights if not is_macro else ())
              if output_name not in metrics_by_unique_args[unique_args]:
                metrics_by_unique_args[unique_args][output_name] = []
              metrics_by_unique_args[unique_args][output_name].extend(instances)

  # Convert Unique args and outputs to calls to compute TF metrics
  result = []
  for args, metrics_by_output in metrics_by_unique_args.items():
    class_weights = dict(args.class_weights) if args.class_weights else None
    result.extend(
        tf_metric_wrapper.tf_metric_computations(
            metrics_by_output,
            eval_config=eval_config,
            model_name=args.model_name,
            sub_key=args.sub_key,
            aggregation_type=args.aggregation_type,
            class_weights=class_weights))
  return result


def _process_tfma_metrics_specs(
    tfma_metrics_specs: List[config.MetricsSpec],
    per_tfma_spec_metric_instances: List[List[metric_types.Metric]],
    eval_config: config.EvalConfig,
    schema: Optional[schema_pb2.Schema]) -> metric_types.MetricComputations:
  """Processes list of TFMA MetricsSpecs to create computations."""

  #
  # Computations are per metric, so separate by metrics and the specs associated
  # with them.
  #

  # Dict[bytes, List[config.MetricSpec]] (hash(MetricConfig) -> [MetricSpec])
  tfma_specs_by_metric_config = {}
  # Dict[bytes, metric_types.Metric] (hash(MetricConfig) -> Metric)
  hashed_metrics = {}
  for i, spec in enumerate(tfma_metrics_specs):
    for metric_config, metric in zip(spec.metrics,
                                     per_tfma_spec_metric_instances[i]):
      # Note that hashing by SerializeToString() is only safe if used within the
      # same process.
      config_hash = metric_config.SerializeToString()
      if config_hash not in tfma_specs_by_metric_config:
        hashed_metrics[config_hash] = metric
        tfma_specs_by_metric_config[config_hash] = []
      tfma_specs_by_metric_config[config_hash].append(spec)

  #
  # Create computations for each metric.
  #

  result = []
  for config_hash, specs in tfma_specs_by_metric_config.items():
    metric = hashed_metrics[config_hash]
    for spec in specs:
      sub_keys_by_aggregation_type = _create_sub_keys(spec)
      # Keep track of sub-keys that can be shared between macro averaging and
      # binarization. For example, if macro averaging is being performed over
      # 10 classes and 5 of the classes are also being binarized, then those 5
      # classes can be re-used by the macro averaging calculation. The
      # remaining 5 classes need to be added as private metrics since those
      # classes were not requested but are still needed for the macro
      # averaging calculation.
      if None in sub_keys_by_aggregation_type:
        shared_sub_keys = set(sub_keys_by_aggregation_type[None])
      else:
        shared_sub_keys = set()
      for aggregation_type, sub_keys in sub_keys_by_aggregation_type.items():
        class_weights = _class_weights(spec) if aggregation_type else None
        is_macro = (
            aggregation_type and (aggregation_type.macro_average or
                                  aggregation_type.weighted_macro_average))
        if is_macro:
          updated_sub_keys = []
          for sub_key in sub_keys:
            for key in _macro_average_sub_keys(sub_key, class_weights):
              if key not in shared_sub_keys:
                updated_sub_keys.append(key)
          if not updated_sub_keys:
            continue
          aggregation_type = None
          class_weights = None
          sub_keys = updated_sub_keys
          instance = _private_tfma_metric(metric)
        else:
          instance = metric
        result.extend(
            instance.computations(
                eval_config=eval_config,
                schema=schema,
                model_names=list(spec.model_names) or [''],
                output_names=list(spec.output_names) or [''],
                sub_keys=sub_keys,
                aggregation_type=aggregation_type,
                class_weights=class_weights if class_weights else None,
                query_key=spec.query_key))
  return result


def _create_sub_keys(
    spec: config.MetricsSpec
) -> Dict[Optional[metric_types.AggregationType],
          List[Optional[metric_types.SubKey]]]:
  """Creates sub keys per aggregation type."""
  result = {}
  if spec.HasField('binarize'):
    sub_keys = []
    if spec.binarize.class_ids.values:
      for v in spec.binarize.class_ids.values:
        sub_keys.append(metric_types.SubKey(class_id=v))
    if spec.binarize.k_list.values:
      for v in spec.binarize.k_list.values:
        sub_keys.append(metric_types.SubKey(k=v))
    if spec.binarize.top_k_list.values:
      for v in spec.binarize.top_k_list.values:
        sub_keys.append(metric_types.SubKey(top_k=v))
    if sub_keys:
      result[None] = sub_keys
  if spec.HasField('aggregate'):
    sub_keys = []
    for top_k in spec.aggregate.top_k_list.values:
      sub_keys.append(metric_types.SubKey(top_k=top_k))
    if not sub_keys:
      sub_keys = [None]
    result[_aggregation_type(spec)] = sub_keys
  return result if result else {None: [None]}


def _macro_average_sub_keys(
    sub_key: Optional[metric_types.SubKey],
    class_weights: Dict[int, float]) -> Iterable[metric_types.SubKey]:
  """Returns sub-keys required in order to compute macro average sub-key.

  Args:
    sub_key: SubKey associated with macro_average or weighted_macro_average.
    class_weights: Class weights associated with sub-key.

  Raises:
    ValueError: If invalid sub-key passed or class weights required but not
      passed.
  """
  if not sub_key:
    if not class_weights:
      raise ValueError(
          'class_weights are required in order to compute macro average over '
          'all classes: sub_key={}, class_weights={}'.format(
              sub_key, class_weights))
    return [metric_types.SubKey(class_id=i) for i in class_weights.keys()]
  elif sub_key.top_k:
    return [metric_types.SubKey(k=i + 1) for i in range(sub_key.top_k)]
  else:
    raise ValueError('invalid sub_key for performing macro averaging: '
                     'sub_key={}'.format(sub_key))


def _aggregation_type(
    spec: config.MetricsSpec) -> Optional[metric_types.AggregationType]:
  """Returns AggregationType associated with AggregationOptions at offset."""
  if spec.aggregate.micro_average:
    return metric_types.AggregationType(micro_average=True)
  if spec.aggregate.macro_average:
    return metric_types.AggregationType(macro_average=True)
  if spec.aggregate.weighted_macro_average:
    return metric_types.AggregationType(weighted_macro_average=True)
  return None


def _class_weights(spec: config.MetricsSpec) -> Optional[Dict[int, float]]:
  """Returns class weights associated with AggregationOptions at offset."""
  if spec.aggregate.HasField('top_k_list'):
    if spec.aggregate.class_weights:
      raise ValueError('class_weights are not supported when top_k_list used: '
                       'spec={}'.format(spec))
    return None
  return dict(spec.aggregate.class_weights) or None


def _metric_config(cfg: Text) -> Dict[Text, Any]:
  """Returns deserializable metric config from JSON string."""
  if not cfg:
    json_cfg = '{}'
  elif cfg[0] != '{':
    json_cfg = '{' + cfg + '}'
  else:
    json_cfg = cfg
  return json.loads(json_cfg)


def _maybe_add_name_to_config(cfg: Dict[Text, Any],
                              class_name: Text) -> Dict[Text, Any]:
  """Adds default name field to metric config if not present."""
  if 'name' not in cfg:
    # Use snake_case version of class name as default name.
    intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', class_name)
    cfg['name'] = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
  return cfg


def _tf_class_and_config(
    metric_config: config.MetricConfig) -> Tuple[Text, Dict[Text, Any]]:
  """Returns the tensorflow class and config associated with metric_config."""
  cls_name = metric_config.class_name
  cfg = _metric_config(metric_config.config)
  for name, defaults in _TF_CONFIG_DEFAULTS.items():
    if name == cls_name:
      if 'class_name' in defaults:
        cls_name = defaults['class_name']
      if 'config' in defaults:
        tmp = cfg
        cfg = copy.copy(defaults['config'])
        cfg.update(tmp)
      break

  # The same metric type may be used for different keys when multi-class metrics
  # are used (e.g. AUC for class0, # class1, etc). TF tries to generate unique
  # metric names even though these metrics are already unique within a
  # MetricKey. To workaroudn this issue, if a name is not set, then add a
  # default name ourselves.
  return cls_name, _maybe_add_name_to_config(cfg, cls_name)


def _serialize_tf_metric(
    metric: tf.keras.metrics.Metric) -> config.MetricConfig:
  """Serializes TF metric."""
  cfg = metric_util.serialize_metric(metric)
  return config.MetricConfig(
      class_name=cfg['class_name'],
      config=json.dumps(cfg['config'], sort_keys=True))


def _deserialize_tf_metric(
    metric_config: config.MetricConfig,
    custom_objects: Dict[Text, Type[tf.keras.metrics.Metric]]
) -> tf.keras.metrics.Metric:
  """Deserializes a tf.keras.metrics metric."""
  cls_name, cfg = _tf_class_and_config(metric_config)
  with tf.keras.utils.custom_object_scope(custom_objects):
    return tf.keras.metrics.deserialize({'class_name': cls_name, 'config': cfg})


def _private_tf_metric(
    metric: tf.keras.metrics.Metric) -> tf.keras.metrics.Metric:
  """Creates a private version of given metric."""
  cfg = metric_util.serialize_metric(metric)
  if not cfg['config']['name'].startswith('_'):
    cfg['config']['name'] = '_' + cfg['config']['name']
  with tf.keras.utils.custom_object_scope(
      {metric.__class__.__name__: metric.__class__}):
    return tf.keras.metrics.deserialize(cfg)


def _serialize_tf_loss(loss: tf.keras.losses.Loss) -> config.MetricConfig:
  """Serializes TF loss."""
  cfg = metric_util.serialize_loss(loss)
  return config.MetricConfig(
      class_name=cfg['class_name'],
      module=loss.__class__.__module__,
      config=json.dumps(cfg['config'], sort_keys=True))


def _deserialize_tf_loss(
    metric_config: config.MetricConfig,
    custom_objects: Dict[Text,
                         Type[tf.keras.losses.Loss]]) -> tf.keras.losses.Loss:
  """Deserializes a tf.keras.loss metric."""
  cls_name, cfg = _tf_class_and_config(metric_config)
  with tf.keras.utils.custom_object_scope(custom_objects):
    return tf.keras.losses.deserialize({'class_name': cls_name, 'config': cfg})


def _private_tf_loss(loss: tf.keras.losses.Loss) -> tf.keras.losses.Loss:
  """Creates a private version of given loss."""
  cfg = metric_util.serialize_loss(loss)
  if not cfg['config']['name'].startswith('_'):
    cfg['config']['name'] = '_' + cfg['config']['name']
  with tf.keras.utils.custom_object_scope(
      {loss.__class__.__name__: loss.__class__}):
    return tf.keras.losses.deserialize(cfg)


def _serialize_tfma_metric(metric: metric_types.Metric) -> config.MetricConfig:
  """Serializes TFMA metric."""
  # This implementation is identical to _serialize_tf_metric, but keeping two
  # implementations for symmetry with deserialize where separate implementations
  # are required (and to be consistent with the keras implementation).
  cfg = tf.keras.utils.serialize_keras_object(metric)
  return config.MetricConfig(
      class_name=cfg['class_name'],
      config=json.dumps(cfg['config'], sort_keys=True))


def _deserialize_tfma_metric(
    metric_config: config.MetricConfig,
    custom_objects: Dict[Text,
                         Type[metric_types.Metric]]) -> metric_types.Metric:
  """Deserializes a tfma.metrics metric."""
  with tf.keras.utils.custom_object_scope(custom_objects):
    return tf.keras.utils.deserialize_keras_object({
        'class_name': metric_config.class_name,
        'config': _metric_config(metric_config.config)
    })


def _private_tfma_metric(metric: metric_types.Metric) -> metric_types.Metric:
  """Creates a private version of given metric."""
  cfg = tf.keras.utils.serialize_keras_object(metric)
  if not cfg['config']['name'].startswith('_'):
    cfg['config']['name'] = '_' + cfg['config']['name']
  with tf.keras.utils.custom_object_scope(
      {metric.__class__.__name__: metric.__class__}):
    return tf.keras.utils.deserialize_keras_object(cfg)
