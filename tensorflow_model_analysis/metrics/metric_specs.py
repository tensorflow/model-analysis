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

import importlib
import json
from typing import Dict, List, Optional, Text, Type, Union
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import types
from tensorflow_model_analysis.metrics import auc_plot
from tensorflow_model_analysis.metrics import calibration
from tensorflow_model_analysis.metrics import calibration_plot
from tensorflow_model_analysis.metrics import example_count
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import multi_class_confusion_matrix_at_thresholds
from tensorflow_model_analysis.metrics import tf_metric_wrapper
from tensorflow_model_analysis.metrics import weighted_example_count

_TFOrTFMAMetric = Union[tf.keras.metrics.Metric, metric_types.Metric]


def specs_from_metrics(
    metrics: Union[List[_TFOrTFMAMetric], Dict[Text, List[_TFOrTFMAMetric]]],
    model_names: Optional[List[Text]] = None,
    output_names: Optional[List[Text]] = None,
    class_ids: Optional[List[int]] = None,
    k_list: Optional[List[int]] = None,
    top_k_list: Optional[List[int]] = None,
    query_key: Optional[Text] = None,
    include_example_count: Optional[bool] = None,
    include_weighted_example_count: Optional[bool] = None
) -> List[config.MetricsSpec]:
  """Returns specs from tf.keras.metrics.Metric or tfma.metrics.Metric classes.

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
    metrics: List of tf.keras.metrics.Metric or tfma.metrics.Metric. For
      multi-output models a dict of dicts may be passed where the first dict is
      indexed by the output_name.
    model_names: Optional model names (if multi-model evaluation).
    output_names: Optional output names (if multi-output models). If the metrics
      are a dict this should not be set.
    class_ids: Optional class IDs to computes metrics for particular classes of
      a multi-class model. If output_names are provided, all outputs are assumed
      to use the same class IDs.
    k_list: Optional list of k values to compute metrics for the kth predicted
      values in a multi-class model prediction. If output_names are provided,
      all outputs are assumed to use the same k value.
    top_k_list: Optional list of top_k values to compute metrics for the top k
      predicted values in a multi-class model prediction. If output_names are
      provided, all outputs are assumed to use the same top_k value. Metrics and
      plots will be based on treating each predicted value in the top_k as
      though they were separate predictions.
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
              class_ids=class_ids,
              k_list=k_list,
              top_k_list=top_k_list,
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
    else:
      metric_configs.append(_serialize_tfma_metric(metric))
  if class_ids:
    specs.append(
        config.MetricsSpec(
            metrics=metric_configs,
            model_names=model_names,
            output_names=output_names,
            binarize=config.BinarizationOptions(class_ids=class_ids),
            query_key=query_key))
  if k_list:
    specs.append(
        config.MetricsSpec(
            metrics=metric_configs,
            model_names=model_names,
            output_names=output_names,
            binarize=config.BinarizationOptions(k_list=k_list),
            query_key=query_key))
  if top_k_list:
    specs.append(
        config.MetricsSpec(
            metrics=metric_configs,
            model_names=model_names,
            output_names=output_names,
            binarize=config.BinarizationOptions(top_k_list=top_k_list),
            query_key=query_key))
  if not class_ids and not k_list and not top_k_list:
    specs.append(
        config.MetricsSpec(
            metrics=metric_configs,
            model_names=model_names,
            output_names=output_names,
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
    specs.append(config.MetricsSpec(metrics=[metric_config]))
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
    loss_functions: Optional[List[tf.keras.metrics.Metric]] = None,
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
    class_ids: Optional[List[int]] = None,
    k_list: Optional[List[int]] = None,
    top_k_list: Optional[List[int]] = None,
    include_loss: bool = True) -> List[config.MetricsSpec]:
  """Returns default metric specs for binary classification problems.

  Args:
    model_names: Optional model names (if multi-model evaluation).
    output_names: Optional list of output names (if multi-output model).
    class_ids: Optional class IDs to compute metrics for particular classes in a
      multi-class model. If output_names are provided, all outputs are assumed
      to use the same class IDs.
    k_list: Optional list of k values to compute metrics for the kth predicted
      values of a multi-class model prediction. If output_names are provided,
      all outputs are assumed to use the same k value.
    top_k_list: Optional list of top_k values to compute metrics for the top k
      predicted values in a multi-class model prediction. If output_names are
      provided, all outputs are assumed to use the same top_k value. Metrics and
      plots will be based on treating each predicted value in the top_k as
      though they were separate predictions.
    include_loss: True to include loss.
  """

  metrics = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='auc_pr', curve='PR'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      calibration.MeanLabel(name='mean_label'),
      calibration.MeanPrediction(name='mean_prediction'),
      calibration.Calibration(name='calibration'),
      auc_plot.AUCPlot(name='auc_plot'),
      calibration_plot.CalibrationPlot(name='calibration_plot')
  ]
  if include_loss:
    metrics.append(tf.keras.metrics.BinaryCrossentropy(name='loss'))

  return specs_from_metrics(
      metrics,
      model_names=model_names,
      output_names=output_names,
      class_ids=class_ids,
      k_list=k_list,
      top_k_list=top_k_list)


def default_multi_class_classification_specs(
    model_names: Optional[List[Text]] = None,
    output_names: Optional[List[Text]] = None,
    top_k_list: Optional[List[int]] = None,
    class_ids: Optional[List[int]] = None,
    k_list: Optional[List[int]] = None,
    sparse: bool = True) -> config.MetricsSpec:
  """Returns default metric specs for multi-class classification problems.

  Args:
    model_names: Optional model names if multi-model evaluation.
    output_names: Optional list of output names (if multi-output model).
    top_k_list: Optional list of top-k values to compute metrics for the top k
      predicted values (e.g. precision@k, recall@k). If output_names are
      provided, all outputs are assumed to use the same top-k value. By default
      only typical multi-class metrics such as precision@k/recall@k are
      computed. If plots, mean_label, etc are also desired then
      default_binary_classification_metrics should be called direclty.
    class_ids: Optional class IDs to compute binary classification metrics for
      using one vs rest. If output_names are provided, all outputs are kassumed
      to use same class IDs.
    k_list: Optional list of k values to compute binary classification metrics
      based on the kth predicted value. If output_names are provided, all
      outputs are assumed to use the same k value.
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
      multi_class_confusion_matrix_at_thresholds
      .MultiClassConfusionMatrixAtThresholds(
          name='multi_class_confusion_matrix_at_thresholds'))
  if top_k_list:
    for top_k in top_k_list:
      metrics.extend([
          tf.keras.metrics.Precision(name='precision', top_k=top_k),
          tf.keras.metrics.Recall(name='recall', top_k=top_k)
      ])

  multi_class_metrics = specs_from_metrics(
      metrics, model_names=model_names, output_names=output_names)
  if class_ids or k_list:
    multi_class_metrics.extend(
        default_binary_classification_specs(model_names, output_names,
                                            class_ids, k_list))
  return multi_class_metrics


def to_computations(
    metrics_specs: List[config.MetricsSpec],
    eval_config: Optional[config.EvalConfig] = None,
    model_loaders: Optional[Dict[Text, types.ModelLoader]] = None
) -> metric_types.MetricComputations:
  """Returns computations associated with given metrics specs."""
  computations = []

  #
  # Split into TF metrics and TFMA metrics
  #

  # Dict[Text, Type[tf.keras.metrics.Metric]]
  tf_metric_classes = {}  # class_name -> class
  # List[metric_types.MetricsSpec]
  tf_metrics_specs = []
  # Dict[Text, Type[metric_types.Metric]]
  tfma_metric_classes = metric_types.registered_metrics()  # class_name -> class
  # List[metric_types.MetricsSpec]
  tfma_metrics_specs = []
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
        cls = getattr(
            importlib.import_module(metric.module_name), metric.class_name)
        if isinstance(metric, tf.keras.metrics.Metric):
          tf_metric_classes[metric.class_name] = cls
          tf_spec.metrics.append(metric)
        else:
          tfma_metric_classes[metric.class_name] = cls
          tfma_spec.metrics.append(metric)
    if tf_spec.metrics:
      tf_metrics_specs.append(tf_spec)
    if tfma_spec.metrics:
      tfma_metrics_specs.append(tfma_spec)

  #
  # Group TF metrics by the subkeys, models and outputs. This is done in reverse
  # because model and subkey processing is done outside of TF and so each unique
  # sub key combination needs to be run through a separate model instance. Note
  # that output_names are handled by the tf_metric_computation since all the
  # outputs are batch calculated in a single model evaluation call.
  #

  # Dict[metric_types.SubKey, Dict[Text, List[config.MetricSpec]]
  tf_specs_by_subkey = {}  # SubKey -> model_name -> [MetricSpec]
  for spec in tf_metrics_specs:
    sub_keys = _create_sub_keys(spec)
    if not sub_keys:
      sub_keys = [None]
    for sub_key in sub_keys:
      if sub_key not in tf_specs_by_subkey:
        tf_specs_by_subkey[sub_key] = {}
      # Dict[Text, List[config.MetricSpec]]
      tf_specs_by_model = tf_specs_by_subkey[sub_key]  # name -> [ModelSpec]
      model_names = spec.model_names
      if not model_names:
        model_names = ['']  # '' is name used when only one model is used
      for model_name in model_names:
        if model_name not in tf_specs_by_model:
          tf_specs_by_model[model_name] = []
        tf_specs_by_model[model_name].append(spec)
  for sub_key, specs_by_model in tf_specs_by_subkey.items():
    for model_name, specs in specs_by_model.items():
      metrics_by_output = {}
      for spec in specs:
        metrics = [
            _deserialize_tf_metric(m, tf_metric_classes) for m in spec.metrics
        ]
        if spec.output_names:
          for output_name in spec.output_names:
            if output_name not in metrics_by_output:
              metrics_by_output[output_name] = []
            metrics_by_output[output_name].extend(metrics)
        else:
          if '' not in metrics_by_output:
            metrics_by_output[''] = []  # '' is name used when only one output
          metrics_by_output[''].extend(metrics)
      model_loader = None
      if model_loaders and model_name in model_loaders:
        model_loader = model_loaders[model_name]
      computations.extend(
          tf_metric_wrapper.tf_metric_computations(
              metrics_by_output,
              eval_config=eval_config,
              model_name=model_name,
              sub_key=sub_key,
              model_loader=model_loader))

  #
  # Group TFMA metric specs by the metric classes
  #

  # Dict[bytes, List[config.MetricSpec]]
  tfma_specs_by_metric_config = {}  # hash(MetricConfig) -> [MetricSpec]
  # Dict[bytes, config.MetricConfig]
  hashed_metric_configs = {}  # hash(MetricConfig) -> MetricConfig
  for spec in tfma_metrics_specs:
    for metric_config in spec.metrics:
      # Note that hashing by SerializeToString() is only safe if used within the
      # same process.
      config_hash = metric_config.SerializeToString()
      if config_hash not in tfma_specs_by_metric_config:
        hashed_metric_configs[config_hash] = metric_config
        tfma_specs_by_metric_config[config_hash] = []
      tfma_specs_by_metric_config[config_hash].append(spec)
  for config_hash, specs in tfma_specs_by_metric_config.items():
    metric = _deserialize_tfma_metric(hashed_metric_configs[config_hash],
                                      tfma_metric_classes)
    for spec in specs:
      sub_keys = _create_sub_keys(spec)
      computations.extend(
          metric.computations(
              eval_config=eval_config,
              model_names=spec.model_names if spec.model_names else [''],
              output_names=spec.output_names if spec.output_names else [''],
              sub_keys=sub_keys,
              query_key=spec.query_key))
  return computations


def _create_sub_keys(
    spec: config.MetricsSpec) -> Optional[List[metric_types.SubKey]]:
  """Creates subkeys associated with spec."""
  sub_keys = None
  if spec.HasField('binarize'):
    sub_keys = []
    if spec.binarize.class_ids:
      for v in spec.binarize.class_ids:
        sub_keys.append(metric_types.SubKey(class_id=v))
    if spec.binarize.k_list:
      for v in spec.binarize.k_list:
        sub_keys.append(metric_types.SubKey(k=v))
    if spec.binarize.top_k_list:
      for v in spec.binarize.top_k_list:
        sub_keys.append(metric_types.SubKey(top_k=v))
  return sub_keys


def _metric_config(cfg: Text) -> Text:
  """Returns JSON deserializable metric config from string."""
  if not cfg:
    return '{}'
  elif cfg[0] != '{':
    return '{' + cfg + '}'
  else:
    return cfg


def _serialize_tf_metric(
    metric: tf.keras.metrics.Metric) -> config.MetricConfig:
  """Serializes TF metric."""
  cfg = tf.keras.metrics.serialize(metric)
  return config.MetricConfig(
      class_name=cfg['class_name'], config=json.dumps(cfg['config']))


def _deserialize_tf_metric(
    metric_config: config.MetricConfig,
    custom_objects: Dict[Text, Type[tf.keras.metrics.Metric]]
) -> tf.keras.metrics.Metric:
  """Deserializes a tf.keras.metrics metric."""
  with tf.keras.utils.custom_object_scope(custom_objects):
    return tf.keras.metrics.deserialize({
        'class_name': metric_config.class_name,
        'config': json.loads(_metric_config(metric_config.config))
    })


def _serialize_tfma_metric(metric: metric_types.Metric) -> config.MetricConfig:
  """Serializes TFMA metric."""
  # This implementation is identical to _serialize_tf_metric, but keeping two
  # implementations for symmetry with deserialize where separate implementations
  # are required (and to be consistent with the keras implementation).
  cfg = tf.keras.utils.serialize_keras_object(metric)
  return config.MetricConfig(
      class_name=cfg['class_name'], config=json.dumps(cfg['config']))


def _deserialize_tfma_metric(
    metric_config: config.MetricConfig,
    custom_objects: Dict[Text,
                         Type[metric_types.Metric]]) -> metric_types.Metric:
  """Deserializes a tfma.metrics metric."""
  with tf.keras.utils.custom_object_scope(custom_objects):
    return tf.keras.utils.deserialize_keras_object({
        'class_name': metric_config.class_name,
        'config': json.loads(_metric_config(metric_config.config))
    })
