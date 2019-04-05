# Copyright 2018 Google LLC
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
"""Library containing helpers for adding post export metrics for evaluation.

These post export metrics can be included in the add_post_export_metrics
parameter of Evaluate to compute them.
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import abc
# Standard Imports
import numpy as np
from six import with_metaclass
import tensorflow as tf
from tensorflow_model_analysis import types
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.post_export_metrics import metrics
from tensorflow_model_analysis.proto import metrics_for_slice_pb2 as metrics_pb2
from typing import Any, Dict, List, Optional, Text, Tuple, Type

from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.ops import metrics_impl


# TODO(b/111754250): revisit it and determine whether to simplify the 4-level
# deep nesting of functions
def _export(name: Text):
  """Decorator for exporting a _PostExportMetric class.

  The net effect of the decorator is to create a function with the given name
  that can be called to create a callback for use with add_metrics_callbacks.

  Example usage:
    @_export('my_metric')
    class _MyMetric(_PostExportMetric):

      def check_compatibility(self, features_dict: types.TensorTypeMaybeDict,
                              predictions_dict: types.TensorTypeMaybeDict,
                              labels_dict: types.TensorTypeMaybeDict) -> None:
        # check compatibility needed for the metric here.

      def get_metric_ops(self, features_dict: types.TensorTypeMaybeDict,
                         predictions_dict: types.TensorTypeMaybeDict,
                         labels_dict: types.TensorTypeMaybeDict
                        ) -> Dict[bytes, Tuple[types.TensorType,
                        types.TensorType]]:
        # metric computation here
        return {'my_metric_key': (value_op, update_op)}

    where callers can call my_metric() as a post export metric passed to
    `add_metrics_callbacks` in tfma APIs when needed.

  Args:
    name: Name of the exported function.

  Returns:
    Decorator for exporting a post export metric class.
  """

  def _actual_export(cls: Type[Any]):
    """This is the actual decorator."""

    def fn(*args, **kwargs):
      """This is the function that the user calls."""

      def callback(features_dict: types.TensorTypeMaybeDict,
                   predictions_dict: types.TensorTypeMaybeDict,
                   labels_dict: types.TensorTypeMaybeDict):
        """This actual callback that goes into add_metrics_callbacks."""
        metric = cls(*args, **kwargs)
        metric.check_compatibility(features_dict, predictions_dict, labels_dict)
        metric_ops = {}
        for key, value in (metric.get_metric_ops(features_dict,
                                                 predictions_dict,
                                                 labels_dict).items()):
          metric_ops[key] = value
        return metric_ops

      # We store the metric's export name in the .name property of the callback.
      callback.name = name
      callback.populate_stats_and_pop = cls(*args,
                                            **kwargs).populate_stats_and_pop
      callback.populate_plots_and_pop = cls(*args,
                                            **kwargs).populate_plots_and_pop
      return callback

    globals()[name] = fn
    return cls

  return _actual_export


# This must be a tuple to avoid mutation (see b/129368983).
DEFAULT_KEY_PREFERENCE = (
    prediction_keys.PredictionKeys.LOGISTIC,
    prediction_keys.PredictionKeys.PREDICTIONS,
    prediction_keys.PredictionKeys.PROBABILITIES,
    prediction_keys.PredictionKeys.LOGITS,
)


def _get_target_tensor(maybe_dict: types.TensorTypeMaybeDict,
                       key_precedence: List[Text]) -> types.TensorType:
  """Returns Tensor for prediction or labels dicts.

  Args:
    maybe_dict: Tensor or dictionary of tensors within which to find the target.
    key_precedence: One or more keys to search for--we will return the first
      tensor found.

  Returns:
    Predictions tensor, or None if none of the expected keys are found in
    the predictions_dict.
  """
  if types.is_tensor(maybe_dict):
    return maybe_dict

  for key in key_precedence:
    ref_tensor = maybe_dict.get(key)
    if ref_tensor is not None:
      return ref_tensor

  return None


def _check_weight_present(features_dict: types.TensorTypeMaybeDict,
                          example_weight_key: Optional[Text] = None):
  """Raise ValueError if the example weight is not present."""
  if (example_weight_key is not None and
      example_weight_key not in features_dict):
    raise ValueError('example weight key %s not found in features_dict. '
                     'features were: %s' %
                     (example_weight_key, features_dict.keys()))


def _populate_to_auc_bounded_value_and_pop(
    combined_metrics: Dict[Text, Any],
    output_metrics: Dict[Text, metrics_pb2.MetricValue],
    metric_key: Text) -> None:
  """Converts the given metric to bounded_value type in dict `output_metrics`.

  The metric to be converted should be in the dict `combined_metrics` with key
  as `metric_key`. The `combined_metrics` should also contain
  metric_keys.lower_bound_key(metric_key) and
  metric_keys.upper_bound_key(metric_key) which store the lower_bound and
  upper_bound of that metric. The result will be stored as bounded_value type in
  dict `output_metrics`. After the conversion, the metric will be poped out from
  the `combined_metrics`.

  Args:
    combined_metrics: The dict containing raw TFMA metrics.
    output_metrics: The dict where we convert the metrics to.
    metric_key: The key in the dict `metrics` for extracting the metric value.
  """
  riemann_sum_lower_bound = combined_metrics.pop(
      metric_keys.lower_bound_key(metric_key))
  if isinstance(riemann_sum_lower_bound, types.ValueWithConfidenceInterval):
    riemann_sum_lower_bound = riemann_sum_lower_bound.unsampled_value
  output_metrics[metric_key].bounded_value.lower_bound.value = (
      riemann_sum_lower_bound)
  riemann_sum_upper_bound = combined_metrics.pop(
      metric_keys.upper_bound_key(metric_key))
  if isinstance(riemann_sum_upper_bound, types.ValueWithConfidenceInterval):
    riemann_sum_upper_bound = riemann_sum_upper_bound.unsampled_value
  output_metrics[metric_key].bounded_value.upper_bound.value = (
      riemann_sum_upper_bound)
  output_metrics[metric_key].bounded_value.methodology = (
      metrics_pb2.BoundedValue.RIEMANN_SUM)

  value = combined_metrics.pop(metric_key)
  if isinstance(value, types.ValueWithConfidenceInterval):
    # Currently taking the computed mean value, conserving legacy functionality.
    # TODO(raz): Need to determine how best to handle confidence interval in
    # this case.
    value = value.value
  output_metrics[metric_key].bounded_value.value.value = value


def _additional_prediction_keys(keys: List[Text],
                                metric_tag: Text,
                                tensor_index: Optional[int] = None
                               ) -> List[Text]:
  """Returns a list of additional keys to try given a metric tag and index.

  If a metric_tag was given then we also search for keys prefixed by the
  metric_tag. In most cases the metric_tag is the head name and
  tf.contrib.estimator.multi_head prefixes the predictions by the head. If
  tensor_index was also provided then we also search under the tag stripped of
  the index. In this case the tag has the form <head_name>_<tensor_index>.

  For example, given the following setup:

    head1 = tf.contrib.estimator.multi_class_head(n_classes=3, name='head1')
    head2 = tf.contrib.estimator.binary_classification_head(name='head2')
    head = tf.contrib.estimator.multi_head([head1, head2])
    ...

  The prediction keys will be under head1/logistic and head2/logistic.

  If the default prediction key search was ['logistic', 'probabilities'] and the
  metric_tag was set to 'head1' to tag post export metrics for head1, then
  additional keys will be searched for under 'head1/logistic' and
  'head1/probabilities'. If a tensor index was also provided to binarize a
  multi-class output using index '3' as the positive class with a metric_tag of
  'head1_3' to distinguish it from other binarized post export metrics, then in
  addition to searching under the keys prefixed by the metric tag (e.g.
  'head1_3/logistic', 'head1_3/probablities'), a search will also be done under
  the tag stripped of the index (e.g. 'head1/logistic', 'head1/probablitites').

  Args:
    keys: Target prediction keys.
    metric_tag: Metric tag.
    tensor_index: Optional index to specify positive class.
  """
  additional_keys = []
  for key in keys:
    if tensor_index:
      suffix = '_%d' % tensor_index
      if metric_tag.endswith(suffix):
        additional_keys.append('%s/%s' % (metric_tag[:-len(suffix)], key))
    additional_keys.append('%s/%s' % (metric_tag, key))
  return additional_keys


class _PostExportMetric(with_metaclass(abc.ABCMeta, object)):
  """Abstract base class for post export metrics."""

  def __init__(self,
               target_prediction_keys: Optional[List[Text]] = None,
               labels_key: Optional[Text] = None,
               metric_tag: Optional[Text] = None,
               tensor_index: Optional[int] = None):
    """Common init of _PostExportMetrics.

    Args:
      target_prediction_keys: Optional acceptable keys in predictions_dict in
        descending order of precedence.
      labels_key: Optionally, the key from labels_dict to use.
      metric_tag: If provided, a custom metric tag. Only necessary to
        disambiguate instances of the same metric on different predictions or
        for readability concerns in tool output.
      tensor_index: Optional index to specify positive class.
    """
    self._target_prediction_keys = (
        target_prediction_keys or list(DEFAULT_KEY_PREFERENCE))
    self._tensor_index = tensor_index
    self._labels_key = labels_key
    if target_prediction_keys:
      self._metric_tag = target_prediction_keys[0]
    if metric_tag:
      # Specified metric tag takes priority over target_prediction_key if
      # defined.
      self._metric_tag = metric_tag
      self._target_prediction_keys.extend(
          _additional_prediction_keys(self._target_prediction_keys, metric_tag,
                                      self._tensor_index))

  def _select_class(self, predictions_tensor, labels_tensor):
    """Gets predictions and labels for the class at index self._tensor_index."""

    def make_multi_hot_labels():
      """Converts class index labels to mutli-hot vector."""
      tf.logging.info(
          'Labels has unknown static shape in dimension 1, indicating a class '
          'index tensor. '
          'Trying to transform labels into one_hot labels for analysis.')
      # Ensure that predictions has the appropriate rank and that the number of
      # classes in predictions > 1.
      predictions_tensor.shape.assert_has_rank(2)
      assert_op = tf.Assert(
          tf.greater_equal(tf.shape(predictions_tensor)[1], 1),
          [predictions_tensor])
      with tf.control_dependencies([assert_op]):
        # One-hot vector for each class index in labels.
        # Result has shape [batch_size, max_num_classes_in_batch, depth]
        one_hots_per_class = tf.one_hot(
            indices=labels_tensor,
            depth=tf.shape(predictions_tensor)[1],
            axis=-1)
        # Sum one-hot vectors to make a multi-hot vector representing all
        # classes.
        return tf.reduce_sum(one_hots_per_class, axis=1)

    labels_tensor.shape.assert_has_rank(2)
    if (labels_tensor.shape[1].value is None or
        labels_tensor.shape[1].value == 1 and self._tensor_index is not None):
      labels_tensor = make_multi_hot_labels()

    assert_op = tf.Assert(
        tf.reduce_all(
            tf.equal(tf.shape(predictions_tensor), tf.shape(labels_tensor))),
        [predictions_tensor, labels_tensor])
    with tf.control_dependencies([assert_op]):
      predictions_for_class = predictions_tensor[:, self._tensor_index]
      labels_for_class = tf.cast(labels_tensor[:, self._tensor_index],
                                 tf.float32)

    return predictions_for_class, labels_for_class

  def _get_labels_and_predictions(self,
                                  predictions_dict: types.TensorTypeMaybeDict,
                                  labels_dict: types.TensorTypeMaybeDict
                                 ) -> Tuple[Any, Any]:
    """Raise TypeError if the predictions and labels cannot be understood."""
    predictions_tensor = _get_target_tensor(predictions_dict,
                                            self._target_prediction_keys)
    if predictions_tensor is None:
      raise KeyError('Cannot find any of %s in predictions_dict %s.' %
                     (self._target_prediction_keys, predictions_dict))
    labels_tensor = _get_target_tensor(labels_dict, [self._labels_key])
    if labels_tensor is None:
      raise KeyError('Cannot find %s in labels_dict %s.' %
                     (self._labels_key, labels_dict))
    if self._tensor_index is None:
      return predictions_tensor, labels_tensor

    # If predictions are multi-class, we can use the tensor_index to choose a
    # class to evaluate.
    return self._select_class(predictions_tensor, labels_tensor)

  def _metric_key(self, base_key: Text) -> Text:
    """Constructs a metric key, including user-specified prefix if necessary.

    In cases with multi-headed models, an evaluation may need multiple instances
    of the same metric for different predictions and/or labels. To support this
    case, the metric should be named with the specified label to disambiguate
    between the two (and prevent key collisions).

    Args:
      base_key: The original key for the metric, often from metric_keys.

    Returns:
      Either the base key, or the key augmented with a specified tag or label.
    """
    if self._metric_tag:
      return metric_keys.tagged_key(base_key, self._metric_tag)
    return base_key

  @abc.abstractmethod
  def check_compatibility(self, features_dict: types.TensorTypeMaybeDict,
                          predictions_dict: types.TensorTypeMaybeDict,
                          labels_dict: types.TensorTypeMaybeDict) -> None:
    """Checks whether this metric is compatible with the model.

    This function should make this determination based on the features,
    predictions and labels dict. It should raise an Exception if the metric is
    not compatible with the model.

    Args:
      features_dict: Dictionary containing references to the features Tensors
        for the model.
      predictions_dict: Dictionary containing references to the predictions
        Tensors for the model.
      labels_dict: Dictionary containing references to the labels Tensors for
        the model.

    Raises:
      Exception if the metric is not compatible with the model.
    """
    raise NotImplementedError('not implemented')

  @abc.abstractmethod
  def get_metric_ops(self, features_dict: types.TensorTypeMaybeDict,
                     predictions_dict: types.TensorTypeMaybeDict,
                     labels_dict: types.TensorTypeMaybeDict
                    ) -> Dict[Text, Tuple[types.TensorType, types.TensorType]]:
    """Returns the metric_ops entry for this metric.

    Note that the metric will be added to metric_ops via
    metric_ops.update(metric.get_metric_ops()).

    Args:
      features_dict: Dictionary containing references to the features Tensors
        for the model.
      predictions_dict: Dictionary containing references to the predictions
        Tensors for the model.
      labels_dict: Dictionary containing references to the labels Tensors for
        the model.

    Returns:
      A metric op dictionary,
      i.e. a dictionary[metric_name] = (value_op, update_op) containing all
      the metrics and ops for this metric.
    """
    raise NotImplementedError('not implemented')

  def populate_stats_and_pop(self, combined_metrics: Dict[Text, Any],
                             output_metrics: Dict[Text, metrics_pb2.MetricValue]
                            ) -> None:
    """Converts the metric in `combined_metrics` to `output_metrics` and pops.

    Please override the method if the metric is NOT plot type and should be
    converted into non-float type. The metric should also be popped out of
    `combined_metrics` after conversion. By default, this method does nothing.
    The metric, along with the rest metrics in `combined_metrics` will be
    converted into float values afterwards.

    Args:
      combined_metrics: The dict containing raw TFMA metrics.
      output_metrics: The dict where we convert the metrics to.
    """
    pass

  def populate_plots_and_pop(self, plots: Dict[Text, Any],
                             output_plots: Dict[Text, metrics_pb2.PlotData]
                            ) -> None:
    """Converts the metric in `plots` to `output_plots` and pops.

    Please override the method if the metric is plot type. The plot should also
    be popped out of `plots` after conversion.

    Args:
      plots: The dict containing raw TFMA plots.
      output_plots: Dict from key to PlotData where we convert the plots to.
    """
    pass


# TODO(b/79364723): make metric key unique for post export metrics with
# different params.
@_export('example_count')
class _ExampleCount(_PostExportMetric):
  """Metric that counts the number of examples processed.

  We get the example count by looking at the predictions dictionary and picking
  a reference Tensor. If we can find a standard key (e.g.
  PredictionKeys.LOGISTIC, etc), we use that as the reference Tensor. Otherwise,
  we just use the first key in sorted order from one of the dictionaries
  (predictions, labels) as the reference Tensor.

  We assume the first dimension is the batch size, and take that to be the
  number of examples in the batch.
  """

  # TODO(b/116341909): Remove these declarations once PyType bug is fixed.
  _labels_key = ...  # type: Text
  _metric_tag = None  # type: Text

  def check_compatibility(self, features_dict: types.TensorTypeMaybeDict,
                          predictions_dict: types.TensorTypeMaybeDict,
                          labels_dict: types.TensorTypeMaybeDict) -> None:
    pass

  def get_metric_ops(self, features_dict: types.TensorTypeMaybeDict,
                     predictions_dict: types.TensorTypeMaybeDict,
                     labels_dict: types.TensorTypeMaybeDict
                    ) -> Dict[Text, Tuple[types.TensorType, types.TensorType]]:
    ref_tensor = _get_target_tensor(predictions_dict,
                                    self._target_prediction_keys)
    if ref_tensor is None:
      # Note that if predictions_dict is a Tensor and not a dict,
      # get_predictions_tensor will return predictions_dict, so if we get
      # here, if means that predictions_dict is a dict without any of the
      # standard keys.
      #
      # If we can't get any of standard keys, then pick the first key
      # in alphabetical order if the predictions dict is non-empty.
      # If the predictions dict is empty, try the labels dict.
      # If that is empty too, default to the empty Tensor.
      tf.logging.info(
          'ExampleCount post export metric: could not find any of '
          'the standard keys in predictions_dict (keys were: %s)',
          predictions_dict.keys())
      if predictions_dict is not None and predictions_dict.keys():
        first_key = sorted(predictions_dict.keys())[0]
        ref_tensor = predictions_dict[first_key]
        tf.logging.info('Using the first key from predictions_dict: %s',
                        first_key)
      elif labels_dict is not None:
        if types.is_tensor(labels_dict):
          ref_tensor = labels_dict
          tf.logging.info('Using the labels Tensor')
        elif labels_dict.keys():
          first_key = sorted(labels_dict.keys())[0]
          ref_tensor = labels_dict[first_key]
          tf.logging.info('Using the first key from labels_dict: %s', first_key)

      if ref_tensor is None:
        tf.logging.info('Could not find a reference Tensor for example count. '
                        'Defaulting to the empty Tensor.')
        ref_tensor = tf.constant([])

    return {
        self._metric_key(metric_keys.EXAMPLE_COUNT):
            metrics.total(tf.shape(ref_tensor)[0])
    }

  def populate_stats_and_pop(self, combine_metrics: Dict[Text, Any],
                             output_metrics: Dict[Text, metrics_pb2.MetricValue]
                            ) -> None:
    count_result = combine_metrics.pop(
        self._metric_key(metric_keys.EXAMPLE_COUNT))
    if isinstance(count_result, types.ValueWithConfidenceInterval):
      # We do not want to display confidence interval bounds on known
      # quantities such as ExampleCount, so we use the calculated value
      # without sampling.
      output_metrics[self._metric_key(
          metric_keys.EXAMPLE_COUNT)].double_value.value = (
              count_result.unsampled_value)
    else:
      output_metrics[self._metric_key(
          metric_keys.EXAMPLE_COUNT)].double_value.value = count_result


@_export('example_weight')
class _ExampleWeight(_PostExportMetric):
  """Metric that computes the sum of example weights."""

  _labels_key = ...  # type: Text
  _metric_tag = None  # type: Text

  def __init__(self,
               example_weight_key: Text,
               metric_tag: Optional[Text] = None) -> None:
    """Create a metric that computes the sum of example weights.

    Args:
      example_weight_key: The key of the example weight column in the
        features_dict.
      metric_tag: If provided, a custom metric tag. Only necessary to
        disambiguate instances of the same metric on different predictions or
        for readability concerns in tool output.
    """
    self._example_weight_key = example_weight_key
    super(_ExampleWeight, self).__init__(metric_tag=metric_tag)

  def check_compatibility(self, features_dict: types.TensorTypeMaybeDict,
                          predictions_dict: types.TensorTypeMaybeDict,
                          labels_dict: types.TensorTypeMaybeDict) -> None:
    _check_weight_present(features_dict, self._example_weight_key)

  def get_metric_ops(self, features_dict: types.TensorTypeMaybeDict,
                     predictions_dict: types.TensorTypeMaybeDict,
                     labels_dict: types.TensorTypeMaybeDict
                    ) -> Dict[Text, Tuple[types.TensorType, types.TensorType]]:
    value = features_dict[self._example_weight_key]
    return {self._metric_key(metric_keys.EXAMPLE_WEIGHT): metrics.total(value)}

  def populate_stats_and_pop(self, combine_metrics: Dict[Text, Any],
                             output_metrics: Dict[Text, metrics_pb2.MetricValue]
                            ) -> None:
    weight_result = combine_metrics.pop(
        self._metric_key(metric_keys.EXAMPLE_WEIGHT))
    if isinstance(weight_result, types.ValueWithConfidenceInterval):
      # We do not want to display confidence interval bounds on known
      # quantities such as ExampleWeight, so we use the calculated value
      # without sampling.
      output_metrics[self._metric_key(
          metric_keys.EXAMPLE_WEIGHT)].double_value.value = (
              weight_result.unsampled_value)
    else:
      output_metrics[self._metric_key(
          metric_keys.EXAMPLE_WEIGHT)].double_value.value = weight_result


_DEFAULT_NUM_BUCKETS = 10000


@_export('squared_pearson_correlation')
class _SquaredPearsonCorrelation(_PostExportMetric):
  """Metric that computes the squared pearson correlation (r squared)."""

  _target_prediction_keys = ...  # type: List[Text]
  _labels_key = ...  # type: Text
  _metric_tag = None  # type: Text
  _tensor_index = ...  # type: int

  def __init__(self,
               example_weight_key: Optional[Text] = None,
               target_prediction_keys: Optional[List[Text]] = None,
               labels_key: Optional[Text] = None,
               metric_tag: Optional[Text] = None) -> None:
    """Create a metric that computes the squared pearson correlation.

    Args:
      example_weight_key: The key of the example weight column in the features
        dict. If None, all predictions are given a weight of 1.0.
      target_prediction_keys: If provided, the prediction keys to look for in
        order.
      labels_key: If provided, a custom label key.
      metric_tag: If provided, a custom metric tag. Only necessary to
        disambiguate instances of the same metric on different predictions.
    """
    self._example_weight_key = example_weight_key
    super(_SquaredPearsonCorrelation, self).__init__(
        target_prediction_keys=target_prediction_keys,
        labels_key=labels_key,
        metric_tag=metric_tag)

  def check_compatibility(self, features_dict: types.TensorTypeMaybeDict,
                          predictions_dict: types.TensorTypeMaybeDict,
                          labels_dict: types.TensorTypeMaybeDict) -> None:
    _check_weight_present(features_dict, self._example_weight_key)
    self._get_labels_and_predictions(predictions_dict, labels_dict)

  def get_metric_ops(self, features_dict: types.TensorTypeMaybeDict,
                     predictions_dict: types.TensorTypeMaybeDict,
                     labels_dict: types.TensorTypeMaybeDict
                    ) -> Dict[Text, Tuple[types.TensorType, types.TensorType]]:
    # Note that we have to squeeze predictions, labels, weights so they are all
    # N element vectors (otherwise some of them might be N x 1 tensors, and
    # multiplying a N element vector with a N x 1 tensor uses matrix
    # multiplication rather than element-wise multiplication).
    predictions, labels = self._get_labels_and_predictions(
        predictions_dict, labels_dict)
    predictions = _flatten_to_one_dim(tf.cast(predictions, tf.float64))
    labels = _flatten_to_one_dim(tf.cast(labels, tf.float64))
    weights = tf.ones_like(predictions)
    if self._example_weight_key:
      weights = _flatten_to_one_dim(
          tf.cast(features_dict[self._example_weight_key], tf.float64))
    return {
        self._metric_key(metric_keys.SQUARED_PEARSON_CORRELATION):
            metrics.squared_pearson_correlation(predictions, labels, weights)
    }

  def populate_stats_and_pop(self, combine_metrics: Dict[Text, Any],
                             output_metrics: Dict[Text, metrics_pb2.MetricValue]
                            ) -> None:
    r_squared = combine_metrics.pop(
        self._metric_key(metric_keys.SQUARED_PEARSON_CORRELATION))
    bounded_value = output_metrics[self._metric_key(
        metric_keys.SQUARED_PEARSON_CORRELATION)].bounded_value
    if isinstance(r_squared, types.ValueWithConfidenceInterval):
      bounded_value.value.value = r_squared.value
      bounded_value.lower_bound.value = r_squared.lower_bound
      bounded_value.upper_bound.value = r_squared.upper_bound
      bounded_value.methodology = metrics_pb2.BoundedValue.POISSON_BOOTSTRAP
    else:
      bounded_value.value.value = r_squared


@_export('calibration_plot_and_prediction_histogram')
class _CalibrationPlotAndPredictionHistogram(_PostExportMetric):
  """Plot metric for calibration plot and prediction histogram.

  Note that this metric is only applicable to models for which the predictions
  and labels are in [0, 1].

  The plot contains uniformly-sized buckets for predictions in [0, 1],
  and additional buckets for predictions less than 0 and greater than 1 at the
  ends.
  """

  _target_prediction_keys = ...  # type: List[Text]
  _labels_key = ...  # type: Text
  _metric_tag = None  # type: Text
  _tensor_index = ...  # type: int

  def __init__(self,
               example_weight_key: Optional[Text] = None,
               num_buckets: int = _DEFAULT_NUM_BUCKETS,
               target_prediction_keys: Optional[List[Text]] = None,
               labels_key: Optional[Text] = None,
               metric_tag: Optional[Text] = None,
               tensor_index: Optional[int] = None) -> None:
    """Create a plot metric for calibration plot and prediction histogram.

    Predictions should be one of:
      (a) a single float in [0, 1]
      (b) a dict containing the LOGISTIC key
      (c) a dict containing the PREDICTIONS key, where the prediction is
          in [0, 1]

    Label should be a single float that is in [0, 1].

    Args:
      example_weight_key: The key of the example weight column in the features
        dict. If None, all predictions are given a weight of 1.0.
      num_buckets: The number of buckets used for the plot.
      target_prediction_keys: If provided, the prediction keys to look for in
        order.
      labels_key: If provided, a custom label key.
      metric_tag: If provided, a custom metric tag. Only necessary to
        disambiguate instances of the same metric on different predictions.
      tensor_index: Optional index to specify class predictions to calculate
        metrics on in the case of multi-class models.
    """
    self._example_weight_key = example_weight_key
    self._num_buckets = num_buckets
    super(_CalibrationPlotAndPredictionHistogram, self).__init__(
        target_prediction_keys,
        labels_key,
        metric_tag,
        tensor_index=tensor_index)

  def check_compatibility(self, features_dict: types.TensorTypeMaybeDict,
                          predictions_dict: types.TensorTypeMaybeDict,
                          labels_dict: types.TensorTypeMaybeDict) -> None:
    _check_weight_present(features_dict, self._example_weight_key)
    self._get_labels_and_predictions(predictions_dict, labels_dict)

  def get_metric_ops(self, features_dict: types.TensorTypeMaybeDict,
                     predictions_dict: types.TensorTypeMaybeDict,
                     labels_dict: types.TensorTypeMaybeDict
                    ) -> Dict[Text, Tuple[types.TensorType, types.TensorType]]:
    # Note that we have to squeeze predictions, labels, weights so they are all
    # N element vectors (otherwise some of them might be N x 1 tensors, and
    # multiplying a N element vector with a N x 1 tensor uses matrix
    # multiplication rather than element-wise multiplication).
    squeezed_weights = None
    if self._example_weight_key:
      squeezed_weights = tf.squeeze(features_dict[self._example_weight_key])
    prediction_tensor, label_tensor = self._get_labels_and_predictions(
        predictions_dict, labels_dict)
    return {
        self._metric_key(metric_keys.CALIBRATION_PLOT_MATRICES):
            metrics.calibration_plot(
                predictions=tf.squeeze(prediction_tensor),
                labels=tf.squeeze(label_tensor),
                left=0.0,
                right=1.0,
                num_buckets=self._num_buckets,
                weights=squeezed_weights),
        self._metric_key(metric_keys.CALIBRATION_PLOT_BOUNDARIES):
            (tf.range(0.0, self._num_buckets + 1) / self._num_buckets,
             tf.no_op()),
    }

  def populate_plots_and_pop(self, plots: Dict[Text, Any],
                             output_plots: Dict[Text, metrics_pb2.PlotData]
                            ) -> None:
    matrices = plots.pop(
        self._metric_key(metric_keys.CALIBRATION_PLOT_MATRICES))
    boundaries = plots.pop(
        self._metric_key(metric_keys.CALIBRATION_PLOT_BOUNDARIES))
    if len(matrices) != len(boundaries) + 1:
      raise ValueError(
          'len(matrices) should be equal to len(boundaries) + 1, but lengths '
          'were len(matrices)=%d and len(boundaries)=%d instead' %
          (len(matrices), len(boundaries)))

    for matrix_row, lower_threshold, upper_threshold in zip(
        matrices, [float('-inf')] + list(boundaries),
        list(boundaries) + [float('inf')]):
      total_pred, total_label, total_weight = matrix_row
      if isinstance(lower_threshold, types.ValueWithConfidenceInterval):
        lower_threshold = lower_threshold.unsampled_value
      if isinstance(upper_threshold, types.ValueWithConfidenceInterval):
        upper_threshold = upper_threshold.unsampled_value
      if isinstance(total_weight, types.ValueWithConfidenceInterval):
        total_weight = total_weight.unsampled_value
      if isinstance(total_pred, types.ValueWithConfidenceInterval):
        total_pred = total_pred.unsampled_value
      if isinstance(total_label, types.ValueWithConfidenceInterval):
        total_label = total_label.unsampled_value
      # TODO(ckuhn): Figure out how this should work with uncertainty calculated
      # using the Poisson bootstrap method.
      output_plots[self._metric_key(
          metric_keys.DEFAULT_PREFIX
      )].calibration_histogram_buckets.buckets.add(
          lower_threshold_inclusive=lower_threshold,
          upper_threshold_exclusive=upper_threshold,
          total_weighted_refined_prediction={
              'value': total_pred,
          },
          total_weighted_label={
              'value': total_label,
          },
          num_weighted_examples={
              'value': total_weight,
          },
      )


def _flatten_to_one_dim(tensor):
  # We use this instead of squeeze so we don't squeeze out a Tensor with
  # shape [0]. Squeezing a Tensor with shape [0] results in a shape of [],
  # which makes concat unhappy.
  return tf.reshape(tensor, [tf.size(tensor)])


def _create_predictions_labels_weights_for_fractional_labels(
    prediction_tensor, label_tensor, weight_tensor):
  """Creates updated predictions, labels, weights Tensors for fractional labels.

  Assumes labels are in [0, 1].

  We treat an example with fractional label as two separate examples, one with
  positive label and one with negative label, where each example is weighted by
  the label accordingly.

  More concretely, an example with prediction p, label l and weight w becomes
  two examples:
   - one with prediction p, label 1.0, and weight w * l
   - one with prediction p, label 0.0, and weight w * (1.0 - l)

  Args:
    prediction_tensor: Prediction tensor (should have dim N).
    label_tensor: Label tensor (should have dim N).
    weight_tensor: Weight tensor (should have dim N).

  Returns:
    Tuple of updated (prediction_tensor, label_tensor, weight_tensor).
  """

  with tf.control_dependencies([
      tf.assert_greater_equal(label_tensor, np.float64(0.0)),
      tf.assert_less_equal(label_tensor, np.float64(1.0))
  ]):
    return (
        tf.concat([prediction_tensor, prediction_tensor], axis=0),
        tf.concat([tf.ones_like(label_tensor),
                   tf.zeros_like(label_tensor)],
                  axis=0),
        tf.concat([
            weight_tensor * label_tensor, weight_tensor * (1.0 - label_tensor)
        ],
                  axis=0),
    )


class _ConfusionMatrixBasedMetric(_PostExportMetric):
  """Base class for metrics that use confusion matrices."""

  _target_prediction_keys = ...  # type: List[Text]
  _labels_key = ...  # type: Text
  _metric_tag = None  # type: Text

  def __init__(self,
               thresholds: List[float],
               example_weight_key: Optional[Text] = None,
               target_prediction_keys: Optional[List[Text]] = None,
               labels_key: Optional[Text] = None,
               metric_tag: Optional[Text] = None,
               tensor_index: Optional[int] = None) -> None:
    """Create a metric that computes the confusion matrix at given thresholds.

    Predictions should be one of:
      (a) a single float in [0, 1]
      (b) a dict containing the LOGISTIC key
      (c) a dict containing the PREDICTIONS key, where the prediction is
          in [0, 1]

    Label should be a single float that is either exactly 0 or exactly 1
    (soft labels, i.e. labels between 0 and 1 are *not* supported).

    Args:
      thresholds: List of thresholds to compute the confusion matrix at.
      example_weight_key: The key of the example weight column in the features
        dict. If None, all predictions are given a weight of 1.0.
      target_prediction_keys: If provided, the prediction keys to look for in
        order.
      labels_key: If provided, a custom label key.
      metric_tag: If provided, a custom metric tag. Only necessary to
        disambiguate instances of the same metric on different predictions.
      tensor_index: Optional index to specify class predictions to calculate
        metrics on in the case of multi-class models.
    """
    self._example_weight_key = example_weight_key
    self._thresholds = sorted(thresholds)
    super(_ConfusionMatrixBasedMetric, self).__init__(
        target_prediction_keys,
        labels_key,
        metric_tag,
        tensor_index=tensor_index)

  def check_compatibility(self, features_dict: types.TensorTypeMaybeDict,
                          predictions_dict: types.TensorTypeMaybeDict,
                          labels_dict: types.TensorTypeMaybeDict) -> None:
    _check_weight_present(features_dict, self._example_weight_key)
    self._get_labels_and_predictions(predictions_dict, labels_dict)

  def joined_confusion_matrix_metric_ops(
      self,
      features_dict: types.TensorTypeMaybeDict,
      predictions_dict: types.TensorTypeMaybeDict,
      labels_dict: types.TensorTypeMaybeDict,
  ) -> Tuple[types.TensorType, types.TensorType]:
    """Calls confusion_matrix_metric_ops and joins the results.

    Args:
      features_dict: Features dict.
      predictions_dict: Predictions dict.
      labels_dict: Labels dict.

    Returns:
      (value_ops, update_ops) for the confusion matrix.  Note that the value_op
      produces a matrix as described in the comments below.
    """
    values, update_ops = self.confusion_matrix_metric_ops(
        features_dict, predictions_dict, labels_dict)
    # The final matrix will look like the following:
    #
    # [ fn@threshold_0 tn@threshold_0 ... recall@threshold_0 ]
    # [ fn@threshold_1 tn@threshold_1 ... recall@threshold_1 ]
    # [       :              :        ...         :          ]
    # [       :              :        ...         :          ]
    # [ fn@threshold_k tn@threshold_k ... recall@threshold_k ]
    #
    value_op = tf.transpose(
        tf.stack([
            values['fn'], values['tn'], values['fp'], values['tp'],
            values['precision'], values['recall']
        ]))
    update_op = tf.group(update_ops['fn'], update_ops['tn'], update_ops['fp'],
                         update_ops['tp'])

    return (value_op, update_op)

  def confusion_matrix_metric_ops(
      self,
      features_dict: types.TensorTypeMaybeDict,
      predictions_dict: types.TensorTypeMaybeDict,
      labels_dict: types.TensorTypeMaybeDict,
  ) -> Tuple[Dict[Text, List[types.TensorType]], Dict[Text, List[types
                                                                 .TensorType]]]:
    """Metric ops for computing confusion matrix at the given thresholds.

    This is factored out because it's common to AucPlots and
    ConfusionMatrixAtThresholds.

    Args:
      features_dict: Features dict.
      predictions_dict: Predictions dict.
      labels_dict: Labels dict.

    Returns:
      (value_ops, update_ops) for the confusion matrix.
    """
    # Note that we have to squeeze predictions, labels, weights so they are all
    # N element vectors (otherwise some of them might be N x 1 tensors, and
    # multiplying a N element vector with a N x 1 tensor uses matrix
    # multiplication rather than element-wise multiplication).
    predictions, labels = self._get_labels_and_predictions(
        predictions_dict, labels_dict)
    prediction_tensor = _flatten_to_one_dim(tf.cast(predictions, tf.float64))
    label_tensor = _flatten_to_one_dim(tf.cast(labels, tf.float64))
    squeezed_weights = tf.ones_like(prediction_tensor)
    if self._example_weight_key:
      squeezed_weights = _flatten_to_one_dim(
          tf.cast(features_dict[self._example_weight_key], tf.float64))
    prediction_tensor, label_tensor, squeezed_weights = (
        _create_predictions_labels_weights_for_fractional_labels(
            prediction_tensor, label_tensor, squeezed_weights))

    # TODO(b/72239826): Expose _confusion_matrix_at_thresholds for OSS?
    values, update_ops = metrics_impl._confusion_matrix_at_thresholds(  # pylint: disable=protected-access
        label_tensor, prediction_tensor, self._thresholds, squeezed_weights)

    values['precision'] = values['tp'] / (values['tp'] + values['fp'])
    values['recall'] = values['tp'] / (values['tp'] + values['fn'])
    return (values, update_ops)  # pytype: disable=bad-return-type


def _set_output_matrix_field(matrix_entry, output_matrix, field_name):
  """Sets bounded and double values for a component of a confusion matrix.

  This is a convenience function to handle setting both value types in the
  confusion matrix proto. We want to migrate to using just the bounded value
  in the UI and analysis, but for some time will be needing to populate both.
  This also handles both scalar values and ValuesWithConfidenceInterval.

  Args:
    matrix_entry: The original value from the metric ops.
    output_matrix: The ConfusionMatrixAtThreshold proto to populate.
    field_name: The name of the double_value field to set.
  """
  bounded_value = getattr(output_matrix, 'bounded_%s' % field_name)
  if isinstance(matrix_entry, types.ValueWithConfidenceInterval):
    bounded_value.value.value = matrix_entry.value
    bounded_value.lower_bound.value = matrix_entry.lower_bound
    bounded_value.upper_bound.value = matrix_entry.upper_bound
    bounded_value.methodology = metrics_pb2.BoundedValue.POISSON_BOOTSTRAP
    setattr(output_matrix, field_name, matrix_entry[0])
  else:
    bounded_value.value.value = matrix_entry
    setattr(output_matrix, field_name, matrix_entry)


def _create_confusion_matrix_proto(
    matrix: List[Any], threshold: float
) -> metrics_pb2.ConfusionMatrixAtThresholds.ConfusionMatrixAtThreshold:
  """Populates matrix proto values from value_op matrix."""
  output_matrix = (
      metrics_pb2.ConfusionMatrixAtThresholds.ConfusionMatrixAtThreshold())
  output_matrix.threshold = threshold
  _set_output_matrix_field(matrix[0], output_matrix, 'false_negatives')
  _set_output_matrix_field(matrix[1], output_matrix, 'true_negatives')
  _set_output_matrix_field(matrix[2], output_matrix, 'false_positives')
  _set_output_matrix_field(matrix[3], output_matrix, 'true_positives')
  _set_output_matrix_field(matrix[4], output_matrix, 'precision')
  _set_output_matrix_field(matrix[5], output_matrix, 'recall')
  return output_matrix


@_export('confusion_matrix_at_thresholds')
class _ConfusionMatrixAtThresholds(_ConfusionMatrixBasedMetric):
  """Confusion matrix at thresholds."""

  def get_metric_ops(self, features_dict: types.TensorTypeMaybeDict,
                     predictions_dict: types.TensorTypeMaybeDict,
                     labels_dict: types.TensorTypeMaybeDict
                    ) -> Dict[Text, Tuple[types.TensorType, types.TensorType]]:
    value_op, update_op = self.joined_confusion_matrix_metric_ops(
        features_dict, predictions_dict, labels_dict)
    # The format and lint tools don't agree on the formatting here.
    # pyformat: disable

    return {
        self._metric_key(metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_MATRICES): (
            value_op, update_op),
        self._metric_key(
            metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_THRESHOLDS): (
                tf.identity(self._thresholds), tf.no_op()),
    }
    # pyformat: enable

  def populate_stats_and_pop(self, combine_metrics: Dict[Text, Any],
                             output_metrics: Dict[Text, metrics_pb2.MetricValue]
                            ) -> None:
    matrices = combine_metrics.pop(
        self._metric_key(metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_MATRICES))
    thresholds = combine_metrics.pop(
        self._metric_key(metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS_THRESHOLDS))
    # We assume that thresholds are already sorted.
    if len(matrices) != len(thresholds):
      raise ValueError(
          'matrices should have the same length as thresholds, but lengths '
          'were: matrices: %d, thresholds: %d' %
          (len(matrices), len(thresholds)))

    for threshold, matrix in zip(thresholds, matrices):
      if isinstance(threshold, types.ValueWithConfidenceInterval):
        threshold = threshold.unsampled_value
      (output_metrics[self._metric_key(
          metric_keys.CONFUSION_MATRIX_AT_THRESHOLDS)]
       .confusion_matrix_at_thresholds.matrices.add().CopyFrom(
           _create_confusion_matrix_proto(matrix, threshold)))


@_export('auc_plots')
class _AucPlots(_ConfusionMatrixBasedMetric):
  """Plot metric for AUROC and AUPRC for predictions in [0, 1]."""

  _thresholds = ...  # type: List[float]
  _labels_key = ...  # type: Text
  _metric_tag = None  # type: Text
  _tensor_index = ...  # type: int

  def __init__(self,
               example_weight_key: Optional[Text] = None,
               num_buckets: int = _DEFAULT_NUM_BUCKETS,
               target_prediction_keys: Optional[List[Text]] = None,
               labels_key: Optional[Text] = None,
               metric_tag: Optional[Text] = None,
               tensor_index: Optional[int] = None) -> None:
    """Create a plot metric for AUROC and AUPRC.

    Predictions should be one of:
      (a) a single float in [0, 1]
      (b) a dict containing the LOGISTIC key
      (c) a dict containing the PREDICTIONS key, where the prediction is
          in [0, 1]

    Label should be a single float that is either exactly 0 or exactly 1
    (soft labels, i.e. labels between 0 and 1 are *not* supported).

    Args:
      example_weight_key: The key of the example weight column in the features
        dict. If None, all predictions are given a weight of 1.0.
      num_buckets: The number of buckets used for plot.
      target_prediction_keys: If provided, the prediction keys to look for in
        order.
      labels_key: If provided, a custom label key.
      metric_tag: If provided, a custom metric tag. Only necessary to
        disambiguate instances of the same metric on different predictions.
      tensor_index: Optional index to specify class predictions to calculate
        metrics on in the case of multi-class models.
    """
    thresholds = [i * 1.0 / num_buckets for i in range(0, num_buckets + 1)]
    thresholds = [-1e-6] + thresholds
    super(_AucPlots, self).__init__(
        example_weight_key=example_weight_key,
        thresholds=thresholds,
        target_prediction_keys=target_prediction_keys,
        labels_key=labels_key,
        metric_tag=metric_tag,
        tensor_index=tensor_index)

  def get_metric_ops(self, features_dict: types.TensorTypeMaybeDict,
                     predictions_dict: types.TensorTypeMaybeDict,
                     labels_dict: types.TensorTypeMaybeDict
                    ) -> Dict[Text, Tuple[types.TensorType, types.TensorType]]:

    value_op, update_op = self.joined_confusion_matrix_metric_ops(
        features_dict, predictions_dict, labels_dict)
    return {
        self._metric_key(metric_keys.AUC_PLOTS_MATRICES): (value_op, update_op),
        self._metric_key(metric_keys.AUC_PLOTS_THRESHOLDS):
            (tf.identity(self._thresholds), tf.no_op()),
    }

  def populate_plots_and_pop(self, plots: Dict[Text, Any],
                             output_plots: Dict[Text, metrics_pb2.PlotData]
                            ) -> None:
    matrices = plots.pop(self._metric_key(metric_keys.AUC_PLOTS_MATRICES))
    thresholds = plots.pop(self._metric_key(metric_keys.AUC_PLOTS_THRESHOLDS))
    if len(matrices) != len(thresholds):
      raise ValueError(
          'len(matrices) should be equal to len(thresholds), but lengths were '
          'len(matrices)=%d and len(thresholds)=%d instead' %
          (len(matrices), len(thresholds)))
    for matrix_row, threshold in zip(matrices, list(thresholds)):
      matrix = output_plots[self._metric_key(
          metric_keys.DEFAULT_PREFIX
      )].confusion_matrix_at_thresholds.matrices.add()
      if isinstance(threshold, types.ValueWithConfidenceInterval):
        threshold = threshold.unsampled_value
      matrix.threshold = threshold
      matrix.CopyFrom(_create_confusion_matrix_proto(matrix_row, threshold))


@_export('auc')
class _Auc(_PostExportMetric):
  """Metric that computes bounded AUC or AUPRC for predictions in [0, 1].

  This calls tf.metrics.auc to do the computation with 10000 buckets instead of
  the default (200) for more precision. We use 'careful_interpolation' summation
  for the metric value, and also 'minoring' and 'majoring' to generate the
  boundaries for the metric.
  """

  _target_prediction_keys = ...  # type: List[Text]
  _labels_key = ...  # type: Text
  _metric_tag = None  # type: Text
  _tensor_index = ...  # type: int

  def __init__(self,
               example_weight_key: Optional[Text] = None,
               curve='ROC',
               num_buckets: int = _DEFAULT_NUM_BUCKETS,
               target_prediction_keys: Optional[List[Text]] = None,
               labels_key: Optional[Text] = None,
               metric_tag: Optional[Text] = None,
               tensor_index: Optional[int] = None) -> None:
    """Create a metric that computes bounded AUROC or AUPRC.

    Predictions should be one of:
      (a) a single float in [0, 1]
      (b) a dict containing the LOGISTIC key
      (c) a dict containing the PREDICTIONS key, where the prediction is
          in [0, 1]

    Label should be a single float that is either exactly 0 or exactly 1
    (soft labels, i.e. labels between 0 and 1 are *not* supported).

    Args:
      example_weight_key: The key of the example weight column in the features
        dict. If None, all predictions are given a weight of 1.0.
      curve: Specifies the name of the curve to be computed, 'ROC' [default] or
        'PR' for the Precision-Recall-curve. It will be passed to
        tf.metrics.auc() directly.
      num_buckets: The number of buckets used for the curve. (num_buckets + 1)
        is used as the num_thresholds in tf.metrics.auc().
      target_prediction_keys: If provided, the prediction keys to look for in
        order.
      labels_key: If provided, a custom label key.
      metric_tag: If provided, a custom metric tag. Only necessary to
        disambiguate instances of the same metric on different predictions.
      tensor_index: Optional index to specify class predictions to calculate
        metrics on in the case of multi-class models.

    Raises:
      ValueError: if the curve is neither 'ROC' nor 'PR'.
    """
    self._example_weight_key = example_weight_key
    self._curve = curve
    self._num_buckets = num_buckets

    if curve == 'ROC':
      self._metric_name = metric_keys.AUC
    elif curve == 'PR':
      self._metric_name = metric_keys.AUPRC
    else:
      raise ValueError('got unsupported curve: %s' % curve)
    super(_Auc, self).__init__(
        target_prediction_keys=target_prediction_keys,
        labels_key=labels_key,
        metric_tag=metric_tag,
        tensor_index=tensor_index)

  def check_compatibility(self, features_dict: types.TensorTypeMaybeDict,
                          predictions_dict: types.TensorTypeMaybeDict,
                          labels_dict: types.TensorTypeMaybeDict) -> None:
    _check_weight_present(features_dict, self._example_weight_key)
    self._get_labels_and_predictions(predictions_dict, labels_dict)

  def get_metric_ops(self, features_dict: types.TensorTypeMaybeDict,
                     predictions_dict: types.TensorTypeMaybeDict,
                     labels_dict: types.TensorTypeMaybeDict
                    ) -> Dict[Text, Tuple[types.TensorType, types.TensorType]]:
    # Note that we have to squeeze predictions, labels, weights so they are all
    # N element vectors (otherwise some of them might be N x 1 tensors, and
    # multiplying a N element vector with a N x 1 tensor uses matrix
    # multiplication rather than element-wise multiplication).
    predictions, labels = self._get_labels_and_predictions(
        predictions_dict, labels_dict)
    predictions = _flatten_to_one_dim(tf.cast(predictions, tf.float64))
    labels = _flatten_to_one_dim(tf.cast(labels, tf.float64))
    weights = tf.ones_like(predictions)
    if self._example_weight_key:
      weights = _flatten_to_one_dim(
          tf.cast(features_dict[self._example_weight_key], tf.float64))

    predictions, labels, weights = (
        _create_predictions_labels_weights_for_fractional_labels(
            predictions, labels, weights))

    value_ops, value_update = tf.metrics.auc(
        labels=labels,
        predictions=predictions,
        weights=weights,
        num_thresholds=self._num_buckets + 1,
        curve=self._curve,
        summation_method='careful_interpolation')
    lower_bound_ops, lower_bound_update = tf.metrics.auc(
        labels=labels,
        predictions=predictions,
        weights=weights,
        num_thresholds=self._num_buckets + 1,
        curve=self._curve,
        summation_method='minoring')
    upper_bound_ops, upper_bound_update = tf.metrics.auc(
        labels=labels,
        predictions=predictions,
        weights=weights,
        num_thresholds=self._num_buckets + 1,
        curve=self._curve,
        summation_method='majoring')

    return {
        self._metric_key(self._metric_name): (value_ops, value_update),
        metric_keys.lower_bound_key(self._metric_key(self._metric_name)):
            (lower_bound_ops, lower_bound_update),
        metric_keys.upper_bound_key(self._metric_key(self._metric_name)):
            (upper_bound_ops, upper_bound_update),
    }

  def populate_stats_and_pop(self, combine_metrics: Dict[Text, Any],
                             output_metrics: Dict[Text, metrics_pb2.MetricValue]
                            ) -> None:
    _populate_to_auc_bounded_value_and_pop(combine_metrics, output_metrics,
                                           self._metric_key(self._metric_name))


def _cast_or_convert(original: tf.Tensor, target_type: tf.DType) -> tf.Tensor:
  if target_type == tf.string and original.dtype != tf.string:
    return tf.as_string(original)
  else:
    return tf.cast(original, target_type)


def _class_ids(probabilities: tf.Tensor) -> tf.Tensor:
  """Returns class_ids associated with given probabilities tensor.

  Args:
    probabilities: Batch of probablitities (i.e. [[class0, class1, ...], ...]
      with shape [batch, n_classes]).

  Returns:
    Class IDs for N classes (i.e. [[0, 1, ..., n-1], ...] with shape
    [batch, n_classes]).
  """
  n_classes = tf.cast(tf.shape(probabilities)[-1], tf.int64)
  # Tensor representing shape of class_ids expanded by batch dims: [1,n_classes]
  expanded_dims = tf.concat(
      [tf.ones_like(tf.shape(probabilities))[:-1], [n_classes]], axis=0)
  # Tensor for multiplying tiles by batch size. Shape should be [batch_size,1]
  batch_multiplier = tf.concat([tf.shape(probabilities)[:-1], [1]], axis=0)
  # Batches of [0, ..., n_classes]
  return tf.tile(
      input=tf.reshape(tensor=tf.range(n_classes), shape=expanded_dims),
      multiples=batch_multiplier)


class _PrecisionRecallAtK(_PostExportMetric):
  """Metric that computes precision or recall at K for classification models.

  Create a metric that computes precision or recall at K.

  Predictions should be a dict containing the CLASSES key and PROBABILITIES
  keys. Predictions should have the same size for all examples. The model
  should NOT, for instance, produce 2 classes on one example and 4 classes on
  another example.

  Labels should be a string Tensor, or a SparseTensor whose dense form is
  a string Tensor whose entries are the corresponding labels. Note that the
  values of the CLASSES in the predictions and that of labels will be compared
  directly, so they should come from the "same vocabulary", so if predictions
  are class IDs, then labels should be class IDs, and so on.
  """

  _target_prediction_keys = ...  # type: List[Text]
  _labels_key = ...  # type: Text
  _metric_tag = None  # type: Text

  def __init__(self,
               metric_name: Text,
               cutoffs: List[int],
               example_weight_key: Optional[Text] = None,
               target_prediction_keys: Optional[List[Text]] = None,
               labels_key: Optional[Text] = None,
               metric_tag: Optional[Text] = None,
               classes_key: Optional[Text] = None,
               probabilities_key: Optional[Text] = None):
    """Creates a metric that computes either precision or recall at `k`.

    Args:
      metric_name: Metric (PRECISION_AT_KEY or RECALL_AT_K) to compute.
      cutoffs: List of `k` values at which to compute the precision and recall.
        Use a value of `k` = 0 to indicate that all predictions should be
        considered.
      example_weight_key: The optional key of the example weight column in the
        features_dict. If not given, all examples will be assumed to have a
        weight of 1.0.
      target_prediction_keys: Ignored (use classes_key and probabilities_key).
      labels_key: Optionally, the key from labels_dict to use.
      metric_tag: If provided, a custom metric tag. Only necessary to
        disambiguate instances of the same metric on different predictions.
      classes_key: Optionally, the key from predictions that specifies classes.
      probabilities_key: Optionally, the key from predictions that specifies
        probabilities.
    """
    self._metric_name = metric_name
    self._cutoffs = cutoffs
    self._example_weight_key = example_weight_key
    classes_key = classes_key or prediction_keys.PredictionKeys.CLASSES
    probabilities_key = (
        probabilities_key or prediction_keys.PredictionKeys.PROBABILITIES)
    self._classes_keys = [classes_key]
    self._probabilities_keys = [probabilities_key]
    if metric_tag:
      self._classes_keys.extend(
          _additional_prediction_keys(self._classes_keys, metric_tag, None))
      self._probabilities_keys.extend(
          _additional_prediction_keys(self._probabilities_keys, metric_tag,
                                      None))

    super(_PrecisionRecallAtK, self).__init__(target_prediction_keys,
                                              labels_key, metric_tag)

  def check_compatibility(self, features_dict: types.TensorTypeMaybeDict,
                          predictions_dict: types.TensorTypeMaybeDict,
                          labels_dict: types.TensorTypeMaybeDict) -> None:
    if not isinstance(predictions_dict, dict):
      raise TypeError('predictions_dict should be a dict. predictions_dict '
                      'was: %s' % predictions_dict)
    if _get_target_tensor(predictions_dict, self._classes_keys) is None:
      raise KeyError('predictions_dict should contain one of %s. '
                     'predictions_dict was: %s' %
                     (self._classes_keys, predictions_dict))
    if _get_target_tensor(predictions_dict, self._probabilities_keys) is None:
      raise KeyError('predictions_dict should contain one of %s. '
                     'predictions_dict was: %s' %
                     (self._probabilities_keys, predictions_dict))
    if self._labels_key:
      labels_dict = labels_dict[self._labels_key]

    if not types.is_tensor(labels_dict):
      raise TypeError('labels_dict should be a tensor. labels_dict was: %s' %
                      labels_dict)
    _check_weight_present(features_dict, self._example_weight_key)

  def get_metric_ops(self, features_dict: types.TensorTypeMaybeDict,
                     predictions_dict: types.TensorTypeMaybeDict,
                     labels_dict: types.TensorTypeMaybeDict
                    ) -> Dict[Text, Tuple[types.TensorType, types.TensorType]]:

    squeezed_weights = None
    if self._example_weight_key:
      squeezed_weights = _flatten_to_one_dim(
          features_dict[self._example_weight_key])

    labels = labels_dict
    if self._labels_key:
      labels = labels_dict[self._labels_key]
    if isinstance(labels_dict, tf.SparseTensor):
      labels = tf.sparse_tensor_to_dense(labels_dict, default_value='')

    # Expand dims if necessary.
    labels = tf.cond(
        tf.equal(tf.rank(labels),
                 1), lambda: tf.expand_dims(labels, -1), lambda: labels)

    classes = _get_target_tensor(predictions_dict, self._classes_keys)
    scores = _get_target_tensor(predictions_dict, self._probabilities_keys)

    # To support canned Estimators which right now only expose the argmax class
    # id, if labels are ints then then the classes are likely class_ids in
    # string form, so we can automatically expand the classes to the full set
    # for matching the labels (see b/113170729).
    if labels.dtype == tf.int64:
      classes = tf.cond(
          # Match only when classes has a single item (i.e. argmax).
          tf.equal(tf.shape(classes)[-1], 1),
          lambda: tf.as_string(_class_ids(scores)),
          lambda: classes)

    labels = _cast_or_convert(labels, classes.dtype)

    if self._metric_name == metric_keys.PRECISION_AT_K:
      metric_ops = metrics.precision_at_k(classes, scores, labels,
                                          self._cutoffs, squeezed_weights)
    else:
      metric_ops = metrics.recall_at_k(classes, scores, labels, self._cutoffs,
                                       squeezed_weights)

    return {self._metric_key(self._metric_name): metric_ops}

  def populate_stats_and_pop(self, combine_metrics: Dict[Text, Any],
                             output_metrics: Dict[Text, metrics_pb2.MetricValue]
                            ) -> None:
    table = combine_metrics.pop(self._metric_key(self._metric_name))
    cutoff_column = table[:, 0]
    value_column = table[:, 1]
    for cutoff, value in zip(cutoff_column, value_column):
      if isinstance(cutoff, types.ValueWithConfidenceInterval):
        cutoff = cutoff.unsampled_value
      row = output_metrics[self._metric_key(
          self._metric_name)].value_at_cutoffs.values.add()
      row.cutoff = int(cutoff)
      if isinstance(value, types.ValueWithConfidenceInterval):
        row.value = value.value
        row.bounded_value.value.value = value.value
        row.bounded_value.upper_bound.value = value.upper_bound
        row.bounded_value.lower_bound.value = value.lower_bound
      else:
        row.value = value
        row.bounded_value.value.value = value


@_export('precision_at_k')
class _PrecisionAtK(_PrecisionRecallAtK):
  """Metric that computes precision at K for classification models.

  Create a metric that computes precision at K.

  Predictions should be a dict containing the CLASSES key and PROBABILITIES
  keys. Predictions should have the same size for all examples. The model
  should NOT, for instance, produce 2 classes on one example and 4 classes on
  another example.

  Labels should be a string Tensor, or a SparseTensor whose dense form is
  a string Tensor whose entries are the corresponding labels. Note that the
  values of the CLASSES in the predictions and that of labels will be compared
  directly, so they should come from the "same vocabulary", so if predictions
  are class IDs, then labels should be class IDs, and so on.
  """

  def __init__(self,
               cutoffs: List[int],
               example_weight_key: Optional[Text] = None,
               target_prediction_keys: Optional[List[Text]] = None,
               labels_key: Optional[Text] = None,
               metric_tag: Optional[Text] = None,
               classes_key: Optional[Text] = None,
               probabilities_key: Optional[Text] = None):
    """Creates a metric that computes the precision at `k`.

    Args:
      cutoffs: List of `k` values at which to compute the precision and recall.
        Use a value of `k` = 0 to indicate that all predictions should be
        considered.
      example_weight_key: The optional key of the example weight column in the
        features_dict. If not given, all examples will be assumed to have a
        weight of 1.0.
      target_prediction_keys: Optional acceptable keys in predictions_dict in
        descending order of precedence.
      labels_key: Optionally, the key from labels_dict to use.
      metric_tag: If provided, a custom metric tag. Only necessary to
        disambiguate instances of the same metric on different predictions.
      classes_key: Optionally, the key from predictions that specifies classes.
      probabilities_key: Optionally, the key from predictions that specifies
        probabilities.
    """
    super(_PrecisionAtK,
          self).__init__(metric_keys.PRECISION_AT_K, cutoffs,
                         example_weight_key, target_prediction_keys, labels_key,
                         metric_tag, classes_key, probabilities_key)

  def populate_stats_and_pop(  # pylint: disable=useless-super-delegation
      self, combine_metrics: Dict[Text, Any],
      output_metrics: Dict[Text, metrics_pb2.MetricValue]) -> None:
    return super(_PrecisionAtK,
                 self).populate_stats_and_pop(combine_metrics, output_metrics)


@_export('recall_at_k')
class _RecallAtK(_PrecisionRecallAtK):
  """Metric that computes recall at K for classification models.

  Create a metric that computes recall at K.

  Predictions should be a dict containing the CLASSES key and PROBABILITIES
  keys. Predictions should have the same size for all examples. The model
  should NOT, for instance, produce 2 classes on one example and 4 classes on
  another example.

  Labels should be a string Tensor, or a SparseTensor whose dense form is
  a string Tensor whose entries are the corresponding labels. Note that the
  values of the CLASSES in the predictions and that of labels will be compared
  directly, so they should come from the "same vocabulary", so if predictions
  are class IDs, then labels should be class IDs, and so on.
  """

  def __init__(self,
               cutoffs: List[int],
               example_weight_key: Optional[Text] = None,
               target_prediction_keys: Optional[List[Text]] = None,
               labels_key: Optional[Text] = None,
               metric_tag: Optional[Text] = None,
               classes_key: Optional[Text] = None,
               probabilities_key: Optional[Text] = None):
    """Creates a metric that computes the recall at `k`.

    Args:
      cutoffs: List of `k` values at which to compute the precision and recall.
        Use a value of `k` = 0 to indicate that all predictions should be
        considered.
      example_weight_key: The optional key of the example weight column in the
        features_dict. If not given, all examples will be assumed to have a
        weight of 1.0.
      target_prediction_keys: Optional acceptable keys in predictions_dict in
        descending order of precedence.
      labels_key: Optionally, the key from labels_dict to use.
      metric_tag: If provided, a custom metric tag. Only necessary to
        disambiguate instances of the same metric on different predictions.
      classes_key: Optionally, the key from predictions that specifies classes.
      probabilities_key: Optionally, the key from predictions that specifies
        probabilities.
    """
    super(_RecallAtK,
          self).__init__(metric_keys.RECALL_AT_K, cutoffs, example_weight_key,
                         target_prediction_keys, labels_key, metric_tag,
                         classes_key, probabilities_key)

  def populate_stats_and_pop(  # pylint: disable=useless-super-delegation
      self, combine_metrics: Dict[Text, Any],
      output_metrics: Dict[Text, metrics_pb2.MetricValue]) -> None:
    return super(_RecallAtK,
                 self).populate_stats_and_pop(combine_metrics, output_metrics)
