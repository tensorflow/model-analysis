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
"""Binary confusion matrices."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Text, Callable

import apache_beam as beam
from tensorflow_model_analysis import config
from tensorflow_model_analysis.metrics import calibration_histogram
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util

DEFAULT_NUM_THRESHOLDS = calibration_histogram.DEFAULT_NUM_BUCKETS

DEFAULT_NUM_EXAMPLE_IDS = 100

BINARY_CONFUSION_MATRICES_NAME = '_binary_confusion_matrices'

Matrices = NamedTuple('Matrices', [('thresholds', List[float]),
                                   ('tp', List[float]), ('tn', List[float]),
                                   ('fp', List[float]), ('fn', List[float]),
                                   ('tp_examples', List[List[Text]]),
                                   ('tn_examples', List[List[Text]]),
                                   ('fp_examples', List[List[Text]]),
                                   ('fn_examples', List[List[Text]])])

_EPSILON = 1e-7


def _interpolated_thresholds(num_thresholds: int) -> List[float]:
  """Returns thresholds interpolated over a range equal to num_thresholds."""
  # The interpolation strategy used here matches that used by keras for AUC.
  thresholds = [
      (i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)
  ]
  return [-_EPSILON] + thresholds + [1.0 + _EPSILON]


def binary_confusion_matrices(
    num_thresholds: Optional[int] = None,
    thresholds: Optional[List[float]] = None,
    name: Optional[Text] = None,
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    output_name: Text = '',
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None,
    use_histogram: Optional[bool] = None,
    extract_label_prediction_and_weight: Optional[Callable[
        ..., Any]] = metric_util.to_label_prediction_example_weight,
    preprocessor: Optional[Callable[..., Any]] = None,
    example_id_key: Optional[Text] = None,
    example_ids_count: Optional[int] = None,
    fractional_labels: float = True) -> metric_types.MetricComputations:
  """Returns metric computations for computing binary confusion matrices.

  Args:
    num_thresholds: Number of thresholds to use. Thresholds will be calculated
      using linear interpolation between 0.0 and 1.0 with equidistant values and
      bondardaries at -epsilon and 1.0+epsilon. Values must be > 0. Only one of
      num_thresholds or thresholds should be used. If used, num_thresholds must
      be > 1.
    thresholds: A specific set of thresholds to use. The caller is responsible
      for marking the boundaries with +/-epsilon if desired. Only one of
      num_thresholds or thresholds should be used. For metrics computed at top k
      this may be a single negative threshold value (i.e. -inf).
    name: Metric name.
    eval_config: Eval config.
    model_name: Optional model name (if multi-model evaluation).
    output_name: Optional output name (if multi-output model type).
    sub_key: Optional sub key.
    aggregation_type: Optional aggregation type.
    class_weights: Optional class weights to apply to multi-class / multi-label
      labels and predictions prior to flattening (when micro averaging is used).
    use_histogram: If true, matrices will be derived from calibration
      histograms.
    extract_label_prediction_and_weight: User-provided function argument that
      yields label, prediction, and example weights for use in calculations
      (relevant only when use_histogram flag is not true).
    preprocessor: User-provided preprocessor for including additional extracts
      in StandardMetricInputs (relevant only when use_histogram flag is not
      true).
    example_id_key: Feature key containing example id (relevant only when
      use_histogram flag is not true).
    example_ids_count: Max number of example ids to be extracted for false
      positives and false negatives (relevant only when use_histogram flag is
      not true).
    fractional_labels: If true, each incoming tuple of (label, prediction, and
      example weight) will be split into two tuples as follows (where l, p, w
      represent the resulting label, prediction, and example weight values): (1)
        l = 0.0, p = prediction, and w = example_weight * (1.0 - label) (2) l =
        1.0, p = prediction, and w = example_weight * label If enabled, an
        exception will be raised if labels are not within [0, 1]. The
        implementation is such that tuples associated with a weight of zero are
        not yielded. This means it is safe to enable fractional_labels even when
        the labels only take on the values of 0.0 or 1.0.

  Raises:
    ValueError: If both num_thresholds and thresholds are set at the same time.
  """
  if num_thresholds is not None and thresholds is not None:
    raise ValueError(
        'only one of thresholds or num_thresholds can be set at a time')
  if num_thresholds is None and thresholds is None:
    num_thresholds = DEFAULT_NUM_THRESHOLDS
  # Keras AUC turns num_thresholds parameters into thresholds which circumvents
  # sharing of settings. If the thresholds match the interpolated version of the
  # thresholds then reset back to num_thresholds.
  if (name is None and thresholds and
      thresholds == _interpolated_thresholds(len(thresholds))):
    num_thresholds = len(thresholds)
    thresholds = None
  if num_thresholds is not None:
    if num_thresholds <= 1:
      raise ValueError('num_thresholds must be > 1')
    # The interpolation strategy used here matches that used by keras for AUC.
    thresholds = _interpolated_thresholds(num_thresholds)
    if name is None:
      name = '{}_{}'.format(BINARY_CONFUSION_MATRICES_NAME, num_thresholds)
  elif name is None:
    name = '{}_{}'.format(BINARY_CONFUSION_MATRICES_NAME, list(thresholds))

  key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key)

  computations = []
  metric_key = None

  if use_histogram is None:
    use_histogram = (
        num_thresholds is not None or
        (len(thresholds) == 1 and thresholds[0] < 0))

  if use_histogram:
    # Use calibration histogram to calculate matrices. For efficiency (unless
    # all predictions are matched - i.e. thresholds <= 0) we will assume that
    # other metrics will make use of the calibration histogram and re-use the
    # default histogram for the given model_name/output_name/sub_key. This is
    # also required to get accurate counts at the threshold boundaries. If this
    # becomes an issue, then calibration histogram can be updated to support
    # non-linear boundaries.
    computations = calibration_histogram.calibration_histogram(
        eval_config=eval_config,
        num_buckets=(
            # For precision/recall_at_k were a single large negative threshold
            # is used, we only need one bucket. Note that the histogram will
            # actually have 2 buckets: one that we set (which handles
            # predictions > -1.0) and a default catch-all bucket (i.e. bucket 0)
            # that the histogram creates for large negative predictions (i.e.
            # predictions <= -1.0).
            1 if len(thresholds) == 1 and thresholds[0] <= 0 else None),
        model_name=model_name,
        output_name=output_name,
        sub_key=sub_key,
        aggregation_type=aggregation_type,
        class_weights=class_weights)
    metric_key = computations[-1].keys[-1]
  else:
    computations = _binary_confusion_matrix_computation(
        eval_config=eval_config,
        thresholds=thresholds,
        model_name=model_name,
        output_name=output_name,
        sub_key=sub_key,
        extract_label_prediction_and_weight=extract_label_prediction_and_weight,
        preprocessor=preprocessor,
        example_id_key=example_id_key,
        example_ids_count=example_ids_count,
        aggregation_type=aggregation_type,
        class_weights=class_weights,
        fractional_labels=fractional_labels)
    metric_key = computations[-1].keys[-1]

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.MetricKey, Matrices]:
    """Returns binary confusion matrices."""
    matrices = None
    if use_histogram:
      if len(thresholds) == 1 and thresholds[0] < 0:
        # This case is used when all positive prediction values are relevant
        # matches (e.g. when calculating top_k for precision/recall where the
        # non-top_k values are expected to have been set to float('-inf')).
        histogram = metrics[metric_key]
      else:
        # Calibration histogram uses intervals of the form [start, end) where
        # the prediction >= start. The confusion matrices want intervals of the
        # form (start, end] where the prediction > start. Add a small epsilon so
        # that >= checks don't match. This correction shouldn't be needed in
        # practice but allows for correctness in small tests.
        rebin_thresholds = [t + _EPSILON if t != 0 else t for t in thresholds]
        if thresholds[0] >= 0:
          # Add -epsilon bucket to account for differences in histogram vs
          # confusion matrix intervals mentioned above. If the epsilon bucket is
          # missing the false negatives and false positives will be 0 for the
          # first threshold.
          rebin_thresholds = [-_EPSILON] + rebin_thresholds
        if thresholds[-1] < 1.0:
          # If the last threshold < 1.0, then add a fence post at 1.0 + epsilon
          # othewise true negatives and true positives will be overcounted.
          rebin_thresholds = rebin_thresholds + [1.0 + _EPSILON]
        histogram = calibration_histogram.rebin(rebin_thresholds,
                                                metrics[metric_key])
      matrices = _historgram_to_binary_confusion_matrices(thresholds, histogram)
    else:
      matrices = _matrix_to_binary_confusion_matrices(thresholds,
                                                      metrics[metric_key])
    return {key: matrices}

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations.append(derived_computation)
  return computations


def _historgram_to_binary_confusion_matrices(
    thresholds: List[float],
    histogram: calibration_histogram.Histogram) -> Matrices:
  """Converts histogram to binary confusion matrices."""
  # tp(i) - sum of positive labels >= bucket i
  # fp(i) - sum of negative labels >= bucket i
  # fn(i) - sum of positive labels < bucket i
  # tn(i) - sum of negative labels < bucket i
  n = len(histogram)
  tp = [0.0] * n
  fp = [0.0] * n
  tn = [0.0] * n
  fn = [0.0] * n
  for i in range(n):
    start = i
    end = n - i - 1
    start_pos = histogram[start].weighted_labels
    start_neg = (
        histogram[start].weighted_examples - histogram[start].weighted_labels)
    end_pos = histogram[end].weighted_labels
    end_neg = (
        histogram[end].weighted_examples - histogram[end].weighted_labels)
    tp[end] = tp[end + 1] + end_pos if end < n - 1 else end_pos
    fp[end] = fp[end + 1] + end_neg if end < n - 1 else end_neg
    if start + 1 < n:
      tn[start + 1] = tn[start] + start_neg
      fn[start + 1] = fn[start] + start_pos
  # Check if need to remove -epsilon bucket (or reset back to 1 bucket).
  threshold_offset = 0
  if (thresholds[0] >= 0 or len(thresholds) == 1) and len(histogram) > 1:
    threshold_offset = 1
  tp = tp[threshold_offset:threshold_offset + len(thresholds)]
  fp = fp[threshold_offset:threshold_offset + len(thresholds)]
  tn = tn[threshold_offset:threshold_offset + len(thresholds)]
  fn = fn[threshold_offset:threshold_offset + len(thresholds)]
  # We sum all values >= bucket i, but TP/FP values greater that 1.0 + EPSILON
  # should be 0.0. The FN/TN above 1.0 + _EPSILON should also be adjusted to
  # match the TP/FP values at the start.
  for i, t in enumerate(thresholds):
    if t >= 1.0 + _EPSILON:
      tp[i] = 0.0
      fp[i] = 0.0
      fn[i] = tp[0]
      tn[i] = fp[0]
  return Matrices(thresholds, tp, tn, fp, fn, [], [], [], [])


_BINARY_CONFUSION_MATRIX_NAME = '_binary_confusion_matrix'

Matrix = NamedTuple('Matrix', [('tp', float), ('tn', float), ('fp', float),
                               ('fn', float), ('tp_examples', List[Text]),
                               ('tn_examples', List[Text]),
                               ('fp_examples', List[Text]),
                               ('fn_examples', List[Text])])

MatrixAccumulator = Dict[float, Matrix]


def _binary_confusion_matrix_computation(
    thresholds: List[float],
    name: Optional[Text] = None,
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    output_name: Text = '',
    sub_key: Optional[metric_types.SubKey] = None,
    extract_label_prediction_and_weight: Optional[Callable[
        ..., Any]] = metric_util.to_label_prediction_example_weight,
    preprocessor: Optional[Callable[..., Any]] = None,
    example_id_key: Optional[Text] = None,
    example_ids_count: Optional[int] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None,
    fractional_labels: float = True) -> metric_types.MetricComputations:
  """Returns metric computations for computing binary confusion matrix."""
  if example_ids_count is None:
    example_ids_count = DEFAULT_NUM_EXAMPLE_IDS

  # To generate unique name for each computation
  if name is None:
    name = '{}_{}_{}_{}'.format(_BINARY_CONFUSION_MATRIX_NAME, list(thresholds),
                                example_id_key, example_ids_count)

  key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key)

  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessor=preprocessor,
          combiner=_BinaryConfusionMatrixCombiner(
              key=key,
              eval_config=eval_config,
              thresholds=thresholds,
              extract_label_prediction_and_weight=extract_label_prediction_and_weight,
              example_id_key=example_id_key,
              example_ids_count=example_ids_count,
              aggregation_type=aggregation_type,
              class_weights=class_weights,
              fractional_labels=fractional_labels))
  ]


class _BinaryConfusionMatrixCombiner(beam.CombineFn):
  """Computes binary confusion matrix."""

  def __init__(self, key: metric_types.MetricKey,
               eval_config: Optional[config.EvalConfig],
               thresholds: List[float],
               extract_label_prediction_and_weight: Callable[..., Any],
               example_id_key: Optional[Text], example_ids_count: float,
               aggregation_type: Optional[metric_types.AggregationType],
               class_weights: Optional[Dict[int,
                                            float]], fractional_labels: float):
    self._key = key
    self._eval_config = eval_config
    self._thresholds = thresholds
    self._extract_label_prediction_and_weight = extract_label_prediction_and_weight
    self._example_id_key = example_id_key
    self._example_ids_count = example_ids_count
    self._aggregation_type = aggregation_type
    self._class_weights = class_weights
    self._fractional_labels = fractional_labels

  def _merge_example_ids(self, list_1: List[Text],
                         list_2: List[Text]) -> List[Text]:
    result = list_1[:self._example_ids_count]
    result.extend(list_2[:self._example_ids_count - len(result)])
    return result

  def _merge_matrix(self, result: MatrixAccumulator, threshold: float,
                    matrix: Matrix):
    if threshold not in result:
      return matrix

    return Matrix(
        tp=result[threshold].tp + matrix.tp,
        tn=result[threshold].tn + matrix.tn,
        fp=result[threshold].fp + matrix.fp,
        fn=result[threshold].fn + matrix.fn,
        tp_examples=self._merge_example_ids(result[threshold].tp_examples,
                                            matrix.tp_examples),
        tn_examples=self._merge_example_ids(result[threshold].tn_examples,
                                            matrix.tn_examples),
        fp_examples=self._merge_example_ids(result[threshold].fp_examples,
                                            matrix.fp_examples),
        fn_examples=self._merge_example_ids(result[threshold].fn_examples,
                                            matrix.fn_examples))

  def create_accumulator(self) -> MatrixAccumulator:
    return {}

  def add_input(
      self, accumulator: MatrixAccumulator,
      element: metric_types.StandardMetricInputs) -> MatrixAccumulator:
    example_id = None
    if self._example_id_key and self._example_id_key in element.features:
      example_id = element.features[self._example_id_key]

    labels = []
    predictions = []
    example_weights = []

    for label, prediction, example_weight in self._extract_label_prediction_and_weight(
        element,
        eval_config=self._eval_config,
        model_name=self._key.model_name,
        output_name=self._key.output_name,
        sub_key=self._key.sub_key,
        fractional_labels=self._fractional_labels,
        flatten=True,
        aggregation_type=self._aggregation_type,
        class_weights=self._class_weights):
      example_weights.append(float(example_weight))
      labels.append(float(label))
      predictions.append(float(prediction))
    result = accumulator
    for threshold in self._thresholds:
      tp = 0.0
      tn = 0.0
      fp = 0.0
      fn = 0.0
      tp_example = None
      tn_example = None
      fp_example = None
      fn_example = None
      for i, _ in enumerate(labels):
        if (labels[i] == 1.0
            if self._fractional_labels else labels[i] > threshold):
          if predictions[i] > threshold:
            tp += example_weights[i]
            tp_example = example_id
          else:
            fn += example_weights[i]
            fn_example = example_id
        else:
          if predictions[i] > threshold:
            fp += example_weights[i]
            fp_example = example_id
          else:
            tn += example_weights[i]
            tn_example = example_id

      result[threshold] = self._merge_matrix(
          result, threshold,
          Matrix(
              tp=tp,
              tn=tn,
              fp=fp,
              fn=fn,
              tp_examples=[tp_example] if tp_example else [],
              tn_examples=[tn_example] if tn_example else [],
              fp_examples=[fp_example] if fp_example else [],
              fn_examples=[fn_example] if fn_example else []))

    return result

  def merge_accumulators(
      self, accumulators: Iterable[MatrixAccumulator]) -> MatrixAccumulator:
    accumulators = iter(accumulators)
    result = next(accumulators)
    for accumulator in accumulators:
      for threshold in self._thresholds:
        if threshold not in accumulator:
          continue
        result[threshold] = self._merge_matrix(result, threshold,
                                               accumulator[threshold])
    return result

  def extract_output(
      self, accumulator: MatrixAccumulator
  ) -> Dict[metric_types.MetricKey, MatrixAccumulator]:
    for threshold in self._thresholds:
      if threshold not in accumulator:
        accumulator[threshold] = Matrix(
            tp=0.0,
            tn=0.0,
            fp=0.0,
            fn=0.0,
            tp_examples=[],
            tn_examples=[],
            fp_examples=[],
            fn_examples=[])
    return {self._key: accumulator}


def _matrix_to_binary_confusion_matrices(
    thresholds: List[float], matrices: MatrixAccumulator) -> Matrices:
  """Converts MatrixAccumulator to binary confusion matrices."""
  result = Matrices(
      thresholds=[],
      tp=[],
      tn=[],
      fp=[],
      fn=[],
      tp_examples=[],
      tn_examples=[],
      fp_examples=[],
      fn_examples=[])
  for threshold in thresholds:
    result.thresholds.append(threshold)
    result.tp.append(matrices[threshold].tp)
    result.tn.append(matrices[threshold].tn)
    result.fp.append(matrices[threshold].fp)
    result.fn.append(matrices[threshold].fn)
    result.tp_examples.append(matrices[threshold].tp_examples)
    result.tn_examples.append(matrices[threshold].tn_examples)
    result.fp_examples.append(matrices[threshold].fp_examples)
    result.fn_examples.append(matrices[threshold].fn_examples)

  return result
