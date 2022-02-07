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

from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

import apache_beam as beam
from tensorflow_model_analysis import types
from tensorflow_model_analysis.metrics import calibration_histogram
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.proto import metrics_for_slice_pb2

DEFAULT_NUM_THRESHOLDS = calibration_histogram.DEFAULT_NUM_BUCKETS
_KERAS_DEFAULT_NUM_THRESHOLDS = 200

DEFAULT_NUM_EXAMPLE_IDS = 100

BINARY_CONFUSION_MATRICES_NAME = '_binary_confusion_matrices'
BINARY_CONFUSION_EXAMPLES_NAME = '_binary_confusion_examples'


class Examples(
    NamedTuple('Examples', [('thresholds', List[float]),
                            ('tp_examples', List[List[str]]),
                            ('tn_examples', List[List[str]]),
                            ('fp_examples', List[List[str]]),
                            ('fn_examples', List[List[str]])])):
  """A set of examples for each binary confusion case at each threshold."""


class Matrices(types.StructuredMetricValue,
               NamedTuple('Matrices', [('thresholds', List[float]),
                                       ('tp', List[float]), ('tn', List[float]),
                                       ('fp', List[float]),
                                       ('fn', List[float])])):
  """A class representing a set of binary confusion matrices at thresholds.

  For each threshold, in addition to the count of examples per prediction and
  label, this class also contains a sample of raw examples. Threshold values are
  sorted, and the entries within  tp[i], tn[i], fp[i], and fn[i] correspond to
  thresholds[i].
  """

  def _apply_binary_op_elementwise(self, other: 'Matrices',
                                   op: Callable[[float, float], float]):
    """Applies an operator elementwise on self and `other` matrices."""
    tp, tn, fp, fn = [], [], [], []
    self_idx, other_idx = 0, 0
    merged_thresholds = []
    while True:
      if (self_idx < len(self.thresholds) and
          other_idx < len(other.thresholds) and
          self.thresholds[self_idx] == other.thresholds[other_idx]):
        # threshold present in both, advance both indices
        merged_thresholds.append(self.thresholds[self_idx])
        tp.append(op(self.tp[self_idx], other.tp[other_idx]))
        tn.append(op(self.tn[self_idx], other.tn[other_idx]))
        fp.append(op(self.fp[self_idx], other.fp[other_idx]))
        fn.append(op(self.fn[self_idx], other.fn[other_idx]))
        self_idx += 1
        other_idx += 1
      elif (self_idx < len(self.thresholds) and
            (other_idx >= len(other.thresholds) or
             self.thresholds[self_idx] < other.thresholds[other_idx])):
        # threshold present in self but missing from other, use default values
        # for other and advance self_idx
        merged_thresholds.append(self.thresholds[self_idx])
        tp.append(op(self.tp[self_idx], 0))
        tn.append(op(self.tn[self_idx], 0))
        fp.append(op(self.fp[self_idx], 0))
        fn.append(op(self.fn[self_idx], 0))
        self_idx += 1
      elif (other_idx < len(other.thresholds) and
            (self_idx >= len(self.thresholds) or
             other.thresholds[self_idx] < self.thresholds[other_idx])):
        # threshold present in other but missing from self, use default values
        # for self and advance other_idx
        merged_thresholds.append(other.thresholds[other_idx])
        tp.append(op(0, other.tp[other_idx]))
        tn.append(op(0, other.tn[other_idx]))
        fp.append(op(0, other.fp[other_idx]))
        fn.append(op(0, other.fn[other_idx]))
        other_idx += 1
      else:
        assert (self_idx >= len(self.thresholds) and
                other_idx >= len(other.thresholds))
        break
    return Matrices(thresholds=merged_thresholds, tp=tp, tn=tn, fp=fp, fn=fn)

  def _apply_binary_op_broadcast(self, other: float,
                                 op: Callable[[float, float], float]):
    """Applies an operator on each element and the provided float."""
    return Matrices(
        thresholds=self.thresholds,
        tp=[op(tp, other) for tp in self.tp],
        tn=[op(tn, other) for tn in self.tn],
        fp=[op(fp, other) for fp in self.fp],
        fn=[op(fn, other) for fn in self.fn])

  def to_proto(self) -> metrics_for_slice_pb2.MetricValue:
    """Converts matrices into ConfusionMatrixAtThresholds proto.

    If precision or recall are undefined then 1.0 and 0.0 will be used.

    Returns:
      A MetricValue proto containing a ConfusionMatrixAtThresholds proto.
    """
    result = metrics_for_slice_pb2.MetricValue()
    confusion_matrix_at_thresholds_proto = result.confusion_matrix_at_thresholds
    for i, threshold in enumerate(self.thresholds):
      precision = 1.0
      if self.tp[i] + self.fp[i] > 0:
        precision = self.tp[i] / (self.tp[i] + self.fp[i])
      recall = 0.0
      if self.tp[i] + self.fn[i] > 0:
        recall = self.tp[i] / (self.tp[i] + self.fn[i])
      confusion_matrix_at_thresholds_proto.matrices.add(
          threshold=round(threshold, 6),
          true_positives=self.tp[i],
          false_positives=self.fp[i],
          true_negatives=self.tn[i],
          false_negatives=self.fn[i],
          precision=precision,
          recall=recall)
    return result


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
    name: Optional[str] = None,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: str = '',
    output_name: str = '',
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None,
    example_weighted: bool = False,
    use_histogram: Optional[bool] = None,
    extract_label_prediction_and_weight: Optional[Callable[
        ..., Any]] = metric_util.to_label_prediction_example_weight,
    preprocessor: Optional[Callable[..., Any]] = None,
    examples_name: Optional[str] = None,
    example_id_key: Optional[str] = None,
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
    name: Metric name containing binary_confusion_matrices.Matrices.
    eval_config: Eval config.
    model_name: Optional model name (if multi-model evaluation).
    output_name: Optional output name (if multi-output model type).
    sub_key: Optional sub key.
    aggregation_type: Optional aggregation type.
    class_weights: Optional class weights to apply to multi-class / multi-label
      labels and predictions prior to flattening (when micro averaging is used).
    example_weighted: True if example weights should be applied.
    use_histogram: If true, matrices will be derived from calibration
      histograms.
    extract_label_prediction_and_weight: User-provided function argument that
      yields label, prediction, and example weights for use in calculations
      (relevant only when use_histogram flag is not true).
    preprocessor: User-provided preprocessor for including additional extracts
      in StandardMetricInputs (relevant only when use_histogram flag is not
      true).
    examples_name: Metric name containing binary_confusion_matrices.Examples.
      (relevant only when use_histogram flag is not true and example_id_key is
      set).
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
  # TF v1 Keras AUC turns num_thresholds parameters into thresholds which
  # circumvents sharing of settings. If the thresholds match the interpolated
  # version of the thresholds then reset back to num_thresholds.
  if thresholds:
    if (not num_thresholds and
        thresholds == _interpolated_thresholds(len(thresholds))):
      num_thresholds = len(thresholds)
      thresholds = None
    elif (num_thresholds
          in (DEFAULT_NUM_THRESHOLDS, _KERAS_DEFAULT_NUM_THRESHOLDS) and
          len(thresholds) == num_thresholds - 2):
      thresholds = None
  if num_thresholds is not None and thresholds is not None:
    raise ValueError(
        'only one of thresholds or num_thresholds can be set at a time: '
        f'num_thesholds={num_thresholds}, thresholds={thresholds}, '
        f'len(thresholds)={len(thresholds)})')
  if num_thresholds is None and thresholds is None:
    num_thresholds = DEFAULT_NUM_THRESHOLDS
  if num_thresholds is not None:
    if num_thresholds <= 1:
      raise ValueError('num_thresholds must be > 1')
    # The interpolation strategy used here matches that used by keras for AUC.
    thresholds = _interpolated_thresholds(num_thresholds)
    thresholds_name_part = str(num_thresholds)
  else:
    thresholds_name_part = str(list(thresholds))

  if use_histogram is None:
    use_histogram = (
        num_thresholds is not None or
        (len(thresholds) == 1 and thresholds[0] < 0))

  if use_histogram and (examples_name or example_id_key or example_ids_count):
    raise ValueError('Example sampling is only performed when not using the '
                     'histogram computation. However, use_histogram is true '
                     f'and one of examples_name ("{examples_name}"), '
                     f'examples_id_key ("{example_id_key}"), '
                     f'or example_ids_count ({example_ids_count}) was '
                     'provided, which will have no effect.')

  if examples_name and not (example_id_key and example_ids_count):
    raise ValueError('examples_name provided but either example_id_key or '
                     'example_ids_count was not. Examples will only be '
                     'returned when both example_id_key and '
                     'example_ids_count are provided, and when the '
                     'non-histogram computation is used. '
                     f'example_id_key: "{example_id_key}" '
                     f'example_ids_count: {example_ids_count}')

  if name is None:
    name = f'{BINARY_CONFUSION_MATRICES_NAME}_{thresholds_name_part}'
  if examples_name is None:
    examples_name = f'{BINARY_CONFUSION_EXAMPLES_NAME}_{thresholds_name_part}'
  matrices_key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      example_weighted=example_weighted)
  examples_key = metric_types.MetricKey(
      name=examples_name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      example_weighted=example_weighted)

  computations = []
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
        class_weights=class_weights,
        example_weighted=example_weighted)
    input_metric_key = computations[-1].keys[-1]
    output_metric_keys = [matrices_key]
  else:
    if bool(example_ids_count) != bool(example_id_key):
      raise ValueError('Both of example_ids_count and example_id_key must be '
                       f'set, but got example_id_key: "{example_id_key}" and '
                       f'example_ids_count: {example_ids_count}.')
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
        example_weighted=example_weighted,
        fractional_labels=fractional_labels)
    input_metric_key = computations[-1].keys[-1]
    # matrices_key is last for backwards compatibility with code that:
    #   1) used this computation as an input for a derived computation
    #   2) only accessed the matrix counts
    #   3) used computations[-1].keys[-1] to access the input key
    output_metric_keys = [examples_key, matrices_key]

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.MetricKey, Union[Matrices, Examples]]:
    """Returns binary confusion matrices."""
    matrices = None
    if use_histogram:
      if len(thresholds) == 1 and thresholds[0] < 0:
        # This case is used when all positive prediction values are relevant
        # matches (e.g. when calculating top_k for precision/recall where the
        # non-top_k values are expected to have been set to float('-inf')).
        histogram = metrics[input_metric_key]
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
                                                metrics[input_metric_key])
      matrices = _histogram_to_binary_confusion_matrices(thresholds, histogram)
      return {matrices_key: matrices}
    else:
      matrices, examples = _accumulator_to_matrices_and_examples(
          thresholds, metrics[input_metric_key])
      return {matrices_key: matrices, examples_key: examples}

  derived_computation = metric_types.DerivedMetricComputation(
      keys=output_metric_keys, result=result)
  computations.append(derived_computation)
  return computations


def _histogram_to_binary_confusion_matrices(
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
  return Matrices(thresholds, tp, tn, fp, fn)


_BINARY_CONFUSION_MATRIX_NAME = '_binary_confusion_matrix'

Matrix = NamedTuple('Matrix', [('tp', float), ('tn', float), ('fp', float),
                               ('fn', float)])

_ThresholdEntry = NamedTuple('_ThresholdEntry', [('matrix', Matrix),
                                                 ('tp_examples', List[str]),
                                                 ('tn_examples', List[str]),
                                                 ('fp_examples', List[str]),
                                                 ('fn_examples', List[str])])

MatrixAccumulator = Dict[float, _ThresholdEntry]


def _binary_confusion_matrix_computation(
    thresholds: List[float],
    name: Optional[str] = None,
    eval_config: Optional[config_pb2.EvalConfig] = None,
    model_name: str = '',
    output_name: str = '',
    sub_key: Optional[metric_types.SubKey] = None,
    extract_label_prediction_and_weight: Optional[Callable[
        ..., Any]] = metric_util.to_label_prediction_example_weight,
    preprocessor: Optional[Callable[..., Any]] = None,
    example_id_key: Optional[str] = None,
    example_ids_count: Optional[int] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None,
    example_weighted: bool = False,
    fractional_labels: float = True) -> metric_types.MetricComputations:
  """Returns metric computations for computing binary confusion matrix."""
  if example_ids_count is None:
    example_ids_count = DEFAULT_NUM_EXAMPLE_IDS

  # To generate unique name for each computation
  if name is None:
    name = (f'{_BINARY_CONFUSION_MATRIX_NAME}_{list(thresholds)}_'
            f'{example_id_key}_{example_ids_count}')

  key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      example_weighted=example_weighted)

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
              example_weighted=example_weighted,
              fractional_labels=fractional_labels))
  ]


class _BinaryConfusionMatrixCombiner(beam.CombineFn):
  """Computes binary confusion matrix."""

  def __init__(self, key: metric_types.MetricKey,
               eval_config: Optional[config_pb2.EvalConfig],
               thresholds: List[float],
               extract_label_prediction_and_weight: Callable[..., Any],
               example_id_key: Optional[str], example_ids_count: float,
               aggregation_type: Optional[metric_types.AggregationType],
               class_weights: Optional[Dict[int, float]],
               example_weighted: bool, fractional_labels: float):
    self._key = key
    self._eval_config = eval_config
    self._thresholds = thresholds
    self._extract_label_prediction_and_weight = extract_label_prediction_and_weight
    self._example_id_key = example_id_key
    self._example_ids_count = example_ids_count
    self._aggregation_type = aggregation_type
    self._class_weights = class_weights
    self._example_weighted = example_weighted
    self._fractional_labels = fractional_labels

  def _merge_example_ids(self, list_1: List[str],
                         list_2: List[str]) -> List[str]:
    result = list_1[:self._example_ids_count]
    result.extend(list_2[:self._example_ids_count - len(result)])
    return result

  def _merge_entry(self, result: MatrixAccumulator, threshold: float,
                   entry: _ThresholdEntry):
    if threshold not in result:
      return entry

    return _ThresholdEntry(
        matrix=Matrix(
            tp=result[threshold].matrix.tp + entry.matrix.tp,
            tn=result[threshold].matrix.tn + entry.matrix.tn,
            fp=result[threshold].matrix.fp + entry.matrix.fp,
            fn=result[threshold].matrix.fn + entry.matrix.fn),
        tp_examples=self._merge_example_ids(result[threshold].tp_examples,
                                            entry.tp_examples),
        tn_examples=self._merge_example_ids(result[threshold].tn_examples,
                                            entry.tn_examples),
        fp_examples=self._merge_example_ids(result[threshold].fp_examples,
                                            entry.fp_examples),
        fn_examples=self._merge_example_ids(result[threshold].fn_examples,
                                            entry.fn_examples))

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
        class_weights=self._class_weights,
        example_weighted=self._example_weighted):
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

      result[threshold] = self._merge_entry(
          result, threshold,
          _ThresholdEntry(
              Matrix(tp=tp, tn=tn, fp=fp, fn=fn),
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
        result[threshold] = self._merge_entry(result, threshold,
                                              accumulator[threshold])
    return result

  def extract_output(
      self, accumulator: MatrixAccumulator
  ) -> Dict[metric_types.MetricKey, MatrixAccumulator]:
    for threshold in self._thresholds:
      if threshold not in accumulator:
        accumulator[threshold] = _ThresholdEntry(
            Matrix(tp=0.0, tn=0.0, fp=0.0, fn=0.0),
            tp_examples=[],
            tn_examples=[],
            fp_examples=[],
            fn_examples=[])
    return {self._key: accumulator}


def _accumulator_to_matrices_and_examples(
    thresholds: List[float],
    acc: MatrixAccumulator) -> Tuple[Matrices, Examples]:
  """Converts MatrixAccumulator to binary confusion matrices."""
  matrices = Matrices(thresholds=[], tp=[], tn=[], fp=[], fn=[])
  examples = Examples(
      thresholds=[],
      tp_examples=[],
      tn_examples=[],
      fp_examples=[],
      fn_examples=[])
  for threshold in thresholds:
    matrices.thresholds.append(threshold)
    matrices.tp.append(acc[threshold].matrix.tp)
    matrices.tn.append(acc[threshold].matrix.tn)
    matrices.fp.append(acc[threshold].matrix.fp)
    matrices.fn.append(acc[threshold].matrix.fn)

    examples.thresholds.append(threshold)
    examples.tp_examples.append(acc[threshold].tp_examples)
    examples.tn_examples.append(acc[threshold].tn_examples)
    examples.fp_examples.append(acc[threshold].fp_examples)
    examples.fn_examples.append(acc[threshold].fn_examples)
  return matrices, examples
