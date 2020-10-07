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
"""Confusion matrix metrics."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import abc
import math

from typing import Any, Dict, List, Optional, Text, Union

import numpy as np
import six
from tensorflow_model_analysis import config
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import metrics_for_slice_pb2

SPECIFICITY_NAME = 'specificity'
FALL_OUT_NAME = 'fall_out'
MISS_RATE_NAME = 'miss_rate'
NEGATIVE_PREDICTIVE_VALUE_NAME = 'negative_predictive_value'
FALSE_DISCOVERY_RATE_NAME = 'false_discovery_rate'
FALSE_OMISSION_RATE_NAME = 'false_omission_rate'
PREVALENCE_NAME = 'prevalence'
PREVALENCE_THRESHOLD_NAME = 'prevalence_threshold'
THREAT_SCORE_NAME = 'threat_score'
BALANCED_ACCURACY_NAME = 'balanced_accuracy'
F1_SCORE_NAME = 'f1_score'
MATTHEWS_CORRELATION_COEFFICENT_NAME = 'matthews_correlation_coefficient'
FOWLKES_MALLOWS_INDEX_NAME = 'fowlkes_mallows_index'
INFORMEDNESS_NAME = 'informedness'
MARKEDNESS_NAME = 'markedness'
POSITIVE_LIKELIHOOD_RATIO_NAME = 'positive_likelihood_ratio'
NEGATIVE_LIKELIHOOD_RATIO_NAME = 'negative_likelihood_ratio'
DIAGNOSTIC_ODDS_RATIO_NAME = 'diagnostic_odds_ratio'
CONFUSION_MATRIX_AT_THRESHOLDS_NAME = 'confusion_matrix_at_thresholds'


class ConfusionMatrixMetric(
    six.with_metaclass(abc.ABCMeta, metric_types.Metric)):
  """Base for confusion matrix metrics."""

  def __init__(self, name: Text, thresholds: Optional[List[float]] = None):
    """Initializes confusion matrix metric.

    Args:
      name: Metric name.
      thresholds: Thresholds to use for specificity. Defaults to [0.5].
    """
    super(ConfusionMatrixMetric, self).__init__(
        metric_util.merge_per_key_computations(self._metric_computation),
        thresholds=thresholds,
        name=name)

  @abc.abstractmethod
  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    """Function for computing metric value from TP, TN, FP, FN values."""
    raise NotImplementedError('Must be implemented in subclasses.')

  def _metric_computation(
      self,
      thresholds: Optional[List[float]] = None,
      name: Text = '',
      eval_config: Optional[config.EvalConfig] = None,
      model_name: Text = '',
      output_name: Text = '',
      sub_key: Optional[metric_types.SubKey] = None,
      aggregation_type: Optional[metric_types.AggregationType] = None,
      class_weights: Optional[Dict[int, float]] = None
  ) -> metric_types.MetricComputations:
    """Returns metric computations for specificity."""
    key = metric_types.MetricKey(
        name=name,
        model_name=model_name,
        output_name=output_name,
        sub_key=sub_key)

    if not thresholds:
      thresholds = [0.5]

    # Make sure matrices are calculated.
    matrices_computations = binary_confusion_matrices.binary_confusion_matrices(
        eval_config=eval_config,
        model_name=model_name,
        output_name=output_name,
        sub_key=sub_key,
        aggregation_type=aggregation_type,
        class_weights=class_weights,
        thresholds=thresholds)
    matrices_key = matrices_computations[-1].keys[-1]

    def result(
        metrics: Dict[metric_types.MetricKey, Any]
    ) -> Dict[metric_types.MetricKey, Union[float, np.ndarray]]:
      matrices = metrics[matrices_key]
      values = []
      for i in range(len(thresholds)):
        values.append(
            self.result(matrices.tp[i], matrices.tn[i], matrices.fp[i],
                        matrices.fn[i]))
      return {key: values[0] if len(thresholds) == 1 else np.array(values)}

    derived_computation = metric_types.DerivedMetricComputation(
        keys=[key], result=result)
    computations = matrices_computations
    computations.append(derived_computation)
    return computations


def _pos_sqrt(value: float) -> float:
  """Returns sqrt of value or raises ValueError if negative."""
  if value < 0:
    raise ValueError('Attempt to take sqrt of negative value: {}'.format(value))
  return math.sqrt(value)


class Specificity(ConfusionMatrixMetric):
  """Specificity (TNR) or selectivity."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = SPECIFICITY_NAME):
    """Initializes specificity metric.

    Args:
      thresholds: Thresholds to use for specificity. Defaults to [0.5].
      name: Metric name.
    """
    super(Specificity, self).__init__(name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    del tp, fn
    denominator = tn + fp
    if denominator > 0.0:
      return tn / denominator
    else:
      return float('nan')


metric_types.register_metric(Specificity)


class FallOut(ConfusionMatrixMetric):
  """Fall-out (FPR)."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = FALL_OUT_NAME):
    """Initializes fall-out metric.

    Args:
      thresholds: Thresholds to use for fall-out. Defaults to [0.5].
      name: Metric name.
    """
    super(FallOut, self).__init__(name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    del tp, fn
    denominator = fp + tn
    if denominator > 0.0:
      return fp / denominator
    else:
      return float('nan')


metric_types.register_metric(FallOut)


class MissRate(ConfusionMatrixMetric):
  """Miss rate (FNR)."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = MISS_RATE_NAME):
    """Initializes miss rate metric.

    Args:
      thresholds: Thresholds to use for miss rate. Defaults to [0.5].
      name: Metric name.
    """
    super(MissRate, self).__init__(name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    del tn, fp
    denominator = fn + tp
    if denominator > 0.0:
      return fn / denominator
    else:
      return float('nan')


metric_types.register_metric(MissRate)


class NegativePredictiveValue(ConfusionMatrixMetric):
  """Negative predictive value (NPV)."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = NEGATIVE_PREDICTIVE_VALUE_NAME):
    """Initializes negative predictive value.

    Args:
      thresholds: Thresholds to use. Defaults to [0.5].
      name: Metric name.
    """
    super(NegativePredictiveValue, self).__init__(
        name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    del tp, fp
    denominator = tn + fn
    if denominator > 0.0:
      return tn / denominator
    else:
      return float('nan')


metric_types.register_metric(NegativePredictiveValue)


class FalseDiscoveryRate(ConfusionMatrixMetric):
  """False discovery rate (FDR)."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = FALSE_DISCOVERY_RATE_NAME):
    """Initializes false discovery rate.

    Args:
      thresholds: Thresholds to use. Defaults to [0.5].
      name: Metric name.
    """
    super(FalseDiscoveryRate, self).__init__(name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    del tn, fn
    denominator = fp + tp
    if denominator > 0.0:
      return fp / denominator
    else:
      return float('nan')


metric_types.register_metric(FalseDiscoveryRate)


class FalseOmissionRate(ConfusionMatrixMetric):
  """False omission rate (FOR)."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = FALSE_OMISSION_RATE_NAME):
    """Initializes false omission rate.

    Args:
      thresholds: Thresholds to use. Defaults to [0.5].
      name: Metric name.
    """
    super(FalseOmissionRate, self).__init__(name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    del tp, fp
    denominator = fn + tn
    if denominator > 0.0:
      return fn / denominator
    else:
      return float('nan')


metric_types.register_metric(FalseOmissionRate)


class Prevalence(ConfusionMatrixMetric):
  """Prevalence."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = PREVALENCE_NAME):
    """Initializes prevalence.

    Args:
      thresholds: Thresholds to use. Defaults to [0.5].
      name: Metric name.
    """
    super(Prevalence, self).__init__(name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    denominator = tp + tn + fp + fn
    if denominator > 0.0:
      return (tp + fn) / denominator
    else:
      return float('nan')


metric_types.register_metric(Prevalence)


class PrevalenceThreshold(ConfusionMatrixMetric):
  """Prevalence threshold (PT)."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = PREVALENCE_THRESHOLD_NAME):
    """Initializes prevalence threshold.

    Args:
      thresholds: Thresholds to use. Defaults to [0.5].
      name: Metric name.
    """
    super(PrevalenceThreshold, self).__init__(name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    tpr_denominator = tp + fn
    tnr_denominator = tn + fp
    if tpr_denominator > 0.0 and tnr_denominator > 0.0:
      tpr = tp / tpr_denominator
      tnr = tn / tnr_denominator
      return (_pos_sqrt(tpr * (1 - tnr)) + tnr - 1) / (tpr + tnr - 1)
    else:
      return float('nan')


metric_types.register_metric(PrevalenceThreshold)


class ThreatScore(ConfusionMatrixMetric):
  """Threat score or critical success index (TS or CSI)."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = THREAT_SCORE_NAME):
    """Initializes threat score.

    Args:
      thresholds: Thresholds to use. Defaults to [0.5].
      name: Metric name.
    """
    super(ThreatScore, self).__init__(name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    del tn
    denominator = tp + fn + fp
    if denominator > 0.0:
      return tp / denominator
    else:
      return float('nan')


metric_types.register_metric(ThreatScore)


class BalancedAccuracy(ConfusionMatrixMetric):
  """Balanced accuracy (BA)."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = BALANCED_ACCURACY_NAME):
    """Initializes balanced accuracy.

    Args:
      thresholds: Thresholds to use. Defaults to [0.5].
      name: Metric name.
    """
    super(BalancedAccuracy, self).__init__(name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    tpr_denominator = tp + fn
    tnr_denominator = tn + fp
    if tpr_denominator > 0.0 and tnr_denominator > 0.0:
      tpr = tp / tpr_denominator
      tnr = tn / tnr_denominator
      return (tpr + tnr) / 2
    else:
      return float('nan')


metric_types.register_metric(BalancedAccuracy)


class F1Score(ConfusionMatrixMetric):
  """F1 score."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = F1_SCORE_NAME):
    """Initializes F1 score.

    Args:
      thresholds: Thresholds to use. Defaults to [0.5].
      name: Metric name.
    """
    super(F1Score, self).__init__(name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    del tn
    denominator = 2 * tp + fp + fn
    if denominator > 0.0:
      # This is the harmonic mean of precision and recall or the same as
      # 2 * (precision * recall) / (precision + recall).
      # See https://en.wikipedia.org/wiki/Confusion_matrix for more information.
      return 2 * tp / denominator
    else:
      return float('nan')


metric_types.register_metric(F1Score)


class MatthewsCorrelationCoefficent(ConfusionMatrixMetric):
  """Matthews corrrelation coefficient (MCC)."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = MATTHEWS_CORRELATION_COEFFICENT_NAME):
    """Initializes matthews corrrelation coefficient.

    Args:
      thresholds: Thresholds to use. Defaults to [0.5].
      name: Metric name.
    """
    super(MatthewsCorrelationCoefficent, self).__init__(
        name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    denominator = _pos_sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator > 0.0:
      return (tp * tn - fp * fn) / denominator
    else:
      return float('nan')


metric_types.register_metric(MatthewsCorrelationCoefficent)


class FowlkesMallowsIndex(ConfusionMatrixMetric):
  """Fowlkes-Mallows index (FM)."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = FOWLKES_MALLOWS_INDEX_NAME):
    """Initializes fowlkes-mallows index.

    Args:
      thresholds: Thresholds to use. Defaults to [0.5].
      name: Metric name.
    """
    super(FowlkesMallowsIndex, self).__init__(name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    del tn
    ppv_denominator = tp + fp
    tpr_denominator = tp + fn
    if ppv_denominator > 0.0 and tpr_denominator > 0.0:
      ppv = tp / ppv_denominator
      tnr = tp / tpr_denominator
      return _pos_sqrt(ppv * tnr)
    else:
      return float('nan')


metric_types.register_metric(FowlkesMallowsIndex)


class Informedness(ConfusionMatrixMetric):
  """Informedness or bookmaker informedness (BM)."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = INFORMEDNESS_NAME):
    """Initializes informedness.

    Args:
      thresholds: Thresholds to use. Defaults to [0.5].
      name: Metric name.
    """
    super(Informedness, self).__init__(name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    positives = tp + fn
    negatives = tn + fp
    if positives > 0.0 and negatives > 0.0:
      tpr = tp / positives
      tnr = tn / negatives
      return tpr + tnr - 1
    else:
      return float('nan')


metric_types.register_metric(Informedness)


class Markedness(ConfusionMatrixMetric):
  """Markedness (MK) or deltaP."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = MARKEDNESS_NAME):
    """Initializes markedness.

    Args:
      thresholds: Thresholds to use. Defaults to [0.5].
      name: Metric name.
    """
    super(Markedness, self).__init__(name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    ppv_denominator = tp + fp
    npv_denominator = tn + fn
    if ppv_denominator > 0.0 and npv_denominator > 0.0:
      ppv = tp / ppv_denominator
      npv = tn / npv_denominator
      return ppv + npv - 1
    else:
      return float('nan')


metric_types.register_metric(Markedness)


class PositiveLikelihoodRatio(ConfusionMatrixMetric):
  """Positive likelihood ratio (LR+)."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = POSITIVE_LIKELIHOOD_RATIO_NAME):
    """Initializes positive likelihood ratio.

    Args:
      thresholds: Thresholds to use. Defaults to [0.5].
      name: Metric name.
    """
    super(PositiveLikelihoodRatio, self).__init__(
        name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    tpr_denominator = tp + fn
    fpr_denominator = fp + tn
    if tpr_denominator > 0.0 and fpr_denominator > 0.0 and fp > 0.0:
      tpr = tp / tpr_denominator
      fpr = fp / fpr_denominator
      return tpr / fpr
    else:
      return float('nan')


metric_types.register_metric(PositiveLikelihoodRatio)


class NegativeLikelihoodRatio(ConfusionMatrixMetric):
  """Negative likelihood ratio (LR-)."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = NEGATIVE_LIKELIHOOD_RATIO_NAME):
    """Initializes negative likelihood ratio.

    Args:
      thresholds: Thresholds to use. Defaults to [0.5].
      name: Metric name.
    """
    super(NegativeLikelihoodRatio, self).__init__(
        name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    fnr_denominator = fn + tp
    tnr_denominator = tn + fp
    if fnr_denominator > 0.0 and tnr_denominator > 0.0 and tn > 0.0:
      fnr = fn / fnr_denominator
      tnr = tn / tnr_denominator
      return fnr / tnr
    else:
      return float('nan')


metric_types.register_metric(NegativeLikelihoodRatio)


class DiagnosticOddsRatio(ConfusionMatrixMetric):
  """Diagnostic odds ratio (DOR)."""

  def __init__(self,
               thresholds: Optional[List[float]] = None,
               name: Text = DIAGNOSTIC_ODDS_RATIO_NAME):
    """Initializes diagnostic odds ratio.

    Args:
      thresholds: Thresholds to use. Defaults to [0.5].
      name: Metric name.
    """
    super(DiagnosticOddsRatio, self).__init__(name=name, thresholds=thresholds)

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    if fn > 0.0 and fp > 0.0 and tn > 0.0:
      return (tp / fn) / (fp / tn)
    else:
      return float('nan')


metric_types.register_metric(DiagnosticOddsRatio)


class ConfusionMatrixAtThresholds(metric_types.Metric):
  """Confusion matrix at thresholds."""

  def __init__(self,
               thresholds: List[float],
               name: Text = CONFUSION_MATRIX_AT_THRESHOLDS_NAME):
    """Initializes confusion matrix at thresholds.

    Args:
      thresholds: Thresholds to use for confusion matrix.
      name: Metric name.
    """
    super(ConfusionMatrixAtThresholds, self).__init__(
        metric_util.merge_per_key_computations(_confusion_matrix_at_thresholds),
        thresholds=thresholds,
        name=name)


metric_types.register_metric(ConfusionMatrixAtThresholds)


def _confusion_matrix_at_thresholds(
    thresholds: List[float],
    name: Text = CONFUSION_MATRIX_AT_THRESHOLDS_NAME,
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    output_name: Text = '',
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None
) -> metric_types.MetricComputations:
  """Returns metric computations for confusion matrix at thresholds."""
  key = metric_types.MetricKey(
      name=name,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key)

  # Make sure matrices are calculated.
  matrices_computations = binary_confusion_matrices.binary_confusion_matrices(
      eval_config=eval_config,
      model_name=model_name,
      output_name=output_name,
      sub_key=sub_key,
      aggregation_type=aggregation_type,
      class_weights=class_weights,
      thresholds=thresholds)
  matrices_key = matrices_computations[-1].keys[-1]

  def result(
      metrics: Dict[metric_types.MetricKey, binary_confusion_matrices.Matrices]
  ) -> Dict[metric_types.MetricKey, Any]:
    return {key: to_proto(thresholds, metrics[matrices_key])}

  derived_computation = metric_types.DerivedMetricComputation(
      keys=[key], result=result)
  computations = matrices_computations
  computations.append(derived_computation)
  return computations


def to_proto(
    thresholds: List[float], matrices: binary_confusion_matrices.Matrices
) -> metrics_for_slice_pb2.ConfusionMatrixAtThresholds:
  """Converts matrices into ConfusionMatrixAtThresholds proto.

  If precision or recall are undefined then 1.0 and 0.0 will be used.

  Args:
    thresholds: Thresholds.
    matrices: Confusion matrices.

  Returns:
    Matrices in ConfusionMatrixAtThresholds proto format.
  """
  pb = metrics_for_slice_pb2.ConfusionMatrixAtThresholds()
  for i, threshold in enumerate(thresholds):
    precision = 1.0
    if matrices.tp[i] + matrices.fp[i] > 0:
      precision = matrices.tp[i] / (matrices.tp[i] + matrices.fp[i])
    recall = 0.0
    if matrices.tp[i] + matrices.fn[i] > 0:
      recall = matrices.tp[i] / (matrices.tp[i] + matrices.fn[i])
    pb.matrices.add(
        threshold=round(threshold, 6),
        true_positives=matrices.tp[i],
        false_positives=matrices.fp[i],
        true_negatives=matrices.tn[i],
        false_negatives=matrices.fn[i],
        precision=precision,
        recall=recall)
  return pb
