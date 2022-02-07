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

import abc
import copy
import enum
import math

from typing import Any, Dict, List, Optional, Union

import numpy as np
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2

AUC_NAME = 'auc'
AUC_PRECISION_RECALL_NAME = 'auc_precision_recall'
SENSITIVITY_AT_SPECIFICITY_NAME = 'sensitivity_at_specificity'
SPECIFICITY_AT_SENSITIVITY_NAME = 'specificity_at_sensitivity'
PRECISION_AT_RECALL_NAME = 'precision_at_recall'
RECALL_AT_PRECISION_NAME = 'recall_at_precision'
TRUE_POSITIVES_NAME = 'true_positives'
TP_NAME = 'tp'
TRUE_NEGATIVES_NAME = 'true_negatives'
TN_NAME = 'tn'
FALSE_POSITIVES_NAME = 'false_positives'
FP_NAME = 'fp'
FALSE_NEGATIVES_NAME = 'false_negatives'
FN_NAME = 'fn'
BINARY_ACCURACY_NAME = 'binary_accuracy'
PRECISION_NAME = 'precision'
PPV_NAME = 'ppv'
RECALL_NAME = 'recall'
TPR_NAME = 'tpr'
SPECIFICITY_NAME = 'specificity'
TNR_NAME = 'tnr'
FALL_OUT_NAME = 'fall_out'
FPR_NAME = 'fpr'
MISS_RATE_NAME = 'miss_rate'
FNR_NAME = 'fnr'
NEGATIVE_PREDICTIVE_VALUE_NAME = 'negative_predictive_value'
NPV_NAME = 'npv'
FALSE_DISCOVERY_RATE_NAME = 'false_discovery_rate'
FALSE_OMISSION_RATE_NAME = 'false_omission_rate'
PREVALENCE_NAME = 'prevalence'
PREVALENCE_THRESHOLD_NAME = 'prevalence_threshold'
THREAT_SCORE_NAME = 'threat_score'
BALANCED_ACCURACY_NAME = 'balanced_accuracy'
F1_SCORE_NAME = 'f1_score'
MATTHEWS_CORRELATION_COEFFICIENT_NAME = 'matthews_correlation_coefficient'
FOWLKES_MALLOWS_INDEX_NAME = 'fowlkes_mallows_index'
INFORMEDNESS_NAME = 'informedness'
MARKEDNESS_NAME = 'markedness'
POSITIVE_LIKELIHOOD_RATIO_NAME = 'positive_likelihood_ratio'
NEGATIVE_LIKELIHOOD_RATIO_NAME = 'negative_likelihood_ratio'
DIAGNOSTIC_ODDS_RATIO_NAME = 'diagnostic_odds_ratio'
PREDICTED_POSITIVE_RATE_NAME = 'predicted_positive_rate'
CONFUSION_MATRIX_AT_THRESHOLDS_NAME = 'confusion_matrix_at_thresholds'


class AUCCurve(enum.Enum):
  ROC = 'ROC'
  PR = 'PR'


class AUCSummationMethod(enum.Enum):
  INTERPOLATION = 'interpolation'
  MAJORING = 'majoring'
  MINORING = 'minoring'


def _pos_sqrt(value: float) -> float:
  """Returns sqrt of value or raises ValueError if negative."""
  if value < 0:
    raise ValueError('Attempt to take sqrt of negative value: {}'.format(value))
  return math.sqrt(value)


def _validate_and_update_sub_key(
    metric_name: str, model_name: str, output_name: str,
    sub_key: metric_types.SubKey, top_k: Optional[int],
    class_id: Optional[int]) -> metric_types.SubKey:
  """Validates and updates sub key.

  This function validates that the top_k and class_id settings that are
  determined by the MetricsSpec.binarize are compatible and do not overlap with
  any settings provided by MetricConfigs.

  Args:
    metric_name: Metric name.
    model_name: Model name.
    output_name: Output name.
    sub_key: Sub key (from MetricsSpec).
    top_k: Top k setting (from MetricConfig).
    class_id: Class ID setting (from MetricConfig).

  Returns:
    Updated sub-key if top_k or class_id params are used.

  Raises:
    ValueError: If validation fails.
  """
  if top_k and class_id:
    raise ValueError(
        f'Metric {metric_name} is configured with both class_id={class_id} and '
        f'top_k={top_k} settings. Only one may be specified at a time.')
  if top_k is not None:
    if sub_key is None or sub_key == metric_types.SubKey():
      sub_key = metric_types.SubKey(top_k=top_k)
    else:
      raise ValueError(
          f'Metric {metric_name} is configured with overlapping settings. '
          f'The metric was initialized with top_k={top_k}, but the '
          f'metric was defined in a spec using sub_key={sub_key}, '
          f'model_name={model_name}, output_name={output_name}\n\n'
          'Binarization related settings can be configured in either the'
          'metrics_spec or the metric, but not both. Either remove the top_k '
          'setting from this metric or remove the metrics_spec.binarize '
          'settings.')
  elif class_id is not None:
    if sub_key is None or sub_key == metric_types.SubKey():
      sub_key = metric_types.SubKey(class_id=class_id)
    else:
      raise ValueError(
          f'Metric {metric_name} is configured with overlapping settings. '
          f'The metric was initialized with class_id={class_id}, but the '
          f'metric was defined in a spec using sub_key={sub_key}, '
          f'model_name={model_name}, output_name={output_name}\n\n'
          'Binarization related settings can be configured in either the'
          'metrics_spec or the metric, but not both. Either remove the '
          'class_id setting from this metric or remove the '
          'metrics_spec.binarize settings.')
  return sub_key


def _find_max_under_constraint(constrained, dependent, value):
  """Returns the maximum of dependent that satisfies contrained >= value.

  Args:
    constrained: Over these values the constraint is specified. A rank-1 np
      array.
    dependent: From these values the maximum that satiesfies the constraint is
      selected. Values in this array and in `constrained` are linked by having
      the same threshold at each position, hence this array must have the same
      shape.
    value: The lower bound where contrained >= value.

  Returns:
    Maximal dependent value, if no value satiesfies the constraint 0.0.
  """
  feasible = np.where(constrained >= value)
  gathered = np.take(dependent, feasible)
  if gathered.size > 0:
    return float(np.where(np.size(feasible) > 0, np.nanmax(gathered), 0.0))
  # If the gathered is empty, return 0.0 assuming all NaNs are 0.0
  return 0.0


class ConfusionMatrixMetricBase(metric_types.Metric, metaclass=abc.ABCMeta):
  """Base for confusion matrix metrics."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               num_thresholds: Optional[int] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None,
               name: Optional[str] = None,
               **kwargs):
    """Initializes confusion matrix metric.

    Args:
      thresholds: (Optional) Thresholds to use for calculating the matrices. Use
        one of either thresholds or num_thresholds.
      num_thresholds: (Optional) Number of thresholds to use for calculating the
        matrices. Use one of either thresholds or num_thresholds.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
      name: (Optional) Metric name.
      **kwargs: (Optional) Additional args to pass along to init (and eventually
        on to _metric_computation and _metric_value)
    """
    super().__init__(
        metric_util.merge_per_key_computations(self._metric_computations),
        thresholds=thresholds,
        num_thresholds=num_thresholds,
        top_k=top_k,
        class_id=class_id,
        name=name,
        **kwargs)

  def _default_threshold(self) -> Optional[float]:
    """Returns default threshold if thresholds or num_thresholds unset."""
    return None

  def get_config(self) -> Dict[str, Any]:
    """Returns serializable config."""
    # Not all subclasses of ConfusionMatrixMetric support all the __init__
    # parameters as part of their __init__, to avoid deserialization issues
    # where an unsupported parameter is passed to the subclass, filter out any
    # parameters that are None.
    kwargs = copy.copy(self.kwargs)
    for arg in ('thresholds', 'num_thresholds', 'top_k', 'class_id'):
      if kwargs[arg] is None:
        del kwargs[arg]
    return kwargs

  @abc.abstractmethod
  def _metric_value(
      self, key: metric_types.MetricKey,
      matrices: binary_confusion_matrices.Matrices) -> Union[float, np.ndarray]:
    """Returns metric value associated with matrices.

    Subclasses may override this method. Any additional kwargs passed to
    __init__ will be forwarded along to this call.

    Args:
      key: Metric key.
      matrices: Computed binary confusion matrices.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  def _metric_computations(self,
                           thresholds: Optional[Union[float,
                                                      List[float]]] = None,
                           num_thresholds: Optional[int] = None,
                           top_k: Optional[int] = None,
                           class_id: Optional[int] = None,
                           name: Optional[str] = None,
                           eval_config: Optional[config_pb2.EvalConfig] = None,
                           model_name: str = '',
                           output_name: str = '',
                           sub_key: Optional[metric_types.SubKey] = None,
                           aggregation_type: Optional[
                               metric_types.AggregationType] = None,
                           class_weights: Optional[Dict[int, float]] = None,
                           example_weighted: bool = False,
                           **kwargs) -> metric_types.MetricComputations:
    """Returns computations for confusion matrix metric."""
    sub_key = _validate_and_update_sub_key(name, model_name, output_name,
                                           sub_key, top_k, class_id)

    key = metric_types.MetricKey(
        name=name,
        model_name=model_name,
        output_name=output_name,
        sub_key=sub_key,
        example_weighted=example_weighted,
        aggregation_type=aggregation_type)

    if num_thresholds is None and thresholds is None:
      # If top_k set, then use -inf as the default threshold setting.
      if sub_key and sub_key.top_k:
        thresholds = [float('-inf')]
      elif self._default_threshold() is not None:
        thresholds = [self._default_threshold()]
    if isinstance(thresholds, float):
      thresholds = [thresholds]

    # Make sure matrices are calculated.
    matrices_computations = binary_confusion_matrices.binary_confusion_matrices(
        num_thresholds=num_thresholds,
        thresholds=thresholds,
        eval_config=eval_config,
        model_name=model_name,
        output_name=output_name,
        sub_key=sub_key,
        aggregation_type=aggregation_type,
        class_weights=class_weights,
        example_weighted=example_weighted)
    matrices_key = matrices_computations[-1].keys[-1]

    def result(
        metrics: Dict[metric_types.MetricKey, Any]
    ) -> Dict[metric_types.MetricKey, Union[float, np.ndarray]]:
      value = self._metric_value(
          key=key, matrices=metrics[matrices_key], **kwargs)
      return {key: value}

    derived_computation = metric_types.DerivedMetricComputation(
        keys=[key], result=result)
    computations = matrices_computations
    computations.append(derived_computation)
    return computations


class ConfusionMatrixMetric(ConfusionMatrixMetricBase):
  """Base for confusion matrix metrics."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               num_thresholds: Optional[int] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None,
               name: Optional[str] = None,
               **kwargs):
    """Initializes confusion matrix metric.

    Args:
      thresholds: (Optional) Thresholds to use for calculating the matrices. Use
        one of either thresholds or num_thresholds.
      num_thresholds: (Optional) Number of thresholds to use for calculating the
        matrices. Use one of either thresholds or num_thresholds.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
      name: (Optional) Metric name.
      **kwargs: (Optional) Additional args to pass along to init (and eventually
        on to _metric_computation and _metric_value)
    """
    super().__init__(
        thresholds=thresholds,
        num_thresholds=num_thresholds,
        top_k=top_k,
        class_id=class_id,
        name=name,
        **kwargs)

  def _default_threshold(self) -> float:
    """Returns default threshold if thresholds or num_thresholds unset."""
    return 0.5

  @abc.abstractmethod
  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    """Function for computing metric value from TP, TN, FP, FN values."""
    raise NotImplementedError('Must be implemented in subclasses.')

  def _metric_value(
      self, key: metric_types.MetricKey,
      matrices: binary_confusion_matrices.Matrices) -> Union[float, np.ndarray]:
    """Returns metric value associated with matrices.

    Subclasses may override this method. Any additional kwargs passed to
    __init__ will be forwarded along to this call. Note that since this method
    is the only one that calls the result method, subclasses that override this
    method are not required to provide an implementation for the result method.

    Args:
      key: Metric key.
      matrices: Computed binary confusion matrices.
    """
    values = []
    for i in range(len(matrices.thresholds)):
      values.append(
          self.result(matrices.tp[i], matrices.tn[i], matrices.fp[i],
                      matrices.fn[i]))
    return values[0] if len(matrices.thresholds) == 1 else np.array(values)


class AUC(ConfusionMatrixMetricBase):
  """Approximates the AUC (Area under the curve) of the ROC or PR curves.

  The AUC (Area under the curve) of the ROC (Receiver operating
  characteristic; default) or PR (Precision Recall) curves are quality measures
  of binary classifiers. Unlike the accuracy, and like cross-entropy
  losses, ROC-AUC and PR-AUC evaluate all the operational points of a model.

  This class approximates AUCs using a Riemann sum. During the metric
  accumulation phase, predictions are accumulated within predefined buckets
  by value. The AUC is then computed by interpolating per-bucket averages. These
  buckets define the evaluated operational points.

  This metric uses `true_positives`, `true_negatives`, `false_positives` and
  `false_negatives` to compute the AUC. To discretize the AUC curve, a linearly
  spaced set of thresholds is used to compute pairs of recall and precision
  values. The area under the ROC-curve is therefore computed using the height of
  the recall values by the false positive rate, while the area under the
  PR-curve is the computed using the height of the precision values by the
  recall.

  This value is ultimately returned as `auc`, an idempotent operation that
  computes the area under a discretized curve of precision versus recall values
  (computed using the aforementioned variables). The `num_thresholds` variable
  controls the degree of discretization with larger numbers of thresholds more
  closely approximating the true AUC. The quality of the approximation may vary
  dramatically depending on `num_thresholds`. The `thresholds` parameter can be
  used to manually specify thresholds which split the predictions more evenly.

  For a best approximation of the real AUC, `predictions` should be distributed
  approximately uniformly in the range [0, 1]. The quality of the AUC
  approximation may be poor if this is not the case. Setting `summation_method`
  to 'minoring' or 'majoring' can help quantify the error in the approximation
  by providing lower or upper bound estimate of the AUC.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  """

  def __init__(self,
               num_thresholds: Optional[int] = None,
               curve: str = 'ROC',
               summation_method: str = 'interpolation',
               name: Optional[str] = None,
               thresholds: Optional[Union[float, List[float]]] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes AUC metric.

    Args:
      num_thresholds: (Optional) Defaults to 10000. The number of thresholds to
        use when discretizing the roc curve. Values must be > 1.
      curve: (Optional) Specifies the name of the curve to be computed, 'ROC'
        [default] or 'PR' for the Precision-Recall-curve.
      summation_method: (Optional) Specifies the [Riemann summation method](
        https://en.wikipedia.org/wiki/Riemann_sum) used. 'interpolation'
          (default) applies mid-point summation scheme for `ROC`. For PR-AUC,
          interpolates (true/false) positives but not the ratio that is
          precision (see Davis & Goadrich 2006 for details); 'minoring' applies
          left summation for increasing intervals and right summation for
          decreasing intervals; 'majoring' does the opposite.
      name: (Optional) string name of the metric instance.
      thresholds: (Optional) A list of floating point values to use as the
        thresholds for discretizing the curve. If set, the `num_thresholds`
        parameter is ignored. Values should be in [0, 1]. Endpoint thresholds
        equal to {-epsilon, 1+epsilon} for a small positive epsilon value will
        be automatically included with these to correctly handle predictions
        equal to exactly 0 or 1.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        num_thresholds=num_thresholds,
        thresholds=thresholds,
        curve=curve,
        summation_method=summation_method,
        name=name,
        top_k=top_k,
        class_id=class_id)

  def _default_name(self) -> str:
    return AUC_NAME

  def _metric_value(self, curve: str, summation_method: str,
                    key: metric_types.MetricKey,
                    matrices: binary_confusion_matrices.Matrices) -> float:
    del key
    curve = AUCCurve(curve.upper())
    summation_method = AUCSummationMethod(summation_method.lower())
    num_thresholds = len(matrices.thresholds)
    tp, tn = np.array(matrices.tp), np.array(matrices.tn)
    fp, fn = np.array(matrices.fp), np.array(matrices.fn)
    if (curve == AUCCurve.PR and
        summation_method == AUCSummationMethod.INTERPOLATION):
      dtp = tp[:num_thresholds - 1] - tp[1:]
      p = tp + fp
      dp = p[:num_thresholds - 1] - p[1:]
      prec_slope = dtp / np.maximum(dp, 0)
      intercept = tp[1:] - prec_slope * p[1:]
      safe_p_ratio = np.where(
          np.logical_and(p[:num_thresholds - 1] > 0, p[1:] > 0),
          p[:num_thresholds - 1] / np.maximum(p[1:], 0), np.ones_like(p[1:]))
      pr_auc_increment = (
          prec_slope * (dtp + intercept * np.log(safe_p_ratio)) /
          np.maximum(tp[1:] + fn[1:], 0))
      return np.nansum(pr_auc_increment)

    # Set `x` and `y` values for the curves based on `curve` config.
    recall = tp / (tp + fn)
    if curve == AUCCurve.ROC:
      fp_rate = fp / (fp + tn)
      x = fp_rate
      y = recall
    elif curve == AUCCurve.PR:
      precision = tp / (tp + fp)
      x = recall
      y = precision

    # Find the rectangle heights based on `summation_method`.
    if summation_method == AUCSummationMethod.INTERPOLATION:
      heights = (y[:num_thresholds - 1] + y[1:]) / 2.
    elif summation_method == AUCSummationMethod.MINORING:
      heights = np.minimum(y[:num_thresholds - 1], y[1:])
    elif summation_method == AUCSummationMethod.MAJORING:
      heights = np.maximum(y[:num_thresholds - 1], y[1:])

    # Sum up the areas of all the rectangles.
    return np.nansum((x[:num_thresholds - 1] - x[1:]) * heights)


metric_types.register_metric(AUC)


class AUCPrecisionRecall(AUC):
  """Alias for AUC(curve='PR')."""

  def __init__(self,
               num_thresholds: Optional[int] = None,
               summation_method: str = 'interpolation',
               name: Optional[str] = None,
               thresholds: Optional[Union[float, List[float]]] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes AUCPrecisionRecall metric.

    Args:
      num_thresholds: (Optional) Defaults to 10000. The number of thresholds to
        use when discretizing the roc curve. Values must be > 1.
      summation_method: (Optional) Specifies the [Riemann summation method](
        https://en.wikipedia.org/wiki/Riemann_sum) used. 'interpolation'
          interpolates (true/false) positives but not the ratio that is
          precision (see Davis & Goadrich 2006 for details); 'minoring' applies
          left summation for increasing intervals and right summation for
          decreasing intervals; 'majoring' does the opposite.
      name: (Optional) string name of the metric instance.
      thresholds: (Optional) A list of floating point values to use as the
        thresholds for discretizing the curve. If set, the `num_thresholds`
        parameter is ignored. Values should be in [0, 1]. Endpoint thresholds
        equal to {-epsilon, 1+epsilon} for a small positive epsilon value will
        be automatically included with these to correctly handle predictions
        equal to exactly 0 or 1.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        num_thresholds=num_thresholds,
        thresholds=thresholds,
        curve='PR',
        summation_method=summation_method,
        name=name,
        top_k=top_k,
        class_id=class_id)

  def _default_name(self) -> str:
    return AUC_PRECISION_RECALL_NAME


metric_types.register_metric(AUCPrecisionRecall)


class SensitivityAtSpecificity(ConfusionMatrixMetricBase):
  """Computes best sensitivity where specificity is >= specified value.

  `Sensitivity` measures the proportion of actual positives that are correctly
  identified as such (tp / (tp + fn)).
  `Specificity` measures the proportion of actual negatives that are correctly
  identified as such (tn / (tn + fp)).

  The threshold for the given specificity value is computed and used to evaluate
  the corresponding sensitivity.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  For additional information about specificity and sensitivity, see
  [the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).
  """

  def __init__(self,
               specificity: float,
               num_thresholds: Optional[int] = None,
               class_id: Optional[int] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None):
    """Initializes SensitivityAtSpecificity metric.


    Args:
      specificity: A scalar value in range `[0, 1]`.
      num_thresholds: (Optional) Defaults to 1000. The number of thresholds to
        use for matching the given specificity.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
      name: (Optional) string name of the metric instance.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
    """
    super().__init__(
        num_thresholds=num_thresholds,
        specificity=specificity,
        class_id=class_id,
        name=name,
        top_k=top_k)

  def _default_name(self) -> str:
    return SENSITIVITY_AT_SPECIFICITY_NAME

  def _metric_value(self, specificity: float, key: metric_types.MetricKey,
                    matrices: binary_confusion_matrices.Matrices) -> float:
    del key
    tp, tn = np.array(matrices.tp), np.array(matrices.tn)
    fp, fn = np.array(matrices.fp), np.array(matrices.fn)
    specificities = tn / (tn + fp)
    sensitivities = tp / (tp + fn)
    return _find_max_under_constraint(specificities, sensitivities, specificity)


metric_types.register_metric(SensitivityAtSpecificity)


class SpecificityAtSensitivity(ConfusionMatrixMetricBase):
  """Computes best specificity where sensitivity is >= specified value.

  `Sensitivity` measures the proportion of actual positives that are correctly
  identified as such (tp / (tp + fn)).
  `Specificity` measures the proportion of actual negatives that are correctly
  identified as such (tn / (tn + fp)).

  The threshold for the given sensitivity value is computed and used to evaluate
  the corresponding specificity.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  For additional information about specificity and sensitivity, see
  [the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).
  """

  def __init__(self,
               sensitivity: float,
               num_thresholds: Optional[int] = None,
               class_id: Optional[int] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None):
    """Initializes SpecificityAtSensitivity metric.


    Args:
      sensitivity: A scalar value in range `[0, 1]`.
      num_thresholds: (Optional) Defaults to 1000. The number of thresholds to
        use for matching the given sensitivity.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
      name: (Optional) string name of the metric instance.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
    """
    super().__init__(
        num_thresholds=num_thresholds,
        sensitivity=sensitivity,
        class_id=class_id,
        name=name,
        top_k=top_k)

  def _default_name(self) -> str:
    return SPECIFICITY_AT_SENSITIVITY_NAME

  def _metric_value(self, sensitivity: float, key: metric_types.MetricKey,
                    matrices: binary_confusion_matrices.Matrices) -> float:
    del key
    tp, tn = np.array(matrices.tp), np.array(matrices.tn)
    fp, fn = np.array(matrices.fp), np.array(matrices.fn)
    specificities = tn / (tn + fp)
    sensitivities = tp / (tp + fn)
    return _find_max_under_constraint(sensitivities, specificities, sensitivity)


metric_types.register_metric(SpecificityAtSensitivity)


class PrecisionAtRecall(ConfusionMatrixMetricBase):
  """Computes best precision where recall is >= specified value.

  The threshold for the given recall value is computed and used to evaluate the
  corresponding precision.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  """

  def __init__(self,
               recall: float,
               num_thresholds: Optional[int] = None,
               class_id: Optional[int] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None):
    """Initializes PrecisionAtRecall metric.


    Args:
      recall: A scalar value in range `[0, 1]`.
      num_thresholds: (Optional) Defaults to 1000. The number of thresholds to
        use for matching the given recall.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
      name: (Optional) string name of the metric instance.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
    """
    if recall < 0 or recall > 1:
      raise ValueError('Argument `recall` must be in the range [0, 1]. '
                       f'Received: recall={recall}')
    super().__init__(
        num_thresholds=num_thresholds,
        recall=recall,
        class_id=class_id,
        name=name,
        top_k=top_k)

  def _default_name(self) -> str:
    return PRECISION_AT_RECALL_NAME

  def _metric_value(self, recall: float, key: metric_types.MetricKey,
                    matrices: binary_confusion_matrices.Matrices) -> float:
    del key
    tp = np.array(matrices.tp)
    fp, fn = np.array(matrices.fp), np.array(matrices.fn)
    recalls = tp / (tp + fn)
    precisions = tp / (tp + fp)
    return _find_max_under_constraint(recalls, precisions, recall)


metric_types.register_metric(PrecisionAtRecall)


class RecallAtPrecision(ConfusionMatrixMetricBase):
  """Computes best recall where precision is >= specified value.

  For a given score-label-distribution the required precision might not
  be achievable, in this case 0.0 is returned as recall.

  This metric creates three local variables, `true_positives`, `false_positives`
  and `false_negatives` that are used to compute the recall at the given
  precision. The threshold for the given precision value is computed and used to
  evaluate the corresponding recall.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  """

  def __init__(self,
               precision: float,
               num_thresholds: Optional[int] = None,
               class_id: Optional[int] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None):
    """Initializes RecallAtPrecision.


    Args:
      precision: A scalar value in range `[0, 1]`.
      num_thresholds: (Optional) Defaults to 1000. The number of thresholds to
        use for matching the given precision.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
      name: (Optional) string name of the metric instance.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
    """
    if precision < 0 or precision > 1:
      raise ValueError('Argument `precision` must be in the range [0, 1]. '
                       f'Received: precision={precision}')
    super().__init__(
        num_thresholds=num_thresholds,
        precision=precision,
        class_id=class_id,
        name=name,
        top_k=top_k)

  def _default_name(self) -> str:
    return RECALL_AT_PRECISION_NAME

  def _metric_value(self, precision: float, key: metric_types.MetricKey,
                    matrices: binary_confusion_matrices.Matrices) -> float:
    del key
    tp = np.array(matrices.tp)
    fp, fn = np.array(matrices.fp), np.array(matrices.fn)
    recalls = tp / (tp + fn)
    precisions = tp / (tp + fp)
    return _find_max_under_constraint(precisions, recalls, precision)


metric_types.register_metric(RecallAtPrecision)


class TruePositives(ConfusionMatrixMetric):
  """Calculates the number of true positives.

  If `sample_weight` is given, calculates the sum of the weights of
  true positives. This metric creates one local variable, `true_positives`
  that is used to keep track of the number of true positives.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  """

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes TruePositives metric.

    Args:
      thresholds: (Optional) Defaults to [0.5]. A float value or a python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). One metric
        value is generated for each threshold value.
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return TRUE_POSITIVES_NAME

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    return tp


metric_types.register_metric(TruePositives)


class TP(TruePositives):
  """Alias for TruePositives."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes TP metric."""
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return TP_NAME


metric_types.register_metric(TP)


class TrueNegatives(ConfusionMatrixMetric):
  """Calculates the number of true negatives.

  If `sample_weight` is given, calculates the sum of the weights of true
  negatives.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  """

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes TrueNegatives metric.

    Args:
      thresholds: (Optional) Defaults to [0.5]. A float value or a python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). One metric
        value is generated for each threshold value.
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return TRUE_NEGATIVES_NAME

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    return tn


metric_types.register_metric(TrueNegatives)


class TN(TrueNegatives):
  """Alias for TrueNegatives."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes TN metric."""
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return TN_NAME


metric_types.register_metric(TN)


class FalsePositives(ConfusionMatrixMetric):
  """Calculates the number of false positives.

  If `sample_weight` is given, calculates the sum of the weights of false
  positives.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  """

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes FalsePositives metric.

    Args:
      thresholds: (Optional) Defaults to [0.5]. A float value or a python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). One metric
        value is generated for each threshold value.
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return FALSE_POSITIVES_NAME

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    return fp


metric_types.register_metric(FalsePositives)


class FP(FalsePositives):
  """Alias for FalsePositives."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes FP metric."""
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return FP_NAME


metric_types.register_metric(FP)


class FalseNegatives(ConfusionMatrixMetric):
  """Calculates the number of false negatives.

  If `sample_weight` is given, calculates the sum of the weights of false
  negatives.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  """

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes FalseNegatives metric.

    Args:
      thresholds: (Optional) Defaults to [0.5]. A float value or a python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). One metric
        value is generated for each threshold value.
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return FALSE_NEGATIVES_NAME

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    return fn


metric_types.register_metric(FalseNegatives)


class FN(FalseNegatives):
  """Alias for FalseNegatives."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes FN metric."""
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return FN_NAME


metric_types.register_metric(FN)


class BinaryAccuracy(ConfusionMatrixMetric):
  """Calculates how often predictions match binary labels.

  This metric computes the accuracy based on (TP + TN) / (TP + FP + TN + FN).

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  """

  def __init__(self,
               threshold: Optional[float] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None,
               name: Optional[str] = None):
    """Initializes BinaryAccuracy metric.

    Args:
      threshold: (Optional) A float value in [0, 1]. The threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). If neither
        threshold nor top_k are set, the default is to calculate with
        `threshold=0.5`.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
      name: (Optional) string name of the metric instance.
    """
    super().__init__(
        thresholds=threshold, top_k=top_k, class_id=class_id, name=name)

  def _default_name(self) -> str:
    return BINARY_ACCURACY_NAME

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    denominator = tp + fp + tn + fn
    if denominator:
      return (tp + tn) / denominator
    else:
      return float('nan')


metric_types.register_metric(BinaryAccuracy)


class Precision(ConfusionMatrixMetric):
  """Computes the precision of the predictions with respect to the labels.

  The metric uses true positives and false positives to compute precision by
  dividing the true positives by the sum of true positives and false positives.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  """

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None,
               name: Optional[str] = None):
    """Initializes Precision metric.

    Args:
      thresholds: (Optional) A float value or a python list/tuple of float
        threshold values in [0, 1]. A threshold is compared with prediction
        values to determine the truth value of predictions (i.e., above the
        threshold is `true`, below is `false`). One metric value is generated
        for each threshold value. If neither thresholds nor top_k are set, the
        default is to calculate precision with `thresholds=0.5`.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
      name: (Optional) string name of the metric instance.
    """
    super().__init__(
        thresholds=thresholds, top_k=top_k, class_id=class_id, name=name)

  def _default_name(self) -> str:
    return PRECISION_NAME

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    del tn, fn
    denominator = tp + fp
    if denominator:
      return tp / denominator
    else:
      return float('nan')


metric_types.register_metric(Precision)


class PPV(Precision):
  """Alias for Precision."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes PPV metric."""
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return PPV_NAME


metric_types.register_metric(PPV)


class Recall(ConfusionMatrixMetric):
  """Computes the recall of the predictions with respect to the labels.

  The metric uses true positives and false negatives to compute recall by
  dividing the true positives by the sum of true positives and false negatives.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  """

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None,
               name: Optional[str] = None):
    """Initializes Recall metric.

    Args:
      thresholds: (Optional) A float value or a python list/tuple of float
        threshold values in [0, 1]. A threshold is compared with prediction
        values to determine the truth value of predictions (i.e., above the
        threshold is `true`, below is `false`). One metric value is generated
        for each threshold value. If neither thresholds nor top_k are set, the
        default is to calculate precision with `thresholds=0.5`.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
      name: (Optional) string name of the metric instance.
    """
    super().__init__(
        thresholds=thresholds, top_k=top_k, class_id=class_id, name=name)

  def _default_name(self) -> str:
    return RECALL_NAME

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    del tn, fp
    denominator = tp + fn
    if denominator > 0.0:
      return tp / denominator
    else:
      return float('nan')


metric_types.register_metric(Recall)


class TPR(Recall):
  """Alias for Recall."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes TPR metric."""
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return TPR_NAME


metric_types.register_metric(TPR)


class Specificity(ConfusionMatrixMetric):
  """Specificity (TNR) or selectivity."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes specificity metric.

    Args:
      thresholds: (Optional) Thresholds to use for specificity. Defaults to
        [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return SPECIFICITY_NAME

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    del tp, fn
    denominator = tn + fp
    if denominator > 0.0:
      return tn / denominator
    else:
      return float('nan')


metric_types.register_metric(Specificity)


class TNR(Specificity):
  """Alias for Specificity."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes TNR metric."""
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return TNR_NAME


metric_types.register_metric(TNR)


class FallOut(ConfusionMatrixMetric):
  """Fall-out (FPR)."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes fall-out metric.

    Args:
      thresholds: (Optional) Thresholds to use for fall-out. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return FALL_OUT_NAME

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    del tp, fn
    denominator = fp + tn
    if denominator > 0.0:
      return fp / denominator
    else:
      return float('nan')


metric_types.register_metric(FallOut)


class FPR(FallOut):
  """Alias for FallOut."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes FPR metric."""
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return FPR_NAME


metric_types.register_metric(FPR)


class MissRate(ConfusionMatrixMetric):
  """Miss rate (FNR)."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes miss rate metric.

    Args:
      thresholds: (Optional) Thresholds to use for miss rate. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return MISS_RATE_NAME

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    del tn, fp
    denominator = fn + tp
    if denominator > 0.0:
      return fn / denominator
    else:
      return float('nan')


metric_types.register_metric(MissRate)


class FNR(MissRate):
  """Alias for MissRate."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes FNR metric."""
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return FNR_NAME


metric_types.register_metric(FNR)


class NegativePredictiveValue(ConfusionMatrixMetric):
  """Negative predictive value (NPV)."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes negative predictive value.

    Args:
      thresholds: (Optional) Thresholds to use. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return NEGATIVE_PREDICTIVE_VALUE_NAME

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    del tp, fp
    denominator = tn + fn
    if denominator > 0.0:
      return tn / denominator
    else:
      return float('nan')


metric_types.register_metric(NegativePredictiveValue)


class NPV(NegativePredictiveValue):
  """Alias for NegativePredictiveValue."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes PPV metric."""
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return NPV_NAME


metric_types.register_metric(NPV)


class FalseDiscoveryRate(ConfusionMatrixMetric):
  """False discovery rate (FDR)."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes false discovery rate.

    Args:
      thresholds: (Optional) Thresholds to use. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return FALSE_DISCOVERY_RATE_NAME

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
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes false omission rate.

    Args:
      thresholds: (Optional) Thresholds to use. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return FALSE_OMISSION_RATE_NAME

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
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes prevalence.

    Args:
      thresholds: (Optional) Thresholds to use. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return PREVALENCE_NAME

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
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes prevalence threshold.

    Args:
      thresholds: (Optional) Thresholds to use. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return PREVALENCE_THRESHOLD_NAME

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
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes threat score.

    Args:
      thresholds: (Optional) Thresholds to use. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return THREAT_SCORE_NAME

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
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes balanced accuracy.

    Args:
      thresholds: (Optional) Thresholds to use. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return BALANCED_ACCURACY_NAME

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
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes F1 score.

    Args:
      thresholds: (Optional) Thresholds to use. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return F1_SCORE_NAME

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


class MatthewsCorrelationCoefficient(ConfusionMatrixMetric):
  """Matthews corrrelation coefficient (MCC)."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes matthews corrrelation coefficient.

    Args:
      thresholds: (Optional) Thresholds to use. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return MATTHEWS_CORRELATION_COEFFICIENT_NAME

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    denominator = _pos_sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator > 0.0:
      return (tp * tn - fp * fn) / denominator
    else:
      return float('nan')


metric_types.register_metric(MatthewsCorrelationCoefficient)


class FowlkesMallowsIndex(ConfusionMatrixMetric):
  """Fowlkes-Mallows index (FM)."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes fowlkes-mallows index.

    Args:
      thresholds: (Optional) Thresholds to use. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return FOWLKES_MALLOWS_INDEX_NAME

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
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes informedness.

    Args:
      thresholds: (Optional) Thresholds to use. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return INFORMEDNESS_NAME

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
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes markedness.

    Args:
      thresholds: (Optional) Thresholds to use. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return MARKEDNESS_NAME

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
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes positive likelihood ratio.

    Args:
      thresholds: (Optional) Thresholds to use. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return POSITIVE_LIKELIHOOD_RATIO_NAME

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
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes negative likelihood ratio.

    Args:
      thresholds: (Optional) Thresholds to use. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return NEGATIVE_LIKELIHOOD_RATIO_NAME

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
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes diagnostic odds ratio.

    Args:
      thresholds: (Optional) Thresholds to use. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return DIAGNOSTIC_ODDS_RATIO_NAME

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    if fn > 0.0 and fp > 0.0 and tn > 0.0:
      return (tp / fn) / (fp / tn)
    else:
      return float('nan')


metric_types.register_metric(DiagnosticOddsRatio)


class PredictedPositiveRate(ConfusionMatrixMetric):
  """Predicted positive rate."""

  def __init__(self,
               thresholds: Optional[Union[float, List[float]]] = None,
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes predicted positive rate.

    Args:
      thresholds: (Optional) Thresholds to use. Defaults to [0.5].
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        thresholds=thresholds, name=name, top_k=top_k, class_id=class_id)

  def _default_name(self) -> str:
    return PREDICTED_POSITIVE_RATE_NAME

  def result(self, tp: float, tn: float, fp: float, fn: float) -> float:
    total_count = tp + fp + tn + fn
    if total_count:
      predicted_positives = tp + fp
      return predicted_positives / total_count
    else:
      return float('nan')


metric_types.register_metric(PredictedPositiveRate)


class ConfusionMatrixAtThresholds(metric_types.Metric):
  """Confusion matrix at thresholds."""

  def __init__(self,
               thresholds: List[float],
               name: Optional[str] = None,
               top_k: Optional[int] = None,
               class_id: Optional[int] = None):
    """Initializes confusion matrix at thresholds.

    Args:
      thresholds: Thresholds to use for confusion matrix.
      name: (Optional) Metric name.
      top_k: (Optional) Used with a multi-class model to specify that the top-k
        values should be used to compute the confusion matrix. The net effect is
        that the non-top-k values are set to -inf and the matrix is then
        constructed from the average TP, FP, TN, FN across the classes. When
        top_k is used, metrics_specs.binarize settings must not be present. Only
        one of class_id or top_k should be configured.
      class_id: (Optional) Used with a multi-class model to specify which class
        to compute the confusion matrix for. When class_id is used,
        metrics_specs.binarize settings must not be present. Only one of
        class_id or top_k should be configured.
    """
    super().__init__(
        metric_util.merge_per_key_computations(self._metric_computations),
        thresholds=thresholds,
        name=name,
        top_k=top_k,
        class_id=class_id)

  def _default_name(self) -> str:
    return CONFUSION_MATRIX_AT_THRESHOLDS_NAME

  def _metric_computations(
      self,
      thresholds: List[float],
      top_k: Optional[int] = None,
      class_id: Optional[int] = None,
      name: Optional[str] = None,
      eval_config: Optional[config_pb2.EvalConfig] = None,
      model_name: str = '',
      output_name: str = '',
      sub_key: Optional[metric_types.SubKey] = None,
      aggregation_type: Optional[metric_types.AggregationType] = None,
      class_weights: Optional[Dict[int, float]] = None,
      example_weighted: bool = False) -> metric_types.MetricComputations:
    """Returns metric computations for confusion matrix at thresholds."""
    sub_key = _validate_and_update_sub_key(name, model_name, output_name,
                                           sub_key, top_k, class_id)
    key = metric_types.MetricKey(
        name=name,
        model_name=model_name,
        output_name=output_name,
        sub_key=sub_key,
        example_weighted=example_weighted,
        aggregation_type=aggregation_type)

    # Make sure matrices are calculated.
    matrices_computations = binary_confusion_matrices.binary_confusion_matrices(
        eval_config=eval_config,
        model_name=model_name,
        output_name=output_name,
        sub_key=sub_key,
        aggregation_type=aggregation_type,
        class_weights=class_weights,
        thresholds=thresholds,
        example_weighted=example_weighted)
    matrices_key = matrices_computations[-1].keys[-1]

    def result(
        metrics: Dict[metric_types.MetricKey,
                      binary_confusion_matrices.Matrices]
    ) -> Dict[metric_types.MetricKey, Any]:
      return {key: metrics[matrices_key]}

    derived_computation = metric_types.DerivedMetricComputation(
        keys=[key], result=result)
    computations = matrices_computations
    computations.append(derived_computation)
    return computations


metric_types.register_metric(ConfusionMatrixAtThresholds)
