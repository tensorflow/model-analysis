# Copyright 2021 Google LLC
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
"""Flip count metric."""

import collections

from typing import Any, Dict, List, Optional, Iterator, Sequence, Tuple

import numpy as np
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import model_util
from tensorflow_model_analysis.utils import util

FLIP_COUNT_NAME = 'flip_count'

# Default list of threshold values used to compute flips.
DEFAULT_THRESHOLDS = (0.5,)

# Default number of example ids returned for each type of flip.
DEFAULT_NUM_EXAMPLE_IDS = 100

# List of sub-metrics computed.
METRICS_LIST = [
    'negative_to_positive',
    'negative_to_positive_examples_ids',
    'negative_examples_count',
    'positive_examples_count',
    'positive_to_negative',
    'positive_to_negative_examples_ids',
]


class FlipCount(metric_types.Metric):
  """Flip count metric.

  Calculates the number of flips in prediction values for original and
  counterfactual examples (which have sensitive terms swapped, ablated etc.)

  It calculates:
  - positive to negative flips
  - negative to positive flips

  Other than flip counts, it also provides sample of example ids for which
  those flips are happening.
  """

  def __init__(self,
               counterfactual_prediction_key: Optional[str] = None,
               example_id_key: Optional[str] = None,
               example_ids_count: int = DEFAULT_NUM_EXAMPLE_IDS,
               name: str = FLIP_COUNT_NAME,
               thresholds: Sequence[float] = DEFAULT_THRESHOLDS):
    """Initializes flip count.

    Args:
      counterfactual_prediction_key: Prediction label key for counterfactual
        example to be used to measure the flip count if the counterfactual
        example is contained within features. Otherwise the baseline model
        prediction will be used as the counterfactual.
      example_id_key: Feature key containing example id.
      example_ids_count: Max number of example ids to be extracted for false
        positives and false negatives.
      name: Metric name.
      thresholds: Thresholds to be used to measure flips.
    """
    super().__init__(
        metric_util.merge_per_key_computations(flip_count),
        name=name,
        thresholds=thresholds,
        counterfactual_prediction_key=counterfactual_prediction_key,
        example_id_key=example_id_key,
        example_ids_count=example_ids_count)

  @property
  def compute_confidence_interval(self) -> bool:
    """Confidence intervals for FlipCount.

    Confidence intervals capture uncertainty in a metric if it were computed on
    more examples. For now, confidence interval has been disabled.
    Returns:
      Whether to compute confidence intervals.
    """
    return False


metric_types.register_metric(FlipCount)


def _calculate_digits(thresholds):
  digits = [len(str(t)) - 2 for t in thresholds]
  return max(max(digits), 1)


def create_metric_keys(
    thresholds: Sequence[float], metrics: List[str], metric_name: str,
    model_name: str, output_name: str, example_weighted: bool
) -> Tuple[List[metric_types.MetricKey], Dict[float, Dict[
    str, metric_types.MetricKey]]]:
  """Creates metric keys map keyed at threshold and metric name."""
  keys = []
  metric_key_by_name_by_threshold = collections.defaultdict(dict)
  num_digits = _calculate_digits(thresholds)
  for threshold in thresholds:
    for metric in metrics:
      key = metric_types.MetricKey(
          name='%s/%s@%.*f' % (metric_name, metric, num_digits, threshold),
          model_name=model_name,
          output_name=output_name,
          example_weighted=example_weighted)
      keys.append(key)
      metric_key_by_name_by_threshold[threshold][metric] = key
  return keys, metric_key_by_name_by_threshold


def flip_count(
    counterfactual_prediction_key: Optional[str] = None,
    example_id_key: Optional[str] = None,
    example_ids_count: int = DEFAULT_NUM_EXAMPLE_IDS,
    name: str = FLIP_COUNT_NAME,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    model_name: str = '',
    output_name: str = '',
    eval_config: Optional[config_pb2.EvalConfig] = None,
    example_weighted: bool = False) -> metric_types.MetricComputations:
  """Returns metric computations for computing flip counts."""
  keys, metric_key_by_name_by_threshold = create_metric_keys(
      thresholds, METRICS_LIST, name, model_name, output_name, example_weighted)

  feature_keys = [counterfactual_prediction_key]
  if example_id_key:
    feature_keys.append(example_id_key)

  def extract_label_prediction_and_weight(
      inputs: metric_types.StandardMetricInputs,
      eval_config: Optional[config_pb2.EvalConfig] = None,
      model_name: str = '',
      output_name: str = '',
      sub_key: Optional[metric_types.SubKey] = None,
      aggregation_type: Optional[metric_types.AggregationType] = None,
      class_weights: Optional[Dict[int, float]] = None,
      example_weighted: bool = False,
      fractional_labels: bool = False,
      flatten: bool = True,
  ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yields label, prediction, and example weights to be used in calculations.

    This function is a customized metric_util.to_label_prediction_example_weight
    function which yields original prediction as label and counterfactual
    prediction as prediction and derive flip count metrics from false positives
    and false negatives of binary confusion matrix.

    Args:
      inputs: Standard metric inputs.
      eval_config: Eval config
      model_name: Optional model name (if multi-model evaluation).
      output_name: Optional output name (if multi-output model type).
      sub_key: Optional sub key. (unused)
      aggregation_type: Optional aggregation type. (unused)
      class_weights: Optional class weights to apply to multi-class /
        multi-label labels and predictions. (unused)
      example_weighted: True if example weights should be applied.
      fractional_labels: If true, each incoming tuple of (label, prediction,
        example weight) will be split into two tuples as follows (where l, p, w
        represent the resulting label, prediction, and example weight values):
          (1) l = 0.0, p = prediction, and w = example_weight * (1.0 - label)
          (2) l = 1.0, p = prediction, and w = example_weight * label If
          enabled, an exception will be raised if labels are not within [0, 1].
          The implementation is such that tuples associated with a weight of
          zero are not yielded. This means it is safe to enable
          fractional_labels even when the labels only take on the values of 0.0
          or 1.0. (unused)
      flatten: True to flatten the final label and prediction outputs so that
        the yielded values are always arrays of size 1. For example, multi-class
        /multi-label outputs would be converted into label and prediction pairs
        that could then be processed by a binary classification metric in order
        to compute a micro average over all classes. (unused)

    Yields:
      Tuple of (label, prediction, example_weight).

    Raises:
      ValueError: If counterfactual prediction key is not found within either
        the features or predictions.
      ValueError: If predictions is None or empty.
    """
    del (sub_key, aggregation_type, class_weights, fractional_labels, flatten
        )  # unused

    # TODO(sokeefe): Look into removing the options to pass counterfactual
    # predictions in a feature and instead as a baseline model.
    if (counterfactual_prediction_key is not None and
        counterfactual_prediction_key in inputs.features):
      counterfactual_prediction = inputs.features[counterfactual_prediction_key]
    elif eval_config is not None:
      counterfactual_model_spec = model_util.get_baseline_model_spec(
          eval_config)
      if counterfactual_model_spec is not None:
        _, counterfactual_prediction, _ = next(
            metric_util.to_label_prediction_example_weight(
                inputs,
                eval_config=eval_config,
                model_name=counterfactual_model_spec.name,
                output_name=output_name,
                example_weighted=example_weighted,
                fractional_labels=False,  # Labels are ignored for flip counts.
                flatten=False,  # Flattened below
                allow_none=True,  # Allow None labels
                require_single_example_weight=True))
      else:
        raise ValueError('The Counterfactual model must be listed with '
                         f'`is_baseline` equal to `True`. Found: {eval_config}')
    else:
      raise ValueError(
          '`counterfactual_prediction` was not found within the provided '
          'inputs. It must be included as either a feature key or within the '
          'predictions. Found:\n'
          f'`counterfactual_prediction_key`: {counterfactual_prediction_key}\n'
          f'`inputs.prediction`:{inputs.prediction}')

    if counterfactual_prediction is None:
      raise ValueError(
          '%s feature key is None (required for FlipCount metric)' %
          counterfactual_prediction_key)

    def get_by_keys(value: Any, keys: List[str]) -> Any:
      if isinstance(value, dict):
        new_value = util.get_by_keys(value, keys, optional=True)
        if new_value is not None:
          return new_value
      return value

    if model_name:
      counterfactual_prediction = get_by_keys(counterfactual_prediction,
                                              [model_name])
    if output_name:
      counterfactual_prediction = get_by_keys(counterfactual_prediction,
                                              [output_name])

    _, prediction, example_weight = next(
        metric_util.to_label_prediction_example_weight(
            inputs,
            eval_config=eval_config,
            model_name=model_name,
            output_name=output_name,
            example_weighted=example_weighted,
            fractional_labels=False,  # Labels are ignored for flip counts.
            flatten=False,  # Flattened below
            allow_none=True,  # Allow None labels
            require_single_example_weight=True))

    if prediction.size != counterfactual_prediction.size:
      raise ValueError(
          'prediction and counterfactual_prediction size should be same for '
          'FlipCount metric, %f != %f' %
          (prediction.size, counterfactual_prediction.size))

    if prediction.size == 0:
      raise ValueError('prediction is empty (required for FlipCount metric)')
    else:  # Always flatten
      example_weight = np.array(
          [float(example_weight) for i in range(prediction.shape[-1])])
      for p, cfp, w in zip(prediction.flatten(),
                           counterfactual_prediction.flatten(),
                           example_weight.flatten()):
        yield np.array([p]), np.array([cfp]), np.array([w])

  # Setting fractional label to false, since prediction is being used as label
  # and it can be a non-binary value.
  computations = binary_confusion_matrices.binary_confusion_matrices(
      thresholds=list(thresholds),
      eval_config=eval_config,
      model_name=model_name,
      output_name=output_name,
      example_weighted=example_weighted,
      extract_label_prediction_and_weight=extract_label_prediction_and_weight,
      preprocessor=metric_types.FeaturePreprocessor(feature_keys=feature_keys),
      example_id_key=example_id_key,
      example_ids_count=example_ids_count,
      fractional_labels=False)
  examples_metric_key, matrices_metric_key = computations[-1].keys

  def result(
      metrics: Dict[metric_types.MetricKey, Any]
  ) -> Dict[metric_types.MetricKey, Any]:
    """Returns flip count metrics values."""
    matrix = metrics[matrices_metric_key]
    examples = metrics[examples_metric_key]

    output = {}
    for i, threshold in enumerate(matrix.thresholds):
      output[metric_key_by_name_by_threshold[threshold]
             ['positive_to_negative']] = matrix.fn[i]
      output[metric_key_by_name_by_threshold[threshold]
             ['negative_to_positive']] = matrix.fp[i]
      output[metric_key_by_name_by_threshold[threshold]
             ['positive_to_negative_examples_ids']] = np.array(
                 examples.fn_examples[i])
      output[metric_key_by_name_by_threshold[threshold]
             ['negative_to_positive_examples_ids']] = np.array(
                 examples.fp_examples[i])
      output[metric_key_by_name_by_threshold[threshold]
             ['positive_examples_count']] = matrix.fn[i] + matrix.tp[i]
      output[metric_key_by_name_by_threshold[threshold]
             ['negative_examples_count']] = matrix.fp[i] + matrix.tn[i]

    return output

  derived_computation = metric_types.DerivedMetricComputation(
      keys=keys, result=result)

  computations.append(derived_computation)
  return computations


metric_types.register_metric(FlipCount)
