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
"""Public API for performing query-based metrics evaluations."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

# Standard Imports

import apache_beam as beam
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import constants as eval_saved_model_constants
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.eval_saved_model import util as eval_saved_model_util
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.evaluators.query_metrics import query_types
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from typing import Any, Dict, List, Optional, Text, Tuple


def QueryBasedMetricsEvaluator(  # pylint: disable=invalid-name
    query_id: Text,
    prediction_key: Text,
    combine_fns: List[beam.CombineFn],
    metrics_key: Text = constants.METRICS_KEY,
    run_after: Text = slice_key_extractor.SLICE_KEY_EXTRACTOR_STAGE_NAME,
) -> evaluator.Evaluator:
  """Creates an Evaluator for evaluating metrics and plots.

  Args:
    query_id: Key of query ID column in the features dictionary.
    prediction_key: Key in predictions dictionary to use as the prediction (for
      sorting examples within the query). Use the empty string if the Estimator
      returns a predictions Tensor (not a dictionary).
    combine_fns: List of query based metrics combine functions.
    metrics_key: Name to use for metrics key in Evaluation output.
    run_after: Extractor to run after (None means before any extractors).

  Returns:
    Evaluator for computing query-based metrics. The output will be stored under
    'metrics' and 'plots' keys.
  """
  # pylint: disable=no-value-for-parameter
  return evaluator.Evaluator(
      stage_name='EvaluateQueryBasedMetrics',
      run_after=run_after,
      ptransform=EvaluateQueryBasedMetrics(
          query_id=query_id,
          prediction_key=prediction_key,
          combine_fns=combine_fns,
          metrics_key=metrics_key))
  # pylint: enable=no-value-for-parameter


class CreateQueryExamples(beam.CombineFn):
  """CombineFn to create query examples for each query id.

  Note that this assumes the number of examples for each query ID is small
  enough to fit in memory.
  """

  def __init__(self, prediction_key: Text):
    if not prediction_key:
      # If prediction key is set to the empty string, the user is telling us
      # that their Estimator returns a predictions Tensor rather than a
      # dictionary. Set the key to the magic key we use in that case.
      self._prediction_key = eval_saved_model_util.default_dict_key(
          eval_saved_model_constants.PREDICTIONS_NAME)
    else:
      self._prediction_key = prediction_key

  def create_accumulator(self):
    return []

  def add_input(self, accumulator: List[types.Extracts],
                extract: types.Extracts) -> List[types.Extracts]:
    accumulator.append(extract)
    return accumulator

  def merge_accumulators(self, accumulators: List[List[types.Extracts]]
                        ) -> List[types.Extracts]:
    result = []
    for acc in accumulators:
      result.extend(acc)
    return result

  def extract_output(self,
                     accumulator: List[types.Extracts]) -> query_types.QueryFPL:

    def fpl_from_extracts(extract: types.Extracts) -> query_types.FPL:  # pylint: disable=invalid-name
      """Make an FPL from an extract."""

      fpl = extract[constants.FEATURES_PREDICTIONS_LABELS_KEY]

      # Unwrap the FPL to get dictionaries like
      # features['feature1'] = [['ex1value1', 'ex1value2'], ['ex2value1', '']]
      # features['feature2'] = [[1], [2]]
      features = {k: v[encoding.NODE_SUFFIX] for k, v in fpl.features.items()}
      predictions = {
          k: v[encoding.NODE_SUFFIX] for k, v in fpl.predictions.items()
      }
      labels = {k: v[encoding.NODE_SUFFIX] for k, v in fpl.labels.items()}

      return dict(features=features, predictions=predictions, labels=labels)

    unsorted_fpls = [fpl_from_extracts(extract) for extract in accumulator]

    if not unsorted_fpls:
      return query_types.QueryFPL(fpls=[], query_id='')

    # Sort result in descending order of prediction
    sort_keys = np.array(
        [x['predictions'][self._prediction_key][0][0] for x in unsorted_fpls])
    sort_indices = np.argsort(sort_keys)[::-1]

    sorted_fpls = []
    for index in sort_indices:
      sorted_fpls.append(unsorted_fpls[index])

    # Query ID will be filled in later
    return query_types.QueryFPL(fpls=sorted_fpls, query_id='')


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
def ComputeQueryBasedMetrics(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    prediction_key: Text,
    query_id: Text,
    combine_fns: List[beam.CombineFn],
) -> beam.pvalue.PCollection:
  """Computes metrics and plots using the EvalSavedModel.

  Args:
    extracts: PCollection of Extracts. The extracts MUST contain a
      FeaturesPredictionsLabels extract keyed by
      tfma.FEATURE_PREDICTIONS_LABELS_KEY and a list of SliceKeyType extracts
      keyed by tfma.SLICE_KEY_TYPES_KEY. Typically these will be added by
      calling the default_extractors function.
    prediction_key: Key in predictions dictionary to use as the prediction (for
      sorting examples within the query). Use the empty string if the Estimator
      returns a predictions Tensor (not a dictionary).
    query_id: Key of query ID column in the features dictionary.
    combine_fns: List of query based metrics combine functions.

  Returns:
    PCollection of (slice key, query-based metrics).
  """

  missing_query_id_counter = beam.metrics.Metrics.counter(
      constants.METRICS_NAMESPACE, 'missing_query_id')

  def key_by_query_id(extract: types.Extracts,
                      query_id: Text) -> Optional[Tuple[Text, types.Extracts]]:
    """Extract the query ID from the extract and key by that."""
    features = extract[constants.FEATURES_PREDICTIONS_LABELS_KEY].features
    if query_id not in features:
      missing_query_id_counter.inc()
      return None
    feature_value = features[query_id][encoding.NODE_SUFFIX]
    if isinstance(feature_value, tf.compat.v1.SparseTensorValue):
      feature_value = feature_value.values
    if feature_value.size != 1:
      raise ValueError('Query ID feature "%s" should have exactly 1 value, but '
                       'found %d instead. Values were: %s' %
                       (query_id, feature_value.size(), feature_value))
    return ('{}'.format(np.asscalar(feature_value)), extract)

  def merge_dictionaries(dictionaries: Tuple[Dict[Text, Any], ...]
                        ) -> Dict[Text, Any]:
    """Merge dictionaries in a tuple into a single dictionary."""
    result = dict()
    for d in dictionaries:
      intersection = set(d.keys()) & set(result.keys())
      if intersection:
        raise ValueError('Overlapping keys found when merging dictionaries. '
                         'Intersection was: %s. Keys up to this point: %s '
                         'keys from next dictionary: %s' %
                         (intersection, result.keys(), d.keys()))
      result.update(d)
    return result

  # pylint: disable=no-value-for-parameter
  return (extracts
          | 'KeyByQueryId' >> beam.Map(key_by_query_id, query_id)
          | 'CreateQueryExamples' >> beam.CombinePerKey(
              CreateQueryExamples(prediction_key=prediction_key))
          | 'DropQueryId' >> beam.Map(lambda kv: kv[1]._replace(query_id=kv[0]))
          | 'CombineGlobally' >> beam.CombineGlobally(
              beam.combiners.SingleInputTupleCombineFn(*combine_fns))
          | 'MergeDictionaries' >> beam.Map(merge_dictionaries)
          | 'AddOverallSliceKey' >> beam.Map(lambda v: ((), v)))
  # pylint: enable=no-value-for-parameter


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(evaluator.Evaluation)
def EvaluateQueryBasedMetrics(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    prediction_key: Text,
    query_id: Text,
    combine_fns: List[beam.CombineFn],
    metrics_key: Text = constants.METRICS_KEY,
) -> evaluator.Evaluation:
  """Evaluates query-based metrics.

  Args:
    extracts: PCollection of Extracts. The extracts MUST contain a
      FeaturesPredictionsLabels extract keyed by
      tfma.FEATURE_PREDICTION_LABELS_KEY and a list of SliceKeyType extracts
      keyed by tfma.SLICE_KEY_TYPES_KEY. Typically these will be added by
      calling the default_extractors function.
    prediction_key: Key in predictions dictionary to use as the prediction (for
      sorting examples within the query). Use the empty string if the Estimator
      returns a predictions Tensor (not a dictionary).
    query_id: Key of query ID column in the features dictionary.
    combine_fns: List of query based metrics combine functions.
    metrics_key: Name to use for metrics key in Evaluation output.

  Returns:
    Evaluation containing metrics dictionaries keyed by 'metrics'.
  """

  # pylint: disable=no-value-for-parameter
  metrics = (
      extracts
      | 'Filter' >> extractor.Filter(include=[
          constants.FEATURES_PREDICTIONS_LABELS_KEY,
          constants.SLICE_KEY_TYPES_KEY
      ])
      | 'ComputeQueryBasedMetrics' >> ComputeQueryBasedMetrics(
          query_id=query_id,
          combine_fns=combine_fns,
          prediction_key=prediction_key))
  # pylint: enable=no-value-for-parameter

  return {metrics_key: metrics}
