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
"""Public API for performing metrics and plots evaluations."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

# Standard Imports

import apache_beam as beam
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.evaluators import aggregate
from tensorflow_model_analysis.evaluators import counter_util
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.slicer import slicer
from tensorflow_model_analysis.writers import metrics_and_plots_serialization
from typing import Any, Dict, Optional, Text, Tuple, Generator


# TODO(mdreves): Perhaps keep this as the only public method and privatize
# several other PTransforms and functions in this modoule (and other parts of
# TFMA).
def MetricsAndPlotsEvaluator(  # pylint: disable=invalid-name
    eval_shared_model: types.EvalSharedModel,
    desired_batch_size: Optional[int] = None,
    metrics_key: Text = constants.METRICS_KEY,
    plots_key: Text = constants.PLOTS_KEY,
    run_after: Text = slice_key_extractor.SLICE_KEY_EXTRACTOR_STAGE_NAME,
    compute_confidence_intervals: Optional[bool] = False,
    k_anonymization_count: int = 1,
    serialize=False) -> evaluator.Evaluator:
  """Creates an Evaluator for evaluating metrics and plots.

  Args:
    eval_shared_model: Shared model parameters for EvalSavedModel.
    desired_batch_size: Optional batch size for batching in Aggregate.
    metrics_key: Name to use for metrics key in Evaluation output.
    plots_key: Name to use for plots key in Evaluation output.
    run_after: Extractor to run after (None means before any extractors).
    compute_confidence_intervals: Whether or not to compute confidence
      intervals.
    k_anonymization_count: If the number of examples in a specific slice is less
      than k_anonymization_count, then an error will be returned for that slice.
      This will be useful to ensure privacy by not displaying the aggregated
      data for smaller number of examples.
    serialize: If true, serialize the metrics to protos as part of the
      evaluation as well.

  Returns:
    Evaluator for evaluating metrics and plots. The output will be stored under
    'metrics' and 'plots' keys.
  """
  # pylint: disable=no-value-for-parameter
  return evaluator.Evaluator(
      stage_name='EvaluateMetricsAndPlots',
      run_after=run_after,
      ptransform=EvaluateMetricsAndPlots(
          eval_shared_model=eval_shared_model,
          desired_batch_size=desired_batch_size,
          metrics_key=metrics_key,
          plots_key=plots_key,
          compute_confidence_intervals=compute_confidence_intervals,
          k_anonymization_count=k_anonymization_count,
          serialize=serialize))


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
# No typehint for output type, since it's a multi-output DoFn result that
# Beam doesn't support typehints for yet (BEAM-3280).
def ComputeMetricsAndPlots(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    eval_shared_model: types.EvalSharedModel,
    desired_batch_size: Optional[int] = None,
    compute_confidence_intervals: Optional[bool] = False,
    random_seed_for_testing: Optional[int] = None
) -> Tuple[beam.pvalue.DoOutputsTuple, beam.pvalue.PCollection]:
  """Computes metrics and plots using the EvalSavedModel.

  Args:
    extracts: PCollection of Extracts. The extracts MUST contain a
      FeaturesPredictionsLabels extract keyed by
      tfma.FEATURE_PREDICTIONS_LABELS_KEY and a list of SliceKeyType extracts
      keyed by tfma.SLICE_KEY_TYPES_KEY. Typically these will be added by
      calling the default_extractors function.
    eval_shared_model: Shared model parameters for EvalSavedModel including any
      additional metrics (see EvalSharedModel for more information on how to
      configure additional metrics).
    desired_batch_size: Optional batch size for batching in Aggregate.
    compute_confidence_intervals: Set to True to run metrics analysis over
      multiple bootstrap samples and compute uncertainty intervals.
    random_seed_for_testing: Provide for deterministic tests only.

  Returns:
    Tuple of Tuple[PCollection of (slice key, metrics),
    PCollection of (slice key, plot metrics)] and
    PCollection of (slice_key and its example count).
  """
  # pylint: disable=no-value-for-parameter

  _ = (
      extracts.pipeline
      | counter_util.IncrementMetricsComputationCounters(
          eval_shared_model.add_metrics_callbacks))

  slices = (
      extracts
      # Downstream computation only cares about FPLs, so we prune before fanout.
      # Note that fanout itself will prune the slice keys.
      | 'PruneExtracts' >> extractor.Filter(include=[
          constants.FEATURES_PREDICTIONS_LABELS_KEY,
          constants.SLICE_KEY_TYPES_KEY
      ])
      # Input: one example at a time, with slice keys in extracts.
      # Output: one fpl example per slice key (notice that the example turns
      #         into n logical examples, references to which are replicated once
      #         per applicable slice key).
      | 'FanoutSlices' >> slicer.FanoutSlices())

  slices_count = (
      slices
      | 'ExtractSliceKeys' >> beam.Keys()
      | 'CountPerSliceKey' >> beam.combiners.Count.PerElement())

  aggregated_metrics = (
      slices
      # Metrics are computed per slice key.
      # Output: Multi-outputs, a dict of slice key to computed metrics, and
      # plots if applicable.
      | 'ComputePerSliceMetrics' >> aggregate.ComputePerSliceMetrics(
          eval_shared_model=eval_shared_model,
          desired_batch_size=desired_batch_size,
          compute_confidence_intervals=compute_confidence_intervals,
          random_seed_for_testing=random_seed_for_testing))

  return (aggregated_metrics, slices_count)


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(evaluator.Evaluation)
def EvaluateMetricsAndPlots(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    eval_shared_model: types.EvalSharedModel,
    desired_batch_size: Optional[int] = None,
    metrics_key: Text = constants.METRICS_KEY,
    plots_key: Text = constants.PLOTS_KEY,
    compute_confidence_intervals: Optional[bool] = False,
    k_anonymization_count: int = 1,
    serialize: bool = False) -> evaluator.Evaluation:
  """Evaluates metrics and plots using the EvalSavedModel.

  Args:
    extracts: PCollection of Extracts. The extracts MUST contain a
      FeaturesPredictionsLabels extract keyed by
      tfma.FEATURE_PREDICTION_LABELS_KEY and a list of SliceKeyType extracts
      keyed by tfma.SLICE_KEY_TYPES_KEY. Typically these will be added by
      calling the default_extractors function.
    eval_shared_model: Shared model parameters for EvalSavedModel including any
      additional metrics (see EvalSharedModel for more information on how to
      configure additional metrics).
    desired_batch_size: Optional batch size for batching in Aggregate.
    metrics_key: Name to use for metrics key in Evaluation output.
    plots_key: Name to use for plots key in Evaluation output.
    compute_confidence_intervals: Whether or not to compute confidence
      intervals.
    k_anonymization_count: If the number of examples in a specific slice is less
      than k_anonymization_count, then an error will be returned for that slice.
      This will be useful to ensure privacy by not displaying the aggregated
      data for smaller number of examples.
    serialize: If true, serialize the metrics to protos as part of the
      evaluation as well.

  Returns:
    Evaluation containing metrics and plots dictionaries keyed by 'metrics'
    and 'plots'.
  """
  # pylint: disable=no-value-for-parameter

  (metrics, plots), slices_count = (
      extracts
      | 'ComputeMetricsAndPlots' >> ComputeMetricsAndPlots(
          eval_shared_model,
          desired_batch_size,
          compute_confidence_intervals=compute_confidence_intervals))

  if k_anonymization_count > 1:
    metrics = (
        metrics
        | 'FilterMetricsForSmallSlices' >> _FilterOutSlices(
            slices_count, k_anonymization_count))
    plots = (
        plots
        | 'FilterPlotsForSmallSlices' >> _FilterOutSlices(
            slices_count, k_anonymization_count))

  if serialize:
    metrics, plots = (
        (metrics, plots)
        | 'SerializeMetricsAndPlots' >>
        metrics_and_plots_serialization.SerializeMetricsAndPlots(
            post_export_metrics=eval_shared_model.add_metrics_callbacks))

  return {metrics_key: metrics, plots_key: plots}


@beam.ptransform_fn
@beam.typehints.with_input_types(Tuple[slicer.SliceKeyType, types.Extracts])
@beam.typehints.with_output_types(Tuple[slicer.SliceKeyType, types.Extracts])
def _FilterOutSlices(  # pylint: disable=invalid-name
    values: beam.pvalue.PCollection, slices_count: beam.pvalue.PCollection,
    k_anonymization_count: int) -> beam.pvalue.PCollection:
  """Filter out slices with examples count lower than k_anonymization_count.

  Since we might filter out certain slices to preserve privacy in the case of
  small slices, to make end users aware of this, we will append filtered out
  slice keys with empty data, and a debug message explaining the omission.

  Args:
    values: PCollection of aggregated data keyed at slice_key
    slices_count: PCollection of slice keys and their example count.
    k_anonymization_count: If the number of examples in a specific slice is less
      than k_anonymization_count, then an error will be returned for that slice.
      This will be useful to ensure privacy by not displaying the aggregated
      data for smaller number of examples.

  Returns:
    A PCollection keyed at all the possible slice_key and aggregated data for
    slice keys with example count more than k_anonymization_count and error
    message for filtered out slices.
  """

  class FilterOutSmallSlicesDoFn(beam.DoFn):
    """DoFn to filter out small slices.

    For slices (excluding overall slice) with examples count lower than
    k_anonymization_count, it adds an error message.

    Args:
      element: Tuple containing slice key and a dictionary containing
        corresponding elements from merged pcollections.

    Returns:
      PCollection of (slice_key, aggregated_data or error message)
    """

    def process(
        self, element: Tuple[slicer.SliceKeyType, Dict[Text, Any]]
    ) -> Generator[Tuple[slicer.SliceKeyType, Dict[Text, Any]], None, None]:
      (slice_key, value) = element
      if value['values']:
        if (not slice_key or value['slices_count'][0] >= k_anonymization_count):
          yield (slice_key, value['values'][0])
        else:
          yield (slice_key, {
              metric_keys.ERROR_METRIC:
                  'Example count for this slice key is lower than '
                  'the minimum required value: %d. No data is aggregated for '
                  'this slice.' % k_anonymization_count
          })

  return ({
      'values': values,
      'slices_count': slices_count
  }
          | 'CoGroupingSlicesCountAndAggregatedData' >> beam.CoGroupByKey()
          | 'FilterOutSmallSlices' >> beam.ParDo(FilterOutSmallSlicesDoFn()))
