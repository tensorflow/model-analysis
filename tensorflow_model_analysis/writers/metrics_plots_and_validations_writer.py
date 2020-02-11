# Lint as: python3
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
"""Metrics, plots, and validations writer."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Dict, Optional, List, Text

import apache_beam as beam
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.proto import validation_result_pb2
from tensorflow_model_analysis.writers import metrics_and_plots_serialization
from tensorflow_model_analysis.writers import writer


def MetricsPlotsAndValidationsWriter(
    output_paths: Dict[Text, Text],
    add_metrics_callbacks: List[types.AddMetricsCallbackType],
    metrics_key: Text = constants.METRICS_KEY,
    plots_key: Text = constants.PLOTS_KEY,
    validations_key: Text = constants.VALIDATIONS_KEY) -> writer.Writer:
  """Returns metrics and plots writer.

  Args:
    output_paths: Output paths keyed by output key (e.g. 'metrics', 'plots').
    add_metrics_callbacks: Optional list of metric callbacks (if used).
    metrics_key: Name to use for metrics key in Evaluation output.
    plots_key: Name to use for plots key in Evaluation output.
    validations_key: Name to use for validations key in Evaluation output.
  """
  return writer.Writer(
      stage_name='WriteMetricsAndPlots',
      ptransform=_WriteMetricsPlotsAndValidations(  # pylint: disable=no-value-for-parameter
          output_paths=output_paths,
          add_metrics_callbacks=add_metrics_callbacks,
          metrics_key=metrics_key,
          plots_key=plots_key,
          validations_key=validations_key))


def _SerializeValidations(
    validations: validation_result_pb2.ValidationResult) -> bytes:
  """Converts the given validations into serialized proto ValidationResult."""
  return validations.SerializeToString()


class SerializeValidations(beam.PTransform):  # pylint: disable=invalid-name
  """Converts validations to serialized protos."""

  def expand(self, validations: beam.pvalue.PCollection):
    """Converts the given validations into serialized proto.

    Args:
      validations: PCollection of ValidationResults.

    Returns:
      PCollection of serialized proto ValidationResults.
    """
    validations = validations | 'SerializeValidations' >> beam.Map(
        _SerializeValidations)
    return validations


@beam.typehints.with_input_types(validation_result_pb2.ValidationResult)
@beam.typehints.with_output_types(validation_result_pb2.ValidationResult)
class _CombineValidations(beam.CombineFn):
  """Combines the ValidationResults protos.

  Combines PCollection of ValidationResults for different metrics and slices.
  """

  def create_accumulator(self) -> None:
    return

  def add_input(
      self, result: 'Optional[validation_result_pb2.ValidationResult]',
      new_input: 'Optional[validation_result_pb2.ValidationResult]'
  ) -> 'Optional[validation_result_pb2.ValidationResult]':
    if new_input is None:
      return None
    if result is None:
      result = validation_result_pb2.ValidationResult(validation_ok=True)
    result.validation_ok &= new_input.validation_ok
    result.metric_validations_per_slice.extend(
        new_input.metric_validations_per_slice)
    return result

  def merge_accumulators(
      self,
      accumulators: 'List[Optional[validation_result_pb2.ValidationResult]]'
  ) -> 'Optional[validation_result_pb2.ValidationResult]':
    accumulators = [accumulator for accumulator in accumulators if accumulator]
    if not accumulators:
      return None
    result = validation_result_pb2.ValidationResult(validation_ok=True)
    for new_input in accumulators:
      result.metric_validations_per_slice.extend(
          new_input.metric_validations_per_slice)
      result.validation_ok &= new_input.validation_ok
    return result

  def extract_output(
      self, accumulator: 'Optional[validation_result_pb2.ValidationResult]'
  ) -> 'Optional[validation_result_pb2.ValidationResult]':
    # Verification fails if there is empty input.
    if not accumulator:
      result = validation_result_pb2.ValidationResult(validation_ok=False)
      return result
    return accumulator


@beam.ptransform_fn
@beam.typehints.with_input_types(evaluator.Evaluation)
@beam.typehints.with_output_types(beam.pvalue.PDone)
def _WriteMetricsPlotsAndValidations(
    evaluation: evaluator.Evaluation, output_paths: Dict[Text, Text],
    add_metrics_callbacks: List[types.AddMetricsCallbackType],
    metrics_key: Text, plots_key: Text, validations_key: Text):
  """PTransform to write metrics and plots."""
  # Skip write if no metrics, plots, or validations are used.
  if (metrics_key not in evaluation and plots_key not in evaluation and
      validations_key not in evaluation):
    return beam.pvalue.PDone(list(evaluation.values())[0].pipeline)

  if metrics_key in evaluation:
    metrics = (
        evaluation[metrics_key] |
        'SerializeMetrics' >> metrics_and_plots_serialization.SerializeMetrics(
            add_metrics_callbacks=add_metrics_callbacks))
    if constants.METRICS_KEY in output_paths:
      # We only use a single shard here because metrics are usually single
      # values so even with 1M slices and a handful of metrics the size
      # requirements will only be a few hundred MB.
      _ = metrics | 'WriteMetrics' >> beam.io.WriteToTFRecord(
          file_path_prefix=output_paths[constants.METRICS_KEY],
          shard_name_template='')

  if plots_key in evaluation:
    plots = (
        evaluation[plots_key]
        | 'SerializePlots' >> metrics_and_plots_serialization.SerializePlots(
            add_metrics_callbacks=add_metrics_callbacks))
    if constants.PLOTS_KEY in output_paths:
      # We only use a single shard here because we are assuming that plots will
      # not be enabled when millions of slices are in use. By default plots are
      # stored with 1K thresholds with each plot entry taking up to 7 fields
      # (tp, fp, ... recall) so if this assumption is false the output can end
      # up in the hundreds of GB.
      _ = plots | 'WritePlots' >> beam.io.WriteToTFRecord(
          file_path_prefix=output_paths[constants.PLOTS_KEY],
          shard_name_template='')

  if validations_key in evaluation:
    validations = (
        evaluation[validations_key]
        |
        'MergeValidationResults' >> beam.CombineGlobally(_CombineValidations())
        | 'SerializeValidationResults' >> SerializeValidations())
    if constants.VALIDATIONS_KEY in output_paths:
      # We only use a single shard here because validations are usually single
      # values.
      _ = validations | 'WriteValidations' >> beam.io.WriteToTFRecord(
          file_path_prefix=output_paths[constants.VALIDATIONS_KEY],
          shard_name_template='')
  return beam.pvalue.PDone(metrics.pipeline)
