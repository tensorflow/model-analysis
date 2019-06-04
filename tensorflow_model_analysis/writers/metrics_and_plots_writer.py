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
"""Metrics and plots writer."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import apache_beam as beam

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.writers import metrics_and_plots_serialization
from tensorflow_model_analysis.writers import writer

from typing import Dict, Text


def MetricsAndPlotsWriter(eval_shared_model: types.EvalSharedModel,
                          output_paths: Dict[Text, Text]) -> writer.Writer:
  return writer.Writer(
      stage_name='WriteMetricsAndPlots',
      ptransform=_WriteMetricsAndPlots(  # pylint: disable=no-value-for-parameter
          eval_shared_model=eval_shared_model,
          output_paths=output_paths))


@beam.ptransform_fn
@beam.typehints.with_input_types(evaluator.Evaluation)
@beam.typehints.with_output_types(beam.pvalue.PDone)
def _WriteMetricsAndPlots(evaluation: evaluator.Evaluation,
                          eval_shared_model: types.EvalSharedModel,
                          output_paths: Dict[Text, Text]):
  """PTransform to write metrics and plots."""

  metrics = evaluation[constants.METRICS_KEY]
  plots = evaluation[constants.PLOTS_KEY]

  metrics, plots = (
      (metrics, plots)
      | 'SerializeMetricsAndPlots' >>
      metrics_and_plots_serialization.SerializeMetricsAndPlots(
          post_export_metrics=eval_shared_model.add_metrics_callbacks))

  if constants.METRICS_KEY in output_paths:
    _ = metrics | 'WriteMetrics' >> beam.io.WriteToTFRecord(
        file_path_prefix=output_paths[constants.METRICS_KEY],
        shard_name_template='')

  if constants.PLOTS_KEY in output_paths:
    _ = plots | 'WritePlots' >> beam.io.WriteToTFRecord(
        file_path_prefix=output_paths[constants.PLOTS_KEY],
        shard_name_template='')

  return beam.pvalue.PDone(metrics.pipeline)
