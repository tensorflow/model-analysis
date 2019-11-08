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
"""Configuration types."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from tensorflow_model_analysis.proto import config_pb2

# Define types here to avoid type errors between OSS and internal code.
InputDataSpec = config_pb2.InputDataSpec
ModelSpec = config_pb2.ModelSpec
SlicingSpec = config_pb2.SlicingSpec
OutputDataSpec = config_pb2.OutputDataSpec
BinarizationOptions = config_pb2.BinarizationOptions
MetricConfig = config_pb2.MetricConfig
MetricsSpec = config_pb2.MetricsSpec
Options = config_pb2.Options
EvalConfig = config_pb2.EvalConfig


def verify_eval_config(eval_config: EvalConfig):
  """Verifies eval config."""
  model_specs_by_name = {}
  baseline = None
  if eval_config.model_specs:
    for spec in eval_config.model_specs:
      if spec.name in eval_config.model_specs:
        raise ValueError(
            'more than one model_spec found for model "{}": {}'.format(
                spec.name, [spec, model_specs_by_name[spec.name]]))
      model_specs_by_name[spec.name] = spec
      if spec.is_baseline:
        if baseline is not None:
          raise ValueError('only one model_spec may be a baseline, found: '
                           '{} and {}'.format(spec, baseline))
        baseline = spec

  output_data_specs_by_name = {}
  if eval_config.output_data_specs:
    for spec in eval_config.output_data_specs:
      if baseline and baseline.name == spec.model_name:
        raise ValueError(
            'baseline model "{}" should not have an output_data_spec '
            'because baseline outputs are included with the candidates: '
            '{}'.format(baseline.name, eval_config.output_data_specs))
      if spec.model_name not in model_specs_by_name:
        raise ValueError(
            'model_name "{}" for output_data_spec {} unknown'.format(
                spec.model_name, spec))
      if spec.model_name in output_data_specs_by_name:
        raise ValueError(
            'more than one output_data_spec found for model "{}": {}'.format(
                spec.model_name,
                [spec, output_data_specs_by_name[spec.model_name]]))
      output_data_specs_by_name[spec.model_name] = spec
