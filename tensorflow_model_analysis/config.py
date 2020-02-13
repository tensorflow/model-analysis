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
"""Configuration types."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from absl import logging
from tensorflow_model_analysis.proto import config_pb2

# Define types here to avoid type errors between OSS and internal code.
ModelSpec = config_pb2.ModelSpec
SlicingSpec = config_pb2.SlicingSpec
BinarizationOptions = config_pb2.BinarizationOptions
AggregationOptions = config_pb2.AggregationOptions
MetricConfig = config_pb2.MetricConfig
MetricsSpec = config_pb2.MetricsSpec
MetricDirection = config_pb2.MetricDirection
GenericChangeThreshold = config_pb2.GenericChangeThreshold
GenericValueThreshold = config_pb2.GenericValueThreshold
MetricThreshold = config_pb2.MetricThreshold
Options = config_pb2.Options
EvalConfig = config_pb2.EvalConfig


def verify_eval_config(eval_config: EvalConfig):
  """Verifies eval config."""
  if not eval_config.model_specs:
    raise ValueError(
        'At least one model_spec is required: eval_config={}'.format(
            eval_config))

  model_specs_by_name = {}
  baseline = None
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


def update_eval_config_with_defaults(eval_config: EvalConfig):
  """Returns a new config with default settings applied."""
  updated_config = EvalConfig()
  updated_config.CopyFrom(eval_config)
  if not updated_config.model_specs:
    updated_config.model_specs.add()
  baseline_spec = None
  model_names = []
  for spec in updated_config.model_specs:
    if spec.is_baseline:
      baseline_spec = spec
    model_names.append(spec.name)
  if len(model_names) == 1:
    logging.info(
        'single ModelSpec is being used, name "%s" will be ignored: '
        'config=%s', model_names[0], eval_config)
    updated_config.model_specs[0].name = ''
    model_names = ['']
  for spec in updated_config.metrics_specs:
    if not spec.model_names:
      spec.model_names.extend(model_names)
    elif len(model_names) == 1:
      del spec.model_names[:]
      spec.model_names.append('')
    elif baseline_spec and baseline_spec.name not in spec.model_names:
      spec.model_names.append(baseline_spec.name)
  return updated_config
