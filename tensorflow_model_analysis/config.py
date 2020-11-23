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

from typing import Optional

from absl import logging
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.proto import config_pb2

# Define types here to avoid type errors between OSS and internal code.
ModelSpec = config_pb2.ModelSpec
SlicingSpec = config_pb2.SlicingSpec
CrossSlicingSpec = config_pb2.CrossSlicingSpec
BinarizationOptions = config_pb2.BinarizationOptions
ConfidenceIntervalOptions = config_pb2.ConfidenceIntervalOptions
AggregationOptions = config_pb2.AggregationOptions
MetricConfig = config_pb2.MetricConfig
MetricsSpec = config_pb2.MetricsSpec
MetricDirection = config_pb2.MetricDirection
GenericChangeThreshold = config_pb2.GenericChangeThreshold
GenericValueThreshold = config_pb2.GenericValueThreshold
MetricThreshold = config_pb2.MetricThreshold
Options = config_pb2.Options
PerSliceMetricThreshold = config_pb2.PerSliceMetricThreshold
PerSliceMetricThresholds = config_pb2.PerSliceMetricThresholds
CrossSliceMetricThreshold = config_pb2.CrossSliceMetricThreshold
CrossSliceMetricThresholds = config_pb2.CrossSliceMetricThresholds
EvalConfig = config_pb2.EvalConfig


def verify_eval_config(eval_config: EvalConfig,
                       baseline_required: Optional[bool] = None):
  """Verifies eval config."""
  if not eval_config.model_specs:
    raise ValueError(
        'At least one model_spec is required: eval_config=\n{}'.format(
            eval_config))

  model_specs_by_name = {}
  baseline = None
  for spec in eval_config.model_specs:
    if spec.label_key and spec.label_keys:
      raise ValueError('only one of label_key or label_keys should be used at '
                       'a time: model_spec=\n{}'.format(spec))
    if spec.prediction_key and spec.prediction_keys:
      raise ValueError(
          'only one of prediction_key or prediction_keys should be used at '
          'a time: model_spec=\n{}'.format(spec))
    if spec.example_weight_key and spec.example_weight_keys:
      raise ValueError(
          'only one of example_weight_key or example_weight_keys should be '
          'used at a time: model_spec=\n{}'.format(spec))
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

  if len(model_specs_by_name) > 1 and '' in model_specs_by_name:
    raise ValueError('A name is required for all ModelSpecs when multiple '
                     'models are used: eval_config=\n{}'.format(eval_config))

  if baseline_required and not baseline:
    raise ValueError(
        'A baseline ModelSpec is required: eval_config=\n{}'.format(
            eval_config))


def update_eval_config_with_defaults(
    eval_config: EvalConfig,
    maybe_add_baseline: Optional[bool] = None,
    maybe_remove_baseline: Optional[bool] = None,
    has_baseline: Optional[bool] = False,
    rubber_stamp: Optional[bool] = False) -> EvalConfig:
  """Returns a new config with default settings applied.

  a) Add or remove a model_spec according to "has_baseline".
  b) Fix the model names (model_spec.name) to tfma.CANDIDATE_KEY and
     tfma.BASELINE_KEY.
  c) Update the metrics_specs with the fixed model name.

  Args:
    eval_config: Original eval config.
    maybe_add_baseline: DEPRECATED. True to add a baseline ModelSpec to the
      config as a copy of the candidate ModelSpec that should already be
      present. This is only applied if a single ModelSpec already exists in the
      config and that spec doesn't have a name associated with it. When applied
      the model specs will use the names tfma.CANDIDATE_KEY and
      tfma.BASELINE_KEY. Only one of maybe_add_baseline or
      maybe_remove_baseline should be used.
    maybe_remove_baseline: DEPRECATED. True to remove a baseline ModelSpec from
      the config if it already exists. Removal of the baseline also removes any
      change thresholds. Only one of maybe_add_baseline or maybe_remove_baseline
      should be used.
    has_baseline: True to add a baseline ModelSpec to the config as a copy of
      the candidate ModelSpec that should already be present. This is only
      applied if a single ModelSpec already exists in the config and that spec
      doesn't have a name associated with it. When applied the model specs will
      use the names tfma.CANDIDATE_KEY and tfma.BASELINE_KEY. False to remove a
      baseline ModelSpec from the config if it already exists. Removal of the
      baseline also removes any change thresholds. Only one of has_baseline or
      maybe_remove_baseline should be used.
    rubber_stamp: True if this model is being rubber stamped. When a model is
      rubber stamped diff thresholds will be ignored if an associated baseline
      model is not passed.
  """
  if (not has_baseline and has_change_threshold(eval_config) and
      not rubber_stamp):
    # TODO(b/173657964): Raise an error instead of logging an error.
    logging.error('There are change thresholds, but the baseline is missing. '
                  'This is allowed only when rubber stamping (first run).')

  updated_config = EvalConfig()
  updated_config.CopyFrom(eval_config)
  # if user requests CIs but doesn't set method, use JACKKNIFE
  if (eval_config.options.compute_confidence_intervals.value and
      eval_config.options.confidence_intervals.method ==
      ConfidenceIntervalOptions.UNKNOWN_CONFIDENCE_INTERVAL_METHOD):
    updated_config.options.confidence_intervals.method = (
        config_pb2.ConfidenceIntervalOptions.JACKKNIFE)
  if maybe_add_baseline and maybe_remove_baseline:
    raise ValueError('only one of maybe_add_baseline and maybe_remove_baseline '
                     'should be used')
  if maybe_add_baseline or maybe_remove_baseline:
    logging.warning(
        """"maybe_add_baseline" and "maybe_remove_baseline" are deprecated,
        please use "has_baseline" instead.""")
    if has_baseline:
      raise ValueError(
          """"maybe_add_baseline" and "maybe_remove_baseline" are ignored if
          "has_baseline" is set.""")
  if has_baseline is not None:
    if has_baseline:
      maybe_add_baseline = True
    else:
      maybe_remove_baseline = True

  # Has a baseline model.
  if (maybe_add_baseline and len(updated_config.model_specs) == 1 and
      not updated_config.model_specs[0].name):
    baseline = updated_config.model_specs.add()
    baseline.CopyFrom(updated_config.model_specs[0])
    baseline.name = constants.BASELINE_KEY
    baseline.is_baseline = True
    updated_config.model_specs[0].name = constants.CANDIDATE_KEY
    logging.info(
        'Adding default baseline ModelSpec based on the candidate ModelSpec '
        'provided. The candidate model will be called "%s" and the baseline '
        'will be called "%s": updated_config=\n%s', constants.CANDIDATE_KEY,
        constants.BASELINE_KEY, updated_config)

  # Does not have a baseline.
  if maybe_remove_baseline:
    tmp_model_specs = []
    for model_spec in updated_config.model_specs:
      if not model_spec.is_baseline:
        tmp_model_specs.append(model_spec)
    del updated_config.model_specs[:]
    updated_config.model_specs.extend(tmp_model_specs)
    for metrics_spec in updated_config.metrics_specs:
      for metric in metrics_spec.metrics:
        if metric.threshold.ByteSize():
          metric.threshold.ClearField('change_threshold')
        for per_slice_threshold in metric.per_slice_thresholds:
          if per_slice_threshold.threshold.ByteSize():
            per_slice_threshold.threshold.ClearField('change_threshold')
        for cross_slice_threshold in metric.cross_slice_thresholds:
          if cross_slice_threshold.threshold.ByteSize():
            cross_slice_threshold.threshold.ClearField('change_threshold')
      for threshold in metrics_spec.thresholds.values():
        if threshold.ByteSize():
          threshold.ClearField('change_threshold')
      for per_slice_thresholds in metrics_spec.per_slice_thresholds.values():
        for per_slice_threshold in per_slice_thresholds.thresholds:
          if per_slice_threshold.threshold.ByteSize():
            per_slice_threshold.threshold.ClearField('change_threshold')
      for cross_slice_thresholds in metrics_spec.cross_slice_thresholds.values(
      ):
        for cross_slice_threshold in cross_slice_thresholds.thresholds:
          if cross_slice_threshold.threshold.ByteSize():
            cross_slice_threshold.threshold.ClearField('change_threshold')
    logging.info(
        'Request was made to ignore the baseline ModelSpec and any change '
        'thresholds. This is likely because a baseline model was not provided: '
        'updated_config=\n%s', updated_config)

  if not updated_config.model_specs:
    updated_config.model_specs.add()

  baseline_spec = None
  model_names = []
  for spec in updated_config.model_specs:
    if spec.is_baseline:
      baseline_spec = spec
    model_names.append(spec.name)
  if len(model_names) == 1 and model_names[0]:
    logging.info(
        'ModelSpec name "%s" is being ignored and replaced by "" because a '
        'single ModelSpec is being used', model_names[0])
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


def has_change_threshold(eval_config: EvalConfig) -> bool:
  """Checks whether the eval_config has any change thresholds.

  Args:
    eval_config: the TFMA eval_config.

  Returns:
    True when there are change thresholds otherwise False.
  """

  for metrics_spec in eval_config.metrics_specs:
    for metric in metrics_spec.metrics:
      if metric.threshold.change_threshold.ByteSize():
        return True
      for per_slice_threshold in metric.per_slice_thresholds:
        if per_slice_threshold.threshold.change_threshold.ByteSize():
          return True
      for cross_slice_threshold in metric.cross_slice_thresholds:
        if cross_slice_threshold.threshold.change_threshold.ByteSize():
          return True
    for threshold in metrics_spec.thresholds.values():
      if threshold.change_threshold.ByteSize():
        return True
    for per_slice_thresholds in metrics_spec.per_slice_thresholds.values():
      for per_slice_threshold in per_slice_thresholds.thresholds:
        if per_slice_threshold.threshold.change_threshold.ByteSize():
          return True
    for cross_slice_thresholds in metrics_spec.cross_slice_thresholds.values():
      for cross_slice_threshold in cross_slice_thresholds.thresholds:
        if cross_slice_threshold.threshold.change_threshold.ByteSize():
          return True
  return False
