# Lint as: python3
# Copyright 2020 Google LLC
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
"""EvalConfig writer."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import os
import pickle

from typing import Any, Dict, Optional, Text, Tuple

import apache_beam as beam
import six
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import version as tfma_version
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.writers import writer

from google.protobuf import json_format

EVAL_CONFIG_FILE = 'eval_config'
EVAL_CONFIG_FILE_FORMAT = 'json'


def _check_version(version: Text, path: Text):
  if not version:
    raise ValueError(
        'could not find TFMA version in raw deserialized dictionary for '
        'file at %s' % path)
  # We don't actually do any checking for now, since we don't have any
  # compatibility issues.


def _serialize_eval_run(eval_config: config.EvalConfig, data_location: Text,
                        file_format: Text, model_locations: Dict[Text,
                                                                 Text]) -> Text:
  return json_format.MessageToJson(
      config_pb2.EvalRun(
          eval_config=eval_config,
          version=tfma_version.VERSION,
          data_location=data_location,
          file_format=file_format,
          model_locations=model_locations))


def load_eval_run(
    output_path: Text,
    output_file_format: Text = EVAL_CONFIG_FILE_FORMAT,
    filename: Optional[Text] = None
) -> Tuple[Optional[config.EvalConfig], Text, Text, Dict[Text, Text]]:
  """Returns eval config, data location, file format, and model locations.

  Args:
    output_path: Directory containing config file.
    output_file_format: Format of output file. Currently only 'json' is
      supported.
    filename: Name of output file (including extension if any).

  Returns:
    Tuple of (EvalConfig, data location, file format, model locations). If an
    EvalConfig is not found at the given path, None will be returned.
  """
  if filename is None:
    filename = EVAL_CONFIG_FILE + '.' + output_file_format
  path = os.path.join(output_path, filename)
  if tf.io.gfile.exists(path):
    with tf.io.gfile.GFile(path, 'r') as f:
      pb = json_format.Parse(f.read(), config_pb2.EvalRun())
      _check_version(pb.version, output_path)
      return (pb.eval_config, pb.data_location, pb.file_format,
              pb.model_locations)

  # Legacy suppport (to be removed in future).
  # The previous version did not include file extension.
  path = os.path.splitext(path)[0]
  if tf.io.gfile.exists(path):
    serialized_record = six.next(
        tf.compat.v1.python_io.tf_record_iterator(path))
    final_dict = pickle.loads(serialized_record)
    _check_version(final_dict, output_path)
    old_config = final_dict['eval_config']
    slicing_specs = None
    if old_config.slice_spec:
      slicing_specs = [s.to_proto() for s in old_config.slice_spec]
    options = config.Options()
    options.compute_confidence_intervals.value = (
        old_config.compute_confidence_intervals)
    options.min_slice_size.value = old_config.k_anonymization_count
    return (config.EvalConfig(slicing_specs=slicing_specs,
                              options=options), old_config.data_location, '', {
                                  '': old_config.model_location
                              })

  # No config found
  return (None, '', '', {})


def EvalConfigWriter(  # pylint: disable=invalid-name
    output_path: Text,
    eval_config: config.EvalConfig,
    output_file_format: Text = EVAL_CONFIG_FILE_FORMAT,
    data_location: Optional[Text] = None,
    data_file_format: Optional[Text] = None,
    model_locations: Optional[Dict[Text, Text]] = None,
    filename: Optional[Text] = None) -> writer.Writer:
  """Returns eval config writer.

  Args:
    output_path: Output path to write config to.
    eval_config: EvalConfig.
    output_file_format: Output file format. Currently on 'json' is supported.
    data_location: Optional path indicating where data is read from. This is
      only used for display purposes.
    data_file_format: Optional format of the input examples. This is only used
      for display purposes.
    model_locations: Dict of model locations keyed by model name. This is only
      used for display purposes.
    filename: Name of file to store the config as.
  """
  if data_location is None:
    data_location = '<user provided PCollection>'
  if data_file_format is None:
    data_file_format = '<unknown>'
  if model_locations is None:
    model_locations = {'': '<unknown>'}
  if filename is None:
    filename = EVAL_CONFIG_FILE + '.' + output_file_format

  return writer.Writer(
      stage_name='WriteEvalConfig',
      ptransform=_WriteEvalConfig(  # pylint: disable=no-value-for-parameter
          eval_config=eval_config,
          output_path=output_path,
          output_file_format=output_file_format,
          data_location=data_location,
          data_file_format=data_file_format,
          model_locations=model_locations,
          filename=filename))


@beam.ptransform_fn
@beam.typehints.with_input_types(Any)
@beam.typehints.with_output_types(beam.pvalue.PDone)
def _WriteEvalConfig(  # pylint: disable=invalid-name
    evaluation: evaluator.Evaluation, eval_config: config.EvalConfig,
    output_path: Text, output_file_format: Text, data_location: Text,
    data_file_format: Text, model_locations: Dict[Text, Text],
    filename: Text) -> beam.pvalue.PDone:
  """Writes EvalConfig to file.

  Args:
    evaluation: Evaluation data. This transform only makes use of the pipeline.
    eval_config: EvalConfig.
    output_path: Output path.
    output_file_format: Output file format.
    data_location: Path indicating where input data is read from.
    data_file_format: Format of the input data.
    model_locations: Dict of model locations keyed by model name.
    filename: Name of file to store the config as.

  Returns:
    beam.pvalue.PDone.
  """
  pipeline = list(evaluation.values())[0].pipeline

  # Skip writing file if its output is disabled
  if EVAL_CONFIG_FILE in eval_config.options.disabled_outputs.values:
    return beam.pvalue.PDone(pipeline)

  if output_file_format and output_file_format != EVAL_CONFIG_FILE_FORMAT:
    raise ValueError(
        'only "{}" format is currently supported: output_file_format={}'.format(
            EVAL_CONFIG_FILE_FORMAT, output_file_format))

  return (pipeline
          | 'CreateEvalConfig' >> beam.Create([
              _serialize_eval_run(eval_config, data_location, data_file_format,
                                  model_locations)
          ])
          | 'WriteEvalConfig' >> beam.io.WriteToText(
              os.path.join(output_path, filename), shard_name_template=''))
