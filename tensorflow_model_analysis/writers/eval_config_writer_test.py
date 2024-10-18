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
"""Test for using the EvalConfigWriter API."""

import os
import pickle
from typing import Dict, List, NamedTuple, Optional, Union

import tensorflow as tf
from tensorflow_model_analysis import version as tfma_version
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from tensorflow_model_analysis.utils import test_util
from tensorflow_model_analysis.writers import eval_config_writer


LegacyConfig = NamedTuple(
    'LegacyConfig', [('model_location', str), ('data_location', str),
                     ('slice_spec', Optional[List[slicer.SingleSliceSpec]]),
                     ('example_count_metric_key', str),
                     ('example_weight_metric_key', Union[str, Dict[str, str]]),
                     ('compute_confidence_intervals', bool),
                     ('k_anonymization_count', int)])


class EvalConfigWriterTest(test_util.TensorflowModelAnalysisTest):

  def testSerializeDeserializeLegacyEvalConfig(self):
    output_path = self._getTempDir()
    old_config = LegacyConfig(
        model_location='/path/to/model',
        data_location='/path/to/data',
        slice_spec=[
            slicer.SingleSliceSpec(
                columns=['country'], features=[('age', 5), ('gender', 'f')]),
            slicer.SingleSliceSpec(
                columns=['interest'], features=[('age', 6), ('gender', 'm')])
        ],
        example_count_metric_key=None,
        example_weight_metric_key='key',
        compute_confidence_intervals=False,
        k_anonymization_count=1)
    final_dict = {}
    final_dict['tfma_version'] = tfma_version.VERSION
    final_dict['eval_config'] = old_config
    with tf.io.TFRecordWriter(os.path.join(output_path, 'eval_config')) as w:
      w.write(pickle.dumps(final_dict))
    got_eval_config, got_data_location, _, got_model_locations = (
        eval_config_writer.load_eval_run(output_path))
    options = config_pb2.Options()
    options.compute_confidence_intervals.value = (
        old_config.compute_confidence_intervals)
    options.min_slice_size.value = old_config.k_anonymization_count
    eval_config = config_pb2.EvalConfig(
        slicing_specs=[
            config_pb2.SlicingSpec(
                feature_keys=['country'],
                feature_values={
                    'age': '5',
                    'gender': 'f'
                }),
            config_pb2.SlicingSpec(
                feature_keys=['interest'],
                feature_values={
                    'age': '6',
                    'gender': 'm'
                })
        ],
        options=options)
    self.assertEqual(eval_config, got_eval_config)
    self.assertEqual(old_config.data_location, got_data_location)
    self.assertLen(got_model_locations, 1)
    self.assertEqual(old_config.model_location,
                     list(got_model_locations.values())[0])

  def testSerializeDeserializeEvalConfig(self):
    output_path = self._getTempDir()
    options = config_pb2.Options()
    options.compute_confidence_intervals.value = False
    options.min_slice_size.value = 1
    eval_config = config_pb2.EvalConfig(
        slicing_specs=[
            config_pb2.SlicingSpec(
                feature_keys=['country'],
                feature_values={
                    'age': '5',
                    'gender': 'f'
                }),
            config_pb2.SlicingSpec(
                feature_keys=['interest'],
                feature_values={
                    'age': '6',
                    'gender': 'm'
                })
        ],
        options=options)
    data_location = '/path/to/data'
    file_format = 'tfrecords'
    model_location = '/path/to/model'
    with tf.io.gfile.GFile(os.path.join(output_path, 'eval_config.json'),
                           'w') as f:
      f.write(
          eval_config_writer._serialize_eval_run(eval_config, data_location,
                                                 file_format,
                                                 {'': model_location}))
    got_eval_config, got_data_location, got_file_format, got_model_locations = (
        eval_config_writer.load_eval_run(output_path))
    self.assertEqual(eval_config, got_eval_config)
    self.assertEqual(data_location, got_data_location)
    self.assertEqual(file_format, got_file_format)
    self.assertEqual({'': model_location}, got_model_locations)


