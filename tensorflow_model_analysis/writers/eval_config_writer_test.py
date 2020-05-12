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
"""Test for using the EvalConfigWriter API."""

import os
import pickle

from typing import Dict, List, NamedTuple, Optional, Text, Union

import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import version as tfma_version
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from tensorflow_model_analysis.writers import eval_config_writer

LegacyConfig = NamedTuple(
    'LegacyConfig',
    [('model_location', Text), ('data_location', Text),
     ('slice_spec', Optional[List[slicer.SingleSliceSpec]]),
     ('example_count_metric_key', Text),
     ('example_weight_metric_key', Union[Text, Dict[Text, Text]]),
     ('compute_confidence_intervals', bool), ('k_anonymization_count', int)])


class EvalConfigWriterTest(testutil.TensorflowModelAnalysisTest):

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
    options = config.Options()
    options.compute_confidence_intervals.value = (
        old_config.compute_confidence_intervals)
    options.min_slice_size.value = old_config.k_anonymization_count
    eval_config = config.EvalConfig(
        slicing_specs=[
            config.SlicingSpec(
                feature_keys=['country'],
                feature_values={
                    'age': '5',
                    'gender': 'f'
                }),
            config.SlicingSpec(
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
    options = config.Options()
    options.compute_confidence_intervals.value = False
    options.min_slice_size.value = 1
    eval_config = config.EvalConfig(
        slicing_specs=[
            config.SlicingSpec(
                feature_keys=['country'],
                feature_values={
                    'age': '5',
                    'gender': 'f'
                }),
            config.SlicingSpec(
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


if __name__ == '__main__':
  tf.test.main()
