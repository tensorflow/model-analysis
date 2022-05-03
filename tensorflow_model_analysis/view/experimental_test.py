# Copyright 2022 Google LLC
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

import os

from absl.testing import absltest
import pandas as pd
import tensorflow as tf
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.view import experimental

from google.protobuf import text_format


class ExperimentalTest(tf.test.TestCase):

  def testLoadMetricsAsDataframe_DoubleValueOnly(self):
    metrics_for_slice = text_format.Parse(
        """
        slice_key {
           single_slice_keys {
             column: "age"
             float_value: 38.0
           }
           single_slice_keys {
             column: "sex"
             bytes_value: "Female"
           }
         }
         metric_keys_and_values {
           key {
             name: "mean_absolute_error"
             example_weighted {
             }
           }
           value {
             double_value {
               value: 0.1
             }
           }
         }
         metric_keys_and_values {
           key {
             name: "mean_squared_logarithmic_error"
             example_weighted {
             }
           }
           value {
             double_value {
               value: 0.02
             }
           }
         }
         """, metrics_for_slice_pb2.MetricsForSlice())
    path = os.path.join(absltest.get_default_test_tmpdir(), 'metrics.tfrecord')
    with tf.io.TFRecordWriter(path) as writer:
      writer.write(metrics_for_slice.SerializeToString())
    df = experimental.load_metrics_as_dataframe(path)

    expected = pd.DataFrame({
        'slice': [
            'age = 38.0; sex = b\'Female\'', 'age = 38.0; sex = b\'Female\''
        ],
        'name': ['mean_absolute_error', 'mean_squared_logarithmic_error'],
        'model_name': ['', ''],
        'output_name': ['', ''],
        'example_weighted': [False, False],
        'is_diff': [False, False],
        'display_value': [str(0.1), str(0.02)],
        'metric_value': [
            metrics_for_slice_pb2.MetricValue(double_value={'value': 0.1}),
            metrics_for_slice_pb2.MetricValue(double_value={'value': 0.02})
        ],
    })
    pd.testing.assert_frame_equal(expected, df)

    # Include empty column.
    df = experimental.load_metrics_as_dataframe(
        path, include_empty_columns=True)
    expected = pd.DataFrame({
        'slice': [
            'age = 38.0; sex = b\'Female\'', 'age = 38.0; sex = b\'Female\''
        ],
        'name': ['mean_absolute_error', 'mean_squared_logarithmic_error'],
        'model_name': ['', ''],
        'output_name': ['', ''],
        'sub_key': [None, None],
        'aggregation_type': [None, None],
        'example_weighted': [False, False],
        'is_diff': [False, False],
        'display_value': [str(0.1), str(0.02)],
        'metric_value': [
            metrics_for_slice_pb2.MetricValue(double_value={'value': 0.1}),
            metrics_for_slice_pb2.MetricValue(double_value={'value': 0.02})
        ],
        'confidence_interval': [None, None],
    })
    pd.testing.assert_frame_equal(expected, df)


if __name__ == '__main__':
  tf.test.main()
