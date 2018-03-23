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
"""Simple tests for testutil."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow.core.example import example_pb2


class TestUtilTest(testutil.TensorflowModelAnalysisTest):

  def testMakeExample(self):
    expected = example_pb2.Example()
    expected.features.feature['single_float'].float_list.value[:] = [1.0]
    expected.features.feature['single_int'].float_list.value[:] = [2]
    expected.features.feature['single_str'].bytes_list.value[:] = ['apple']
    expected.features.feature['multi_float'].float_list.value[:] = [4.0, 5.0]
    expected.features.feature['multi_int'].float_list.value[:] = [6, 7]
    expected.features.feature['multi_str'].bytes_list.value[:] = [
        'orange', 'banana'
    ]
    self.assertEqual(expected,
                     self._makeExample(
                         single_float=1.0,
                         single_int=2,
                         single_str='apple',
                         multi_float=[4.0, 5.0],
                         multi_int=[6, 7],
                         multi_str=['orange', 'banana']))


if __name__ == '__main__':
  tf.test.main()
