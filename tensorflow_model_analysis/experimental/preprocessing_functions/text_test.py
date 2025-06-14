# Copyright 2021 Google LLC
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
"""Tests for text_util."""
from absl.testing import parameterized
import tensorflow as tf
from tensorflow_model_analysis.experimental.preprocessing_functions import text


class TextTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('EmptyString', [''], [[]]),
      ('SingleString', ['Test foo Bar'], [['test', 'foo', 'bar']]),
      (
          'BatchedString',
          ['app dog', 'test foo bar'],
          [['app', 'dog', ''], ['test', 'foo', 'bar']],
      ),
  )
  def testWhitespaceTokenization(self, input_text, expected_output):
    # TODO(b/194508683) Delete the check when TF1 is deprecated.
    if tf.__version__ < '2':
      return

    actual = text.whitespace_tokenization(input_text).to_tensor()
    expected = tf.constant(expected_output, dtype=tf.string)
    self.assertAllEqual(actual, expected)


