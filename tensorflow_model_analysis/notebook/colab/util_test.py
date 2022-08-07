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
"""Tests for util."""
import tensorflow as tf
from tensorflow_model_analysis.notebook.colab import util


class UtilTest(tf.test.TestCase):

  def testGenerateSimpleHtmlForTfmaComponent(self):
    tfma_component_name = 'tfma-nb-slicing-metrics'
    html_code = util.generate_html_for_tfma_component(tfma_component_name, None,
                                                      None, '')
    # Count opening and closing tag
    self.assertEqual(html_code.count(tfma_component_name), 2)

  def testGenerateForNotTrustedTfmaComponent(self):
    tfma_component_name = 'my-component'
    with self.assertRaises(ValueError):
      util.generate_html_for_tfma_component(tfma_component_name, None, None, '')


if __name__ == '__main__':
  tf.test.main()
