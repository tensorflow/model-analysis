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
"""Tests for weighted example count metric."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.metrics import weighted_example_count


class WeightedExampleCountTest(testutil.TensorflowModelAnalysisTest,
                               parameterized.TestCase):

  @parameterized.named_parameters(
      ('basic', '', ''), ('multi-model', 'model', ''),
      ('multi-output', '', 'output'),
      ('multi-model-multi-output', 'model', 'output'))
  def testWeightedExampleCount(self, model_name, output_name):
    metric = weighted_example_count.WeightedExampleCount().computations(
        model_names=[model_name], output_names=[output_name])[0]

    example1 = {'labels': None, 'predictions': None, 'example_weights': [0.5]}
    example2 = {'labels': None, 'predictions': None, 'example_weights': [1.0]}
    example3 = {'labels': None, 'predictions': None, 'example_weights': [0.7]}

    if output_name:
      example1['example_weights'] = {output_name: example1['example_weights']}
      example2['example_weights'] = {output_name: example2['example_weights']}
      example3['example_weights'] = {output_name: example3['example_weights']}

    if model_name:
      example1['example_weights'] = {model_name: example1['example_weights']}
      example2['example_weights'] = {model_name: example2['example_weights']}
      example3['example_weights'] = {model_name: example3['example_weights']}

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(metric.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          weighted_example_count_key = metric_types.MetricKey(
              name='weighted_example_count',
              model_name=model_name,
              output_name=output_name)
          self.assertDictElementsAlmostEqual(
              got_metrics, {weighted_example_count_key: (0.5 + 1.0 + 0.7)})

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.test.main()
