# Copyright 2023 Google LLC
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
"""Tests for cross entropy related metrics."""
from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
from tensorflow_model_analysis.metrics import cross_entropy_metrics
from tensorflow_model_analysis.metrics import metric_util


class CrossEntropyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # To be consistent with Keras, a single example can have multiple
      # predictions and labels.
      dict(
          testcase_name='_binary_two_examples',
          extracts=[
              {
                  'labels': np.array([0]),
                  'predictions': np.array([0.6]),
              },
              {
                  'labels': np.array([1]),
                  'predictions': np.array([0.6]),
                  'example_weights': np.array([0.8]),
              },
          ],
          metric=cross_entropy_metrics.BinaryCrossEntropy(
              from_logits=False, label_smoothing=0.0
          ),
          expected_value=0.736083,
      ),
      dict(
          testcase_name='_binary_two_examples_per_batch_from_logits',
          extracts=[
              {
                  'labels': np.array([0, 1]),
                  'predictions': np.array([-18.6, 0.51]),
              },
              {
                  'labels': np.array([0, 0]),
                  'predictions': np.array([2.94, -12.8]),
              },
          ],
          metric=cross_entropy_metrics.BinaryCrossEntropy(
              from_logits=True, label_smoothing=0.0
          ),
          expected_value=0.865457,
      ),
      dict(
          testcase_name='_binary_two_examples_with_label_smoothing',
          extracts=[
              {
                  'labels': np.array([0]),
                  'predictions': np.array([0.6]),
              },
              {
                  'labels': np.array([1]),
                  'predictions': np.array([0.6]),
                  'example_weights': np.array([0.8]),
              },
          ],
          metric=cross_entropy_metrics.BinaryCrossEntropy(
              from_logits=False, label_smoothing=0.1
          ),
          expected_value=0.733831,
      ),
      dict(
          testcase_name='_categorical_two_examples',
          extracts=[
              {
                  'labels': np.array([0, 1, 0]),
                  'predictions': np.array([0.05, 0.95, 0]),
              },
              {
                  'labels': np.array([0, 0, 1]),
                  'predictions': np.array([0.1, 0.8, 0.1]),
              },
          ],
          metric=cross_entropy_metrics.CategoricalCrossEntropy(
              from_logits=False, label_smoothing=0.0
          ),
          expected_value=1.176939,
      ),
      dict(
          testcase_name='_categorical_two_examples_with_weights',
          extracts=[
              {
                  'labels': np.array([0, 1, 0]),
                  'predictions': np.array([0.05, 0.95, 0]),
                  'example_weights': np.array([0.3]),
              },
              {
                  'labels': np.array([0, 0, 1]),
                  'predictions': np.array([0.1, 0.8, 0.1]),
                  'example_weights': np.array([0.7]),
              },
          ],
          metric=cross_entropy_metrics.CategoricalCrossEntropy(
              from_logits=False, label_smoothing=0.0
          ),
          expected_value=1.627198,
      ),
      dict(
          testcase_name='_categorical_two_examples_from_logits',
          extracts=[
              {
                  'labels': np.array([0, 1, 0]),
                  'predictions': np.array([-5, 0.95, -10]),
              },
              {
                  'labels': np.array([0, 0, 1]),
                  'predictions': np.array([5, -1, 0.5]),
              },
          ],
          metric=cross_entropy_metrics.CategoricalCrossEntropy(
              from_logits=True, label_smoothing=0.0
          ),
          expected_value=2.258058,
      ),
      dict(
          testcase_name='_categorical_two_examples_from_logits_with_smoothing',
          extracts=[
              {
                  'labels': np.array([0, 1, 0]),
                  'predictions': np.array([-5, 0.95, -10]),
              },
              {
                  'labels': np.array([0, 0, 1]),
                  'predictions': np.array([5, -1, 0.5]),
              },
          ],
          metric=cross_entropy_metrics.CategoricalCrossEntropy(
              from_logits=True, label_smoothing=0.1
          ),
          expected_value=2.489725,
      ),
  )
  def testBinaryCrossEntropy(self, extracts, metric, expected_value):
    computations = metric.computations(example_weighted=True)
    computation = computations[0]
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(extracts)
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeMetric' >> beam.CombinePerKey(computation.combiner)
      )

      # pylint: enable=no-value-for-parameter
      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          key = computation.keys[0]
          self.assertIn(key, got_metrics)
          self.assertAlmostEqual(got_metrics[key], expected_value, places=5)
        except AssertionError as err:
          raise util.BeamAssertException() from err

      util.assert_that(result, check_result, label='result')


