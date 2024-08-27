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
"""Tests for mean_regression_error related metrics."""


import pytest
from typing import Iterator
from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.metrics import mean_regression_error
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util


class IdTransformPreprocessor(metric_types.Preprocessor):
  """ID transform preprocessor."""

  def __init__(
      self,
  ):
    super().__init__(
        name=metric_util.generate_private_name_from_arguments(
            'test_id_transform_preprocessor',
        )
    )

  def process(
      self, extracts: types.Extracts
  ) -> Iterator[metric_types.StandardMetricInputs]:
    yield extracts


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class MeanRegressionErrorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # example1 is |0.1 - 1| * 0.1 + |0.3 - 0| * 0.5 + |0.5 - 2| * 1
      # = 0.09 + 0.15 + 1.5 = 1.74
      # example2 is |1 - 0.5| + |2 - 1| + |3 - 5| = 3.5
      # example3 is |3 - 5| = 2
      # average error: (1.74 + 3.5 + 2) / 5.6 = 1.292857
      (
          '_mean_absolute_error',
          mean_regression_error.MeanAbsoluteError(),
          1.292857,
      ),
      # example1 is |0.1 - 1|^2 * 0.1 + |0.3 - 0|^2 * 0.5 + |0.5 - 2|^2 * 1
      # = 0.081 + 0.045 + 2.25 = 2.376
      # example2 is |1 - 0.5|^2 + |2 - 1|^2 + |3 - 5|^2 = 5.25
      # example3 is |3 - 5|^2 = 4
      # average error: (2.376 + 5.25 + 4) / 5.6 = 2.07607
      (
          '_mean_squared_error',
          mean_regression_error.MeanSquaredError(),
          2.07607,
      ),
      # example1 is 100 * (|0.1 - 1| / 0.1 * 0.1 + |0.3 - 0| / 0.3 * 0.5 +
      # |0.5 - 2| / 0.5 * 1) = 440
      # example2 is 100 * (|1 - 0.5| / 1 + |2 - 1| / 2 + |3 - 5| / 3) = 166.66
      # example3 is 100 * (|3 - 5| / 3) = 66.66
      # average error: (440 + 166.66 + 66.66) / 5.6 = 120.238095
      (
          '_mean_absolute_percentage_error',
          mean_regression_error.MeanAbsolutePercentageError(),
          120.238095,
      ),
      # example1 is |log(0.1+1) - log(1+1)|^2 * 0.1 +
      # |log(0.3+1) - log(0+1)|^2 * 0.5 + |log(0.5+1) - log(2+1)|^2 * 1 =0.55061
      # example2 is |log(1+1) - log(0.5+1)|^2 + |log(2+1) - log(1+1)|^2
      # + |log(3+1) - log(5+1)|^2 = 0.41156
      # example3 is |log(3+1) - log(5+1)|^2 = 0.16440
      # average error: (0.55061 + 0.41156 + 0.16440) / 5.6 = 0.20117
      (
          '_mean_squared_logarithmic_error',
          mean_regression_error.MeanSquaredLogarithmicError(),
          0.20117,
      ),
  )
  def testRegressionErrorWithWeights(self, metric, expected_value):
    computations = metric.computations(example_weighted=True)
    computation = computations[0]
    example1 = {
        'labels': np.array([0.1, 0.3, 0.5]),
        'predictions': np.array([1, 0, 2]),
        'example_weights': np.array([0.1, 0.5, 1]),
    }
    example2 = {
        'labels': np.array([1, 2, 3]),
        'predictions': np.array([0.5, 1.0, 5]),
        'example_weights': np.array([1.0]),
    }
    example3 = {
        'labels': np.array([3]),
        'predictions': np.array([5]),
        'example_weights': np.array([1.0]),
    }
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3])
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
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      # example1 is |0.1 - 1| * 0.1 + |0.3 - 0| * 0.5 + |0.5 - 2| * 1
      # = 0.09 + 0.15 + 1.5 = 1.74
      # example2 is |1 - 0.5| + |2 - 1| + |3 - 5| = 3.5
      # example3 is |3 - 5| = 2
      # average error: (1.74 + 3.5 + 2) / 5.6 = 1.292857
      (
          '_mean_absolute_error_with_preprocessors',
          mean_regression_error.MeanAbsoluteError(
              preprocessors=[IdTransformPreprocessor()]
          ),
          1.292857,
      ),
      # example1 is |0.1 - 1|^2 * 0.1 + |0.3 - 0|^2 * 0.5 + |0.5 - 2|^2 * 1
      # = 0.081 + 0.045 + 2.25 = 2.376
      # example2 is |1 - 0.5|^2 + |2 - 1|^2 + |3 - 5|^2 = 5.25
      # example3 is |3 - 5|^2 = 4
      # average error: (2.376 + 5.25 + 4) / 5.6 = 2.07607
      (
          '_mean_squared_error_with_preprocessors',
          mean_regression_error.MeanSquaredError(
              preprocessors=[IdTransformPreprocessor()]
          ),
          2.07607,
      ),
      # example1 is 100 * (|0.1 - 1| / 0.1 * 0.1 + |0.3 - 0| / 0.3 * 0.5 +
      # |0.5 - 2| / 0.5 * 1) = 440
      # example2 is 100 * (|1 - 0.5| / 1 + |2 - 1| / 2 + |3 - 5| / 3) = 166.66
      # example3 is 100 * (|3 - 5| / 3) = 66.66
      # average error: (440 + 166.66 + 66.66) / 5.6 = 120.238095
      (
          '_mean_absolute_percentage_error_with_preprocessors',
          mean_regression_error.MeanAbsolutePercentageError(
              preprocessors=[IdTransformPreprocessor()]
          ),
          120.238095,
      ),
      # example1 is |log(0.1+1) - log(1+1)|^2 * 0.1 +
      # |log(0.3+1) - log(0+1)|^2 * 0.5 + |log(0.5+1) - log(2+1)|^2 * 1 =0.55061
      # example2 is |log(1+1) - log(0.5+1)|^2 + |log(2+1) - log(1+1)|^2
      # + |log(3+1) - log(5+1)|^2 = 0.41156
      # example3 is |log(3+1) - log(5+1)|^2 = 0.16440
      # average error: (0.55061 + 0.41156 + 0.16440) / 5.6 = 0.20117
      (
          '_mean_squared_logarithmic_error_with_preprocessors',
          mean_regression_error.MeanSquaredLogarithmicError(
              preprocessors=[IdTransformPreprocessor()]
          ),
          0.20117,
      ),
  )
  def testRegressionErrorWithWeightsWithPreprocessors(
      self, metric, expected_value
  ):
    computations = metric.computations(example_weighted=True)
    computation = computations[0]
    example1 = {
        'labels': np.array([0.1, 0.3, 0.5]),
        'predictions': np.array([1, 0, 2]),
        'example_weights': np.array([0.1, 0.5, 1]),
    }
    example2 = {
        'labels': np.array([1, 2, 3]),
        'predictions': np.array([0.5, 1.0, 5]),
        'example_weights': np.array([1.0]),
    }
    example3 = {
        'labels': np.array([3]),
        'predictions': np.array([5]),
        'example_weights': np.array([1.0]),
    }
    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3])
          | 'PreProcess' >> beam.ParDo(computation.preprocessors[0])
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
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


