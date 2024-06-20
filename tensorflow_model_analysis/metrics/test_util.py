# Copyright 2024 Google LLC
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
"""Utilities for testing metrics."""

from typing import Iterable

from absl.testing import absltest
import apache_beam as beam
import numpy as np
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.evaluators import metrics_plots_and_validations_evaluator as evaluator
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.slicer import slicer_lib


class TestCase(absltest.TestCase):
  """Base class for metric tests which provides assertMetricEqual."""

  def assertDerivedMetricsEqual(  # pylint: disable=invalid-name
      self,
      expected_metrics: metric_types.MetricsDict,
      metric: metric_types.Metric,
      extracts: Iterable[types.Extracts],
      example_weighted: bool = True,
      enable_debug_print: bool = False,
  ):
    """Asserts that the given metric has the expected values.

    This method exists to allow metric authors to easily test that their code
    behaves correctly when excercised by the standard evaluator. This utility
    relies heavily on the actual evaluator implementation due to the complexity
    of the metric-evaluator contract. Though this pattern is in conflict with
    the principles of unit testing, we consider this to be preferable to many,
    scattered and incorrect versions of the metric-evaluator contract.

    Schematically, this method:
      - generates the computations from the metric instance
      - filters and separates the different types of computations
      - applies those computations in the same way that the evaluator would
        - non-derived: applies preprocessors and a merged combine_fn which
          possibly includes multiple metric CombineFns
        - derived: applies the derived metric computations to the
          non-derived metric results
      - removes any private metrics from the result
      - asserts that the result matches the expected metrics

    Args:
      expected_metrics: The expected metrics dict containing the exact metric
        keys and value.
      metric: The metric instance to test.
      extracts: The extracts to use as input to the evaluator. These should be
        of the format that would be produced by applying the Input-, Features-,
        Predictions-, Labels- and ExampleWeight- Extractors.
      example_weighted: Whether the metric is example weighted.
      enable_debug_print: Whether to print the beam PCollections after each
        stage.

    Raises:
      AssertionError: If the metric does not have the expected values.
    """

    def debug_print(element, stage_name):
      if enable_debug_print:
        print(f'[{stage_name}]\t{element}')
      return element

    computations = evaluator._filter_and_separate_computations(  # pylint: disable=protected-access
        metric.computations(example_weighted=example_weighted)
    )
    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(extracts)
          | 'PrintAfterCreate' >> beam.Map(debug_print, 'AfterCreate')
          | 'AddSlice'
          >> beam.Map(
              lambda x: x
              | {
                  constants.SLICE_KEY_TYPES_KEY: np.array(
                      [slicer_lib.slice_keys_to_numpy_array([()])]
                  )
              }
          )
          | 'PrintAfterAddSlice' >> beam.Map(debug_print, 'AfterAddSlice')
          | 'Preprocess'
          >> beam.ParDo(
              evaluator._PreprocessorDoFn(  # pylint: disable=protected-access
                  computations.non_derived_computations
              )
          )
          | 'PrintAfterPreprocess' >> beam.Map(debug_print, 'AfterPreprocess')
          | 'FanoutSlices' >> slicer_lib.FanoutSlices()
          | 'PrintAfterFanoutSlices'
          >> beam.Map(debug_print, 'AfterFanoutSlices')
          | 'ComputeNonDerivedMetrics'
          >> beam.CombinePerKey(
              evaluator._ComputationsCombineFn(  # pylint: disable=protected-access
                  computations=computations.non_derived_computations
              )
          )
          | 'PrintAfterComputeNonDerivedMetrics'
          >> beam.Map(debug_print, 'AfterComputeNonDerivedMetrics')
          | 'ComputeDerivedMetrics'
          >> evaluator._AddDerivedCrossSliceAndDiffMetrics(  # pylint: disable=protected-access
              derived_computations=computations.derived_computations,
              cross_slice_computations=[],
              cross_slice_specs=[],
          )
          | 'PrintAfterComputeDerivedMetrics'
          >> beam.Map(debug_print, 'AfterComputeDerivedMetrics')
          | 'RemovePrivateMetrics'
          >> beam.MapTuple(evaluator._remove_private_metrics)  # pylint: disable=protected-access
      )

      # pylint: enable=no-value-for-parameter
      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual((), got_slice_key)
          self.assertEqual(expected_metrics.keys(), got_metrics.keys())
          for key, expected_value in expected_metrics.items():
            self.assertIn(key, got_metrics)
            if isinstance(expected_value, np.ndarray):
              if np.issubdtype(expected_value.dtype, np.floating):
                np.testing.assert_almost_equal(
                    expected_value, got_metrics[key], decimal=5
                )
              else:
                np.testing.assert_array_equal(expected_value, got_metrics[key])
            else:
              self.assertEqual(expected_value, got_metrics[key])
        except AssertionError as err:
          raise beam.testing.util.BeamAssertException(err)

      beam.testing.util.assert_that(result, check_result, label='result')
