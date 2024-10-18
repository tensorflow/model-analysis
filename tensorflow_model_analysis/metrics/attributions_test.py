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
"""Tests for attributions metrics."""


import pytest
from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.metrics import attributions
from tensorflow_model_analysis.metrics import metric_specs
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.utils import test_util
from tensorflow_model_analysis.utils.keras_lib import tf_keras


@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class AttributionsTest(
    test_util.TensorflowModelAnalysisTest, parameterized.TestCase
):

  def testHasAttributionsMetrics(self):
    specs_with_attributions = metric_specs.specs_from_metrics({
        'output_name': [
            tf_keras.metrics.MeanSquaredError('mse'),
            attributions.TotalAttributions(),
        ]
    })
    self.assertTrue(
        attributions.has_attributions_metrics(specs_with_attributions))
    specs_without_attributions = metric_specs.specs_from_metrics([
        tf_keras.metrics.MeanSquaredError('mse'),
    ])
    self.assertFalse(
        attributions.has_attributions_metrics(specs_without_attributions))

  def testMeanAttributions(self):
    computation = attributions.MeanAttributions().computations()[-1]

    total_attributions_key = metric_types.AttributionsKey(
        name='_total_attributions')
    example_count_key = metric_types.MetricKey(name='example_count')
    metrics = {
        total_attributions_key: {
            'feature1': 1.0,
            'feature2': -2.0
        },
        example_count_key: 0.5
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([metrics])
          | 'ComputeMetric' >> beam.Map(lambda x: ((), computation.result(x))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_attributions = got[0]
          self.assertEqual(got_slice_key, ())
          mean_attributions_key = metric_types.AttributionsKey(
              name='mean_attributions')
          self.assertIn(mean_attributions_key, got_attributions)
          self.assertDictElementsAlmostEqual(
              got_attributions[mean_attributions_key], {
                  'feature1': 1.0 / 0.5,
                  'feature2': -2.0 / 0.5,
              })

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testMeanAbsoluteAttributions(self):
    computation = attributions.MeanAbsoluteAttributions().computations()[-1]

    total_absolute_attributions_key = metric_types.AttributionsKey(
        name='_total_absolute_attributions')
    example_count_key = metric_types.MetricKey(name='example_count')
    metrics = {
        total_absolute_attributions_key: {
            'feature1': 1.0,
            'feature2': 2.0
        },
        example_count_key: 0.5
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([metrics])
          | 'ComputeMetric' >> beam.Map(lambda x: ((), computation.result(x))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_attributions = got[0]
          self.assertEqual(got_slice_key, ())
          mean_attributions_key = metric_types.AttributionsKey(
              name='mean_absolute_attributions')
          self.assertIn(mean_attributions_key, got_attributions)
          self.assertDictElementsAlmostEqual(
              got_attributions[mean_attributions_key], {
                  'feature1': 1.0 / 0.5,
                  'feature2': 2.0 / 0.5,
              })

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      {
          'testcase_name': 'basic',
          'model_name': '',
          'output_name': '',
          'examples': [{
              'labels': None,
              'predictions': None,
              'example_weights': np.array(1.0),
              'attributions': {
                  'feature1': 1.1,
                  'feature2': -1.2,
              }
          }, {
              'labels': None,
              'predictions': None,
              'example_weights': np.array(1.0),
              'attributions': {
                  'feature1': -2.1,
                  'feature2': 2.2
              }
          }, {
              'labels': None,
              'predictions': None,
              'example_weights': np.array(1.0),
              'attributions': {
                  'feature1': 3.1,
                  'feature2': -3.2
              }
          }],
          'expected_values': {
              'feature1': (1.1 - 2.1 + 3.1),
              'feature2': (-1.2 + 2.2 - 3.2),
          },
      },
      {
          'testcase_name': 'multi-model',
          'model_name': 'model',
          'output_name': '',
          'examples': [{
              'labels': None,
              'predictions': None,
              'example_weights': np.array(1.0),
              'attributions': {
                  'model': {
                      'feature1': 11.1,
                      'feature2': -11.2
                  },
              }
          }, {
              'labels': None,
              'predictions': None,
              'example_weights': np.array(1.0),
              'attributions': {
                  'model': {
                      'feature1': -22.1,
                      'feature2': 22.2
                  },
              }
          }, {
              'labels': None,
              'predictions': None,
              'example_weights': np.array(1.0),
              'attributions': {
                  'model': {
                      'feature1': 33.1,
                      'feature2': -33.2
                  },
              }
          }],
          'expected_values': {
              'feature1': (11.1 - 22.1 + 33.1),
              'feature2': (-11.2 + 22.2 - 33.2),
          },
      },
      {
          'testcase_name': 'multi-model-multi-output',
          'model_name': 'model',
          'output_name': 'output',
          'examples': [{
              'labels': None,
              'predictions': None,
              'example_weights': np.array(1.0),
              'attributions': {
                  'model': {
                      'output': {
                          'feature1': 111.1,
                          'feature2': -111.2
                      },
                  },
              }
          }, {
              'labels': None,
              'predictions': None,
              'example_weights': np.array(1.0),
              'attributions': {
                  'model': {
                      'output': {
                          'feature1': -222.1,
                          'feature2': 222.2
                      },
                  },
              }
          }, {
              'labels': None,
              'predictions': None,
              'example_weights': np.array(1.0),
              'attributions': {
                  'model': {
                      'output': {
                          'feature1': 333.1,
                          'feature2': -333.2
                      },
                  },
              }
          }],
          'expected_values': {
              'feature1': (111.1 - 222.1 + 333.1),
              'feature2': (-111.2 + 222.2 - 333.2),
          },
      },
  )
  def testTotalAttributionsWithMultiModelsAndOutputs(self, model_name,
                                                     output_name, examples,
                                                     expected_values):
    computations = attributions.TotalAttributions().computations(
        model_names=[model_name], output_names=[output_name])

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(examples)
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          |
          'CombineAttributions' >> beam.CombinePerKey(computations[0].combiner)
          | 'ComputeResult' >> beam.Map(  # comment to add lamda on own line
              lambda x: (x[0], computations[1].result(x[1]))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_attributions = got[0]
          self.assertEqual(got_slice_key, ())
          total_attributions_key = metric_types.AttributionsKey(
              name='total_attributions',
              model_name=model_name,
              output_name=output_name)
          self.assertIn(total_attributions_key, got_attributions)
          self.assertDictElementsAlmostEqual(
              got_attributions[total_attributions_key], expected_values)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(('empty', None, {
      'feature1': np.array([6.33, 6.39, 6.36]),
      'feature2': np.array([6.63, 6.69, 6.66]),
  }), ('class_id', metric_types.SubKey(class_id=0), {
      'feature1': 6.33,
      'feature2': 6.63,
  }), ('k', metric_types.SubKey(k=2), {
      'feature1': 6.36,
      'feature2': 6.66,
  }), ('top_k', metric_types.SubKey(top_k=2), {
      'feature1': np.array([6.39, 6.36]),
      'feature2': np.array([6.69, 6.66]),
  }))
  def testTotalAttributionsWithSubKeys(self, sub_key, expected_values):
    computations = attributions.TotalAttributions().computations(
        sub_keys=[sub_key])

    example1 = {
        'labels': None,
        'predictions': None,
        'example_weights': np.array(1.0),
        'attributions': {
            'feature1': [1.11, 1.13, 1.12],
            'feature2': [1.21, 1.23, 1.22]
        }
    }
    example2 = {
        'labels': None,
        'predictions': None,
        'example_weights': np.array(1.0),
        'attributions': {
            'feature1': [2.11, 2.13, 2.12],
            'feature2': [2.21, 2.23, 2.22]
        }
    }
    example3 = {
        'labels': None,
        'predictions': None,
        'example_weights': np.array(1.0),
        'attributions': {
            'feature1': np.array([3.11, 3.13, 3.12]),
            'feature2': np.array([3.21, 3.23, 3.22])
        }
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          |
          'CombineAttributions' >> beam.CombinePerKey(computations[0].combiner)
          | 'ComputeResult' >> beam.Map(  # comment to add lamda on own line
              lambda x: (x[0], computations[1].result(x[1]))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_attributions = got[0]
          self.assertEqual(got_slice_key, ())
          total_attributions_key = metric_types.AttributionsKey(
              name='total_attributions', sub_key=sub_key)
          self.assertIn(total_attributions_key, got_attributions)
          self.assertAllClose(got_attributions[total_attributions_key],
                              expected_values)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      {
          'testcase_name': 'basic',
          'model_name': '',
          'output_name': '',
          'examples': [{
              'labels': None,
              'predictions': None,
              'example_weights': np.array(0.5),
              'attributions': {
                  'feature1': 1.1,
                  'feature2': -1.2,
              }
          }, {
              'labels': None,
              'predictions': None,
              'example_weights': np.array(0.7),
              'attributions': {
                  'feature1': 2.1,
                  'feature2': -2.2
              }
          }, {
              'labels': None,
              'predictions': None,
              'example_weights': np.array(0.9),
              'attributions': {
                  'feature1': 3.1,
                  'feature2': -3.2
              }
          }],
          'expected_values': {
              'feature1': (1.1 * 0.5 + 2.1 * 0.7 + 3.1 * 0.9),
              'feature2': (1.2 * 0.5 + 2.2 * 0.7 + 3.2 * 0.9),
          },
      },
      {
          'testcase_name': 'multi-model',
          'model_name': 'model',
          'output_name': '',
          'examples': [{
              'labels': None,
              'predictions': None,
              'example_weights': None,
              'attributions': {
                  'model': {
                      'feature1': 11.1,
                      'feature2': -11.2
                  },
              }
          }, {
              'labels': None,
              'predictions': None,
              'example_weights': None,
              'attributions': {
                  'model': {
                      'feature1': 22.1,
                      'feature2': -22.2
                  },
              }
          }, {
              'labels': None,
              'predictions': None,
              'example_weights': None,
              'attributions': {
                  'model': {
                      'feature1': 33.1,
                      'feature2': -33.2
                  },
              }
          }],
          'expected_values': {
              'feature1': (11.1 + 22.1 + 33.1),
              'feature2': (11.2 + 22.2 + 33.2),
          },
      },
      {
          'testcase_name': 'multi-model-multi-output',
          'model_name': 'model',
          'output_name': 'output',
          'examples': [{
              'labels': None,
              'predictions': None,
              'example_weights': np.array(1.0),
              'attributions': {
                  'model': {
                      'output': {
                          'feature1': 111.1,
                          'feature2': -111.2
                      },
                  },
              }
          }, {
              'labels': None,
              'predictions': None,
              'example_weights': np.array(1.0),
              'attributions': {
                  'model': {
                      'output': {
                          'feature1': 222.1,
                          'feature2': -222.2
                      },
                  },
              }
          }, {
              'labels': None,
              'predictions': None,
              'example_weights': np.array(1.0),
              'attributions': {
                  'model': {
                      'output': {
                          'feature1': 333.1,
                          'feature2': -333.2
                      },
                  },
              }
          }],
          'expected_values': {
              'feature1': (111.1 + 222.1 + 333.1),
              'feature2': (111.2 + 222.2 + 333.2),
          },
      },
  )
  def testTotalAbsoluteAttributionsWithMultiModelsAndOutputs(
      self, model_name, output_name, examples, expected_values):
    computations = attributions.TotalAbsoluteAttributions().computations(
        model_names=[model_name],
        output_names=[output_name],
        example_weighted=True)

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(examples)
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          |
          'CombineAttributions' >> beam.CombinePerKey(computations[0].combiner)
          | 'ComputeResult' >> beam.Map(  # comment to add lamda on own line
              lambda x: (x[0], computations[1].result(x[1]))))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_attributions = got[0]
          self.assertEqual(got_slice_key, ())
          total_attributions_key = metric_types.AttributionsKey(
              name='total_absolute_attributions',
              model_name=model_name,
              output_name=output_name,
              example_weighted=True)
          self.assertIn(total_attributions_key, got_attributions)
          self.assertDictElementsAlmostEqual(
              got_attributions[total_attributions_key], expected_values)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


