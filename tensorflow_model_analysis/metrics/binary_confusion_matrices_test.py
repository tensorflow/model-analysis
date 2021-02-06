# Lint as: python3
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
"""Tests for binary confusion matrices."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.metrics import binary_confusion_matrices
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util


class BinaryConfusionMatricesTest(testutil.TensorflowModelAnalysisTest,
                                  parameterized.TestCase):

  @parameterized.named_parameters(
      ('using_num_thresholds', {
          'num_thresholds': 3,
      },
       binary_confusion_matrices.Matrices(
           thresholds=[-1e-7, 0.5, 1.0 + 1e-7],
           tp=[2.0, 1.0, 0.0],
           fp=[2.0, 0.0, 0.0],
           tn=[0.0, 2.0, 2.0],
           fn=[0.0, 1.0, 2.0],
           tp_examples=[],
           tn_examples=[],
           fp_examples=[],
           fn_examples=[])),
      ('single_threshold', {
          'thresholds': [0.5],
          'use_histogram': True,
      },
       binary_confusion_matrices.Matrices(
           thresholds=[0.5],
           tp=[1.0],
           fp=[0.0],
           tn=[2.0],
           fn=[1.0],
           tp_examples=[],
           tn_examples=[],
           fp_examples=[],
           fn_examples=[])),
      ('inner_thresholds', {
          'thresholds': [0.25, 0.75],
          'use_histogram': True,
      },
       binary_confusion_matrices.Matrices(
           thresholds=[0.25, 0.75],
           tp=[2.0, 1.0],
           fp=[1.0, 0.0],
           tn=[1.0, 2.0],
           fn=[0.0, 1.0],
           tp_examples=[],
           tn_examples=[],
           fp_examples=[],
           fn_examples=[])),
      ('boundary_thresholds', {
          'thresholds': [0.0, 1.0],
          'use_histogram': True,
      },
       binary_confusion_matrices.Matrices(
           thresholds=[0.0, 1.0],
           tp=[2.0, 0.0],
           fp=[2.0, 0.0],
           tn=[0.0, 2.0],
           fn=[0.0, 2.0],
           tp_examples=[],
           tn_examples=[],
           fp_examples=[],
           fn_examples=[])),
      ('left_boundary', {
          'thresholds': [0.0, 0.5],
          'use_histogram': True,
      },
       binary_confusion_matrices.Matrices(
           thresholds=[0.0, 0.5],
           tp=[2.0, 1.0],
           fp=[2.0, 0.0],
           tn=[0.0, 2.0],
           fn=[0.0, 1.0],
           tp_examples=[],
           tn_examples=[],
           fp_examples=[],
           fn_examples=[])),
      ('right_boundary', {
          'thresholds': [0.5, 1.0],
          'use_histogram': True,
      },
       binary_confusion_matrices.Matrices(
           thresholds=[0.5, 1.0],
           tp=[1.0, 0.0],
           fp=[0.0, 0.0],
           tn=[2.0, 2.0],
           fn=[1.0, 2.0],
           tp_examples=[],
           tn_examples=[],
           fp_examples=[],
           fn_examples=[])),
  )
  def testBinaryConfusionMatrices(self, kwargs, expected_matrices):
    computations = binary_confusion_matrices.binary_confusion_matrices(**kwargs)
    histogram = computations[0]
    matrices = computations[1]

    example1 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.0]),
        'example_weights': np.array([1.0])
    }
    example2 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.5]),
        'example_weights': np.array([1.0])
    }
    example3 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.3]),
        'example_weights': np.array([1.0])
    }
    example4 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.9]),
        'example_weights': np.array([1.0])
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
          |
          'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeMatrices' >> beam.Map(
              lambda x: (x[0], matrices.result(x[1]))))  # pyformat: disable

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, 1)
          name = '_binary_confusion_matrices_{}'.format(
              kwargs['num_thresholds'] if 'num_thresholds' in
              kwargs else kwargs['thresholds'])
          key = metric_types.MetricKey(name=name)
          self.assertIn(key, got_metrics)
          got_matrices = got_metrics[key]
          self.assertEqual(got_matrices, expected_matrices)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(('using_num_thresholds', {
      'num_thresholds': 3,
      'use_histogram': False,
  },
                                   binary_confusion_matrices.Matrices(
                                       thresholds=[-1e-7, 0.5, 1.0 + 1e-7],
                                       tp=[2.0, 1.0, 0.0],
                                       fp=[2.0, 0.0, 0.0],
                                       tn=[0.0, 2.0, 2.0],
                                       fn=[0.0, 1.0, 2.0],
                                       tp_examples=[[], [], []],
                                       tn_examples=[[], [], []],
                                       fp_examples=[[], [], []],
                                       fn_examples=[[], [], []])),
                                  ('single_threshold', {
                                      'thresholds': [0.5],
                                  },
                                   binary_confusion_matrices.Matrices(
                                       thresholds=[0.5],
                                       tp=[1.0],
                                       fp=[0.0],
                                       tn=[2.0],
                                       fn=[1.0],
                                       tp_examples=[[]],
                                       tn_examples=[[]],
                                       fp_examples=[[]],
                                       fn_examples=[[]])),
                                  ('multiple_thresholds', {
                                      'thresholds': [0.25, 0.75],
                                  },
                                   binary_confusion_matrices.Matrices(
                                       thresholds=[0.25, 0.75],
                                       tp=[2.0, 1.0],
                                       fp=[1.0, 0.0],
                                       tn=[1.0, 2.0],
                                       fn=[0.0, 1.0],
                                       tp_examples=[[], []],
                                       tn_examples=[[], []],
                                       fp_examples=[[], []],
                                       fn_examples=[[], []])),
                                  ('with_example_ids', {
                                      'thresholds': [0.1, 0.9],
                                      'example_id_key': 'example_id_key',
                                      'example_ids_count': 2,
                                  },
                                   binary_confusion_matrices.Matrices(
                                       thresholds=[0.1, 0.9],
                                       tp=[2.0, 0.0],
                                       fp=[1.0, 0.0],
                                       tn=[1.0, 2.0],
                                       fn=[0.0, 2.0],
                                       tp_examples=[['id_3', 'id_4'], []],
                                       tn_examples=[['id_1'], ['id_1', 'id_2']],
                                       fp_examples=[['id_2'], []],
                                       fn_examples=[[], ['id_3', 'id_4']])))
  def testBinaryConfusionMatrices_noHistograms(self, kwargs, expected_matrices):
    computations = binary_confusion_matrices.binary_confusion_matrices(**kwargs)
    histogram = computations[0]
    matrices = computations[1]

    example1 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.0]),
        'example_weights': np.array([1.0]),
        'features': {
            'example_id_key': np.array(['id_1']),
        },
    }
    example2 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.5]),
        'example_weights': np.array([1.0]),
        'features': {
            'example_id_key': np.array(['id_2']),
        },
    }
    example3 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.3]),
        'example_weights': np.array([1.0]),
        'features': {
            'example_id_key': np.array(['id_3']),
        },
    }
    example4 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.9]),
        'example_weights': np.array([1.0]),
        'features': {
            'example_id_key': np.array(['id_4']),
        },
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
          |
          'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeMatrices' >> beam.Map(
              lambda x: (x[0], matrices.result(x[1]))))  # pyformat: disable

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, 1)
          name = '_binary_confusion_matrices_{}'.format(
              kwargs['num_thresholds'] if 'num_thresholds' in
              kwargs else kwargs['thresholds'])
          key = metric_types.MetricKey(name=name)
          self.assertIn(key, got_metrics)
          got_matrices = got_metrics[key]
          self.assertEqual(got_matrices, expected_matrices)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testBinaryConfusionMatricesTopK(self):
    computations = binary_confusion_matrices.binary_confusion_matrices(
        thresholds=[float('-inf')],
        sub_key=metric_types.SubKey(top_k=3),
        use_histogram=True)
    histogram = computations[0]
    matrices = computations[1]

    example1 = {
        'labels': np.array([2]),
        'predictions': np.array([0.1, 0.2, 0.1, 0.25, 0.35]),
        'example_weights': np.array([1.0])
    }
    example2 = {
        'labels': np.array([1]),
        'predictions': np.array([0.2, 0.3, 0.05, 0.15, 0.3]),
        'example_weights': np.array([1.0])
    }
    example3 = {
        'labels': np.array([3]),
        'predictions': np.array([0.01, 0.2, 0.09, 0.5, 0.2]),
        'example_weights': np.array([1.0])
    }
    example4 = {
        'labels': np.array([4]),
        'predictions': np.array([0.3, 0.2, 0.05, 0.4, 0.05]),
        'example_weights': np.array([1.0])
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
          |
          'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner)
          | 'ComputeMatrices' >> beam.Map(
              lambda x: (x[0], matrices.result(x[1]))))  # pyformat: disable

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_metrics, 1)
          key = metric_types.MetricKey(
              name='_binary_confusion_matrices_[-inf]',
              sub_key=metric_types.SubKey(top_k=3))
          self.assertIn(key, got_metrics)
          got_matrices = got_metrics[key]
          self.assertEqual(
              got_matrices,
              binary_confusion_matrices.Matrices(
                  thresholds=[float('-inf')],
                  tp=[2.0],
                  fp=[10.0],
                  tn=[6.0],
                  fn=[2.0],
                  tp_examples=[],
                  tn_examples=[],
                  fp_examples=[],
                  fn_examples=[]))

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.test.main()
