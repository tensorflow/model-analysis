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
"""Tests for calibration histogram."""

import dataclasses
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.metrics import calibration_histogram
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.utils import test_util


class CalibrationHistogramTest(test_util.TensorflowModelAnalysisTest):

  def testCalibrationHistogram(self):
    histogram = calibration_histogram.calibration_histogram(
        example_weighted=True)[0]

    example1 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.2]),
        'example_weights': np.array([1.0])
    }
    example2 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.8]),
        'example_weights': np.array([2.0])
    }
    example3 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.5]),
        'example_weights': np.array([3.0])
    }
    example4 = {
        'labels': np.array([1.0]),
        'predictions': np.array([-0.1]),
        'example_weights': np.array([4.0])
    }
    example5 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.5]),
        'example_weights': np.array([5.0])
    }
    example6 = {
        'labels': np.array([1.0]),
        'predictions': np.array([0.8]),
        'example_weights': np.array([6.0])
    }
    example7 = {
        'labels': np.array([0.0]),
        'predictions': np.array([0.2]),
        'example_weights': np.array([7.0])
    }
    example8 = {
        'labels': np.array([1.0]),
        'predictions': np.array([1.1]),
        'example_weights': np.array([8.0])
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([
              example1, example2, example3, example4, example5, example6,
              example7, example8
          ])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_plots = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_plots, 1)
          key = metric_types.PlotKey(
              (
                  '_calibration_histogram:fractional_labels=True,left=0.0,'
                  'num_buckets=10000,prediction_based_bucketing=True,right=1.0'
              ),
              example_weighted=True,
          )
          self.assertIn(key, got_plots)
          got_histogram = got_plots[key]
          self.assertLen(got_histogram, 5)
          self.assertEqual(
              got_histogram[0],
              calibration_histogram.Bucket(
                  bucket_id=0,
                  weighted_labels=1.0 * 4.0,
                  weighted_predictions=-0.1 * 4.0,
                  weighted_examples=4.0))
          self.assertEqual(
              got_histogram[1],
              calibration_histogram.Bucket(
                  bucket_id=2001,
                  weighted_labels=0.0 + 0.0,
                  weighted_predictions=0.2 + 7 * 0.2,
                  weighted_examples=1.0 + 7.0))
          self.assertEqual(
              got_histogram[2],
              calibration_histogram.Bucket(
                  bucket_id=5001,
                  weighted_labels=1.0 * 5.0,
                  weighted_predictions=0.5 * 3.0 + 0.5 * 5.0,
                  weighted_examples=3.0 + 5.0))
          self.assertEqual(
              got_histogram[3],
              calibration_histogram.Bucket(
                  bucket_id=8001,
                  weighted_labels=1.0 * 2.0 + 1.0 * 6.0,
                  weighted_predictions=0.8 * 2.0 + 0.8 * 6.0,
                  weighted_examples=2.0 + 6.0))
          self.assertEqual(
              got_histogram[4],
              calibration_histogram.Bucket(
                  bucket_id=10001,
                  weighted_labels=1.0 * 8.0,
                  weighted_predictions=1.1 * 8.0,
                  weighted_examples=8.0))

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testCalibrationHistogramWithK(self):
    histogram = calibration_histogram.calibration_histogram(
        sub_key=metric_types.SubKey(k=2), example_weighted=True)[0]

    example1 = {
        'labels': np.array([2]),
        'predictions': np.array([0.2, 0.05, 0.1, 0.05]),
        'example_weights': np.array([1.0])
    }
    example2 = {
        'labels': np.array([2]),
        'predictions': np.array([0.7, 0.1, 0.8, 0.5]),
        'example_weights': np.array([2.0])
    }
    example3 = {
        'labels': np.array([3]),
        'predictions': np.array([0.1, 0.5, 0.3, 0.4]),
        'example_weights': np.array([3.0])
    }
    example4 = {
        'labels': np.array([0]),
        'predictions': np.array([-0.1, -0.2, -0.7, -0.4]),
        'example_weights': np.array([4.0])
    }
    example5 = {
        'labels': np.array([1]),
        'predictions': np.array([0.3, 0.5, 0.0, 0.4]),
        'example_weights': np.array([5.0])
    }
    example6 = {
        'labels': np.array([2]),
        'predictions': np.array([0.1, 0.1, 0.8, 0.7]),
        'example_weights': np.array([6.0])
    }
    example7 = {
        'labels': np.array([2]),
        'predictions': np.array([0.0, 0.2, 0.1, 0.0]),
        'example_weights': np.array([7.0])
    }
    example8 = {
        'labels': np.array([0]),
        'predictions': np.array([1.1, 0.3, 1.05, 0.2]),
        'example_weights': np.array([8.0])
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([
              example1, example2, example3, example4, example5, example6,
              example7, example8
          ])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_plots = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_plots, 1)
          key = metric_types.PlotKey(
              name=(
                  '_calibration_histogram:fractional_labels=True,left=0.0,'
                  'num_buckets=10000,prediction_based_bucketing=True,right=1.0'
              ),
              sub_key=metric_types.SubKey(k=2),
              example_weighted=True,
          )
          self.assertIn(key, got_plots)
          got_histogram = got_plots[key]
          self.assertLen(got_histogram, 5)
          self.assertEqual(
              got_histogram[0],
              calibration_histogram.Bucket(
                  bucket_id=0,
                  weighted_labels=0.0 * 4.0,
                  weighted_predictions=-0.2 * 4.0,
                  weighted_examples=4.0))
          self.assertEqual(
              got_histogram[1],
              calibration_histogram.Bucket(
                  bucket_id=1001,
                  weighted_labels=1.0 + 7 * 1.0,
                  weighted_predictions=0.1 + 7 * 0.1,
                  weighted_examples=1.0 + 7.0))
          self.assertEqual(
              got_histogram[2],
              calibration_histogram.Bucket(
                  bucket_id=4001,
                  weighted_labels=1.0 * 3.0 + 0.0 * 5.0,
                  weighted_predictions=0.4 * 3.0 + 0.4 * 5.0,
                  weighted_examples=3.0 + 5.0))
          self.assertEqual(
              got_histogram[3],
              calibration_histogram.Bucket(
                  bucket_id=7001,
                  weighted_labels=0.0 * 2.0 + 0.0 * 6.0,
                  weighted_predictions=0.7 * 2.0 + 0.7 * 6.0,
                  weighted_examples=2.0 + 6.0))
          self.assertEqual(
              got_histogram[4],
              calibration_histogram.Bucket(
                  bucket_id=10001,
                  weighted_labels=0.0 * 8.0,
                  weighted_predictions=1.05 * 8.0,
                  weighted_examples=8.0))

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testTopKCalibrationHistogramWithTopK(self):
    histogram = calibration_histogram.calibration_histogram(
        sub_key=metric_types.SubKey(top_k=2), example_weighted=True)[0]

    example1 = {
        'labels': np.array([2]),
        'predictions': np.array([0.2, 0.05, 0.5, 0.05]),
        'example_weights': np.array([1.0])
    }
    example2 = {
        'labels': np.array([2]),
        'predictions': np.array([0.8, 0.1, 0.8, 0.5]),
        'example_weights': np.array([2.0])
    }
    example3 = {
        'labels': np.array([3]),
        'predictions': np.array([0.2, 0.5, 0.1, 0.1]),
        'example_weights': np.array([3.0])
    }
    example4 = {
        'labels': np.array([0]),
        'predictions': np.array([-0.1, 1.1, -0.7, -0.4]),
        'example_weights': np.array([4.0])
    }

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([example1, example2, example3, example4])
          | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
          | 'AddSlice' >> beam.Map(lambda x: ((), x))
          | 'ComputeHistogram' >> beam.CombinePerKey(histogram.combiner))

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_plots = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertLen(got_plots, 1)
          key = metric_types.PlotKey(
              name=(
                  '_calibration_histogram:fractional_labels=True,left=0.0,'
                  'num_buckets=10000,prediction_based_bucketing=True,right=1.0'
              ),
              sub_key=metric_types.SubKey(top_k=2),
              example_weighted=True,
          )
          self.assertIn(key, got_plots)
          got_histogram = got_plots[key]
          self.assertLen(got_histogram, 5)
          self.assertEqual(
              got_histogram[0],
              calibration_histogram.Bucket(
                  bucket_id=0,
                  weighted_labels=3.0 + 4.0,
                  weighted_predictions=(2 * 1.0 * float('-inf') +
                                        2 * 2.0 * float('-inf') +
                                        2 * 3.0 * float('-inf') +
                                        2 * 4.0 * float('-inf') + -0.1 * 4.0),
                  weighted_examples=(1.0 * 2.0 + 2.0 * 2.0 + 3.0 * 2.0 +
                                     4.0 * 3.0)))
          self.assertEqual(
              got_histogram[1],
              calibration_histogram.Bucket(
                  bucket_id=2001,
                  weighted_labels=0.0 + 0.0,
                  weighted_predictions=0.2 + 3 * 0.2,
                  weighted_examples=1.0 + 3.0))
          self.assertEqual(
              got_histogram[2],
              calibration_histogram.Bucket(
                  bucket_id=5001,
                  weighted_labels=1.0 + 0.0 * 3.0,
                  weighted_predictions=0.5 * 1.0 + 0.5 * 3.0,
                  weighted_examples=1.0 + 3.0))
          self.assertEqual(
              got_histogram[3],
              calibration_histogram.Bucket(
                  bucket_id=8001,
                  weighted_labels=0.0 * 2.0 + 1.0 * 2.0,
                  weighted_predictions=0.8 * 2.0 + 0.8 * 2.0,
                  weighted_examples=2.0 + 2.0))
          self.assertEqual(
              got_histogram[4],
              calibration_histogram.Bucket(
                  bucket_id=10001,
                  weighted_labels=0.0 * 4.0,
                  weighted_predictions=1.1 * 4.0,
                  weighted_examples=4.0))

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testRebin(self):
    # [Bucket(0, -1, -0.01), Bucket(1, 0, 0) ... Bucket(101, 101, 1.01)]
    histogram = [calibration_histogram.Bucket(0, -1, -.01, 1.0)]
    for i in range(100):
      histogram.append(calibration_histogram.Bucket(i + 1, i, i * .01, 1.0))
    histogram.append(calibration_histogram.Bucket(101, 101, 1.01, 1.0))
    # [-1e-7, 0.0, 0.1, ..., 0.9, 1.0, 1.0+1e-7]
    thresholds = [-1e-7] + [i * 1.0 / 10 for i in range(11)] + [1.0 + 1e-7]
    got = calibration_histogram.rebin(thresholds, histogram, 100)

    # labels = (10 * (i-1)) + (1 + 2 + 3 + ... + 9)
    expected = [
        calibration_histogram.Bucket(0, -1, -0.01, 1.0),
        calibration_histogram.Bucket(1, 45.0, 0.45, 10.0),
        calibration_histogram.Bucket(2, 145.0, 1.45, 10.0),
        calibration_histogram.Bucket(3, 245.0, 2.45, 10.0),
        calibration_histogram.Bucket(4, 345.0, 3.45, 10.0),
        calibration_histogram.Bucket(5, 445.0, 4.45, 10.0),
        calibration_histogram.Bucket(6, 545.0, 5.45, 10.0),
        calibration_histogram.Bucket(7, 645.0, 6.45, 10.0),
        calibration_histogram.Bucket(8, 745.0, 7.45, 10.0),
        calibration_histogram.Bucket(9, 845.0, 8.45, 10.0),
        calibration_histogram.Bucket(10, 945.0, 9.45, 10.0),
        calibration_histogram.Bucket(11, 0.0, 0.0, 0.0),
        calibration_histogram.Bucket(12, 101.0, 1.01, 1.0),
    ]
    self.assertLen(got, len(expected))
    for i in range(len(got)):
      self.assertSequenceAlmostEqual(
          dataclasses.astuple(got[i]), dataclasses.astuple(expected[i]))

  def testRebinWithSparseData(self):
    histogram = [
        calibration_histogram.Bucket(4, 5.0, .25, 5.0),  # pred = .05
        calibration_histogram.Bucket(61, 60.0, 36.0, 60.0),  # pred = .6
        calibration_histogram.Bucket(70, 69.0, 47.61, 69.0),  # pred = .69
        calibration_histogram.Bucket(100, 99.0, 98.01, 99.0)  # pred = .99
    ]
    # [0, 0.1, ..., 0.9, 1.0]
    thresholds = [i * 1.0 / 10 for i in range(0, 11)]
    got = calibration_histogram.rebin(thresholds, histogram, 100)

    expected = [
        calibration_histogram.Bucket(0, 5.0, 0.25, 5.0),
        calibration_histogram.Bucket(1, 0.0, 0.0, 0.0),
        calibration_histogram.Bucket(2, 0.0, 0.0, 0.0),
        calibration_histogram.Bucket(3, 0.0, 0.0, 0.0),
        calibration_histogram.Bucket(4, 0.0, 0.0, 0.0),
        calibration_histogram.Bucket(5, 0.0, 0.0, 0.0),
        calibration_histogram.Bucket(6, 129.0, 83.61, 129.0),
        calibration_histogram.Bucket(7, 0.0, 0.0, 0.0),
        calibration_histogram.Bucket(8, 0.0, 0.0, 0.0),
        calibration_histogram.Bucket(9, 99.0, 98.01, 99.0),
        calibration_histogram.Bucket(10, 0.0, 0.0, 0.0),
    ]
    self.assertLen(got, len(expected))
    for i in range(len(got)):
      self.assertSequenceAlmostEqual(
          dataclasses.astuple(got[i]), dataclasses.astuple(expected[i]))


