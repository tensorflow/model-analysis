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
"""Tests for view_types."""

from absl.testing import parameterized

import tensorflow as tf
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.view import view_types


class ViewTypesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('empty_subkey', None, None, None), ('class_id', 1, None, None),
      ('top_k', None, None, 1), ('k', None, 1, None))
  def testEvalResultGetMetrics(self, class_id, k, top_k):

    # Slices
    num_slices = 2
    overall_slice = ()
    male_slice = (('gender', 'male'))

    # Output name
    output_name = ''

    # Metrics for overall slice
    metrics_overall = {
        'accuracy': {
            'doubleValue': 0.5,
        },
        'auc': {
            'doubleValue': 0.8,
        },
    }

    # Metrics for male slice
    metrics_male = {
        'accuracy': {
            'doubleValue': 0.8,
        },
        'auc': {
            'doubleValue': 0.5,
        }
    }

    # EvalResult
    if class_id or k or top_k:
      sub_key = str(metric_types.SubKey(class_id, k, top_k))
    else:
      sub_key = ''
    slicing_metrics = [(overall_slice, {
        output_name: {
            sub_key: metrics_overall
        }
    }), (male_slice, {
        output_name: {
            sub_key: metrics_male
        }
    })]
    eval_result = view_types.EvalResult(slicing_metrics, None, None, None, None,
                                        None, None)

    # Test get_metrics_for_all_slices()
    actual_metrics = eval_result.get_metrics_for_all_slices(
        class_id=class_id, k=k, top_k=top_k)

    # Assert there is one metrics entry per slice
    self.assertLen(actual_metrics, num_slices)

    # Assert the metrics match the expected values
    self.assertDictEqual(actual_metrics[overall_slice], metrics_overall)
    self.assertDictEqual(actual_metrics[male_slice], metrics_male)

    # Test get_metrics_for_slice()
    self.assertDictEqual(
        eval_result.get_metrics_for_slice(class_id=class_id, k=k, top_k=top_k),
        metrics_overall)
    self.assertDictEqual(
        eval_result.get_metrics_for_slice(
            slice_name=male_slice, class_id=class_id, k=k, top_k=top_k),
        metrics_male)

    # Test get_metric_names()
    self.assertSameElements(eval_result.get_metric_names(), ['accuracy', 'auc'])

    # Test get_slice_names()
    self.assertListEqual(eval_result.get_slice_names(),
                         [overall_slice, male_slice])

  @parameterized.named_parameters(
      ('empty_subkey', None, None, None), ('class_id', 1, None, None),
      ('top_k', None, None, 1), ('k', None, 1, None))
  def testEvalResultGetAttributions(self, class_id, k, top_k):

    # Slices
    num_slices = 2
    overall_slice = ()
    male_slice = (('gender', 'male'))

    # Output name
    output_name = ''

    # Metric Name
    metric_name = 'total_attributions'

    # Attributions for overall slice
    attributions_overall = {
        'feature1': {
            'doubleValue': 0.5,
        },
        'feature2': {
            'doubleValue': 0.8,
        },
    }

    # Attributions for male slice
    attributions_male = {
        'feature1': {
            'doubleValue': 0.8,
        },
        'feature2': {
            'doubleValue': 0.5,
        }
    }

    # EvalResult
    if class_id or k or top_k:
      sub_key = str(metric_types.SubKey(class_id, k, top_k))
    else:
      sub_key = ''
    attributions = [(overall_slice, {
        output_name: {
            sub_key: {
                metric_name: attributions_overall
            }
        }
    }), (male_slice, {
        output_name: {
            sub_key: {
                metric_name: attributions_male
            }
        }
    })]
    eval_result = view_types.EvalResult(None, None, attributions, None, None,
                                        None, None)

    # Test get_attributions_for_all_slices()
    actual_attributions = eval_result.get_attributions_for_all_slices(
        class_id=class_id, k=k, top_k=top_k)

    # Assert there is one attributions entry per slice
    self.assertLen(actual_attributions, num_slices)

    # Assert the attributions match the expected values
    self.assertDictEqual(actual_attributions[overall_slice],
                         attributions_overall)
    self.assertDictEqual(actual_attributions[male_slice], attributions_male)

    # Test get_attributions_for_slice()
    self.assertDictEqual(
        eval_result.get_attributions_for_slice(
            class_id=class_id, k=k, top_k=top_k), attributions_overall)
    self.assertDictEqual(
        eval_result.get_attributions_for_slice(
            slice_name=male_slice,
            metric_name=metric_name,
            class_id=class_id,
            k=k,
            top_k=top_k), attributions_male)


if __name__ == '__main__':
  tf.test.main()
