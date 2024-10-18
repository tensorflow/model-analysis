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
"""Tests for metric utils."""

import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2


class UtilTest(tf.test.TestCase):

  def testNameGeneratorFromArguments(self):
    # Basic usage
    self.assertEqual(
        metric_util.generate_private_name_from_arguments('', threshold=[0.5]),
        '_:threshold=[0.5]',
    )
    # Private case with private name
    self.assertEqual(
        metric_util.generate_private_name_from_arguments(
            '_private', threshold=[0.5]
        ),
        '_private:threshold=[0.5]',
    )
    # Multiple arguments
    self.assertEqual(
        metric_util.generate_private_name_from_arguments(
            '_private', threshold=[0.5], class_id=[0], class_type=None
        ),
        '_private:class_id=[0],threshold=[0.5]',
    )

  def testToScalar(self):
    self.assertEqual(1, metric_util.to_scalar(np.array([1])))
    self.assertEqual(1.0, metric_util.to_scalar(np.array(1.0)))
    self.assertEqual('string', metric_util.to_scalar(np.array([['string']])))
    sparse_tensor = types.SparseTensorValue(
        indices=np.array([0]), values=np.array([1]), dense_shape=np.array([1])
    )
    self.assertEqual(1, metric_util.to_scalar(sparse_tensor))

  def testSafeToScalar(self):
    self.assertEqual(1, metric_util.safe_to_scalar(np.array([1])))
    self.assertEqual(1.0, metric_util.safe_to_scalar(np.array(1.0)))
    self.assertEqual(
        'string', metric_util.safe_to_scalar(np.array([['string']]))
    )
    self.assertEqual(0.0, metric_util.safe_to_scalar(np.array([])))
    self.assertEqual(0.0, metric_util.safe_to_scalar([]))
    with self.assertRaisesRegex(
        ValueError, 'Array should have exactly 1 value to a Python scalar'
    ):
      _ = 1, metric_util.safe_to_scalar([1])
    with self.assertRaisesRegex(
        ValueError, 'Array should have exactly 1 value to a Python scalar'
    ):
      _ = metric_util.safe_to_scalar([1, 2])
    with self.assertRaisesRegex(
        ValueError, 'Array should have exactly 1 value to a Python scalar'
    ):
      _ = metric_util.safe_to_scalar(np.array([1, 2]))

  def testPadNoChange(self):
    self.assertAllClose(
        np.array([1.0, 2.0]), metric_util.pad(np.array([1.0, 2.0]), 2, -1.0)
    )

  def testPad1DSingleValue(self):
    self.assertAllClose(
        np.array([1.0, -1.0]), metric_util.pad(np.array([1.0]), 2, -1.0)
    )

  def testPad1DMultipleValues(self):
    self.assertAllClose(
        np.array([1.0, 2.0, -1.0, -1.0]),
        metric_util.pad(np.array([1.0, 2.0]), 4, -1.0),
    )

  def testPad2D(self):
    self.assertAllClose(
        np.array([[1.0, 2.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0, 0.0]]),
        metric_util.pad(np.array([[1.0, 2.0], [3.0, 4.0]]), 5, 0.0),
    )

  def testStandardMetricInputsToNumpy(self):
    example = metric_types.StandardMetricInputs(
        label={'output_name': np.array([2])},
        prediction={'output_name': np.array([0, 0.5, 0.3, 0.9])},
        example_weight={'output_name': np.array([1.0])},
    )
    iterator = metric_util.to_label_prediction_example_weight(
        example, output_name='output_name'
    )

    for expected_label, expected_prediction in zip(
        (0.0, 0.0, 1.0, 0.0), (0.0, 0.5, 0.3, 0.9)
    ):
      got_label, got_pred, got_example_weight = next(iterator)
      self.assertAllClose(got_label, np.array([expected_label]))
      self.assertAllClose(got_pred, np.array([expected_prediction]))
      self.assertAllClose(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsToNumpyWithoutFlatten(self):
    example = metric_types.StandardMetricInputs(
        labels={'output_name': np.array([2])},
        predictions={'output_name': np.array([0, 0.5, 0.3, 0.9])},
        example_weights={'output_name': np.array([1.0])})
    got_label, got_pred, got_example_weight = next(
        metric_util.to_label_prediction_example_weight(
            example, output_name='output_name', flatten=False))

    self.assertAllClose(got_label, np.array([2]))
    self.assertAllClose(got_pred, np.array([0, 0.5, 0.3, 0.9]))
    self.assertAllClose(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsToNumpyWithoutFlattenAndWithSqueeze(self):
    example = metric_types.StandardMetricInputs(
        labels={'output_name': np.array([[2]])},
        predictions={'output_name': np.array([[0, 0.5, 0.3, 0.9]])},
        example_weights={'output_name': np.array([1.0])})
    got_label, got_pred, got_example_weight = next(
        metric_util.to_label_prediction_example_weight(
            example, output_name='output_name', flatten=False))

    self.assertAllClose(got_label, np.array([2]))
    self.assertAllClose(got_pred, np.array([0, 0.5, 0.3, 0.9]))
    self.assertAllClose(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsToNumpyWithoutFlattenAndWithoutSqueeze(self):
    example = metric_types.StandardMetricInputs(
        labels={'output_name': np.array([[2]])},
        predictions={'output_name': np.array([[0, 0.5, 0.3, 0.9]])},
        example_weights={'output_name': np.array([1.0])})
    got_label, got_pred, got_example_weight = next(
        metric_util.to_label_prediction_example_weight(
            example, output_name='output_name', flatten=False, squeeze=False))

    self.assertAllClose(got_label, np.array([[2]]))
    self.assertAllClose(got_pred, np.array([[0, 0.5, 0.3, 0.9]]))
    self.assertAllClose(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsWithZeroWeightsToNumpy(self):
    example = metric_types.StandardMetricInputs(
        labels=np.array([2]),
        predictions=np.array([0, 0.5, 0.3, 0.9]),
        example_weights=np.array([0.0]))
    iterator = metric_util.to_label_prediction_example_weight(
        example, example_weighted=True)

    for expected_label, expected_prediction in zip((0.0, 0.0, 1.0, 0.0),
                                                   (0.0, 0.5, 0.3, 0.9)):
      got_label, got_pred, got_example_weight = next(iterator)
      self.assertAllClose(got_label, np.array([expected_label]))
      self.assertAllClose(got_pred, np.array([expected_prediction]))
      self.assertAllClose(got_example_weight, np.array([0.0]))

  def testStandardMetricInputsWithSparseTensorValue(self):
    example = metric_types.StandardMetricInputs(
        labels=types.SparseTensorValue(
            values=np.array([1]),
            indices=np.array([2]),
            dense_shape=np.array([0, 1])),
        predictions=np.array([0, 0.5, 0.3, 0.9]),
        example_weights=np.array([0.0]))
    iterator = metric_util.to_label_prediction_example_weight(
        example, example_weighted=True)

    for expected_label, expected_prediction in zip((0.0, 0.0, 1.0, 0.0),
                                                   (0.0, 0.5, 0.3, 0.9)):
      got_label, got_pred, got_example_weight = next(iterator)
      self.assertAllClose(got_label, np.array([expected_label]))
      self.assertAllClose(got_pred, np.array([expected_prediction]))
      self.assertAllClose(got_example_weight, np.array([0.0]))

  def testStandardMetricInputsWithZeroWeightsToNumpyWithoutFlatten(self):
    example = metric_types.StandardMetricInputs(
        labels=np.array([2]),
        predictions=np.array([0, 0.5, 0.3, 0.9]),
        example_weights=np.array([0.0]))
    got_label, got_pred, got_example_weight = next(
        metric_util.to_label_prediction_example_weight(
            example, flatten=False, example_weighted=True))

    self.assertAllClose(got_label, np.array([2]))
    self.assertAllClose(got_pred, np.array([0, 0.5, 0.3, 0.9]))
    self.assertAllClose(got_example_weight, np.array([0.0]))

  def testStandardMetricInputsWithClassIDToNumpy(self):
    example = metric_types.StandardMetricInputs(
        labels={'output_name': np.array([2])},
        predictions={'output_name': np.array([0, 0.5, 0.3, 0.9])},
        example_weights={'output_name': np.array([1.0])})
    got_label, got_pred, got_example_weight = next(
        metric_util.to_label_prediction_example_weight(
            example,
            output_name='output_name',
            sub_key=metric_types.SubKey(class_id=2)))

    self.assertAllClose(got_label, np.array([1.0]))
    self.assertAllClose(got_pred, np.array([0.3]))
    self.assertAllClose(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsWithKToNumpy(self):
    example = metric_types.StandardMetricInputs(
        labels={'output_name': np.array([2])},
        predictions={'output_name': np.array([0, 0.5, 0.3, 0.9])},
        example_weights={'output_name': np.array([1.0])})
    got_label, got_pred, got_example_weight = next(
        metric_util.to_label_prediction_example_weight(
            example,
            output_name='output_name',
            sub_key=metric_types.SubKey(k=2)))

    self.assertAllClose(got_label, np.array([0.0]))
    self.assertAllClose(got_pred, np.array([0.5]))
    self.assertAllClose(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsWithKToNumpy2D(self):
    example = metric_types.StandardMetricInputs(
        labels={'output_name': np.array([1, 2])},
        predictions={
            'output_name': np.array([[0, 0.5, 0.3, 0.9], [0.1, 0.4, 0.2, 0.3]])
        },
        example_weights={'output_name': np.array([1.0])})
    got_label, got_pred, got_example_weight = next(
        metric_util.to_label_prediction_example_weight(
            example,
            output_name='output_name',
            sub_key=metric_types.SubKey(k=2),
            flatten=False,
            squeeze=False))

    self.assertAllClose(got_label, np.array([[1], [0]]))
    self.assertAllClose(got_pred, np.array([[0.5], [0.3]]))
    self.assertAllClose(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsWithTopKToNumpy(self):
    example = metric_types.StandardMetricInputs(
        labels={'output_name': np.array([1])},
        predictions={'output_name': np.array([0, 0.5, 0.3, 0.9])},
        example_weights={'output_name': np.array([1.0])})
    iterator = metric_util.to_label_prediction_example_weight(
        example,
        output_name='output_name',
        sub_key=metric_types.SubKey(top_k=2))

    for expected_label, expected_prediction in zip(
        (0.0, 1.0, 0.0, 0.0), (float('-inf'), 0.5, float('-inf'), 0.9)):
      got_label, got_pred, got_example_weight = next(iterator)
      self.assertAllClose(got_label, np.array([expected_label]))
      self.assertAllClose(got_pred, np.array([expected_prediction]))
      self.assertAllClose(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsWithTopKAndClassIdToNumpy(self):
    example = metric_types.StandardMetricInputs(
        labels={'output_name': np.array([1])},
        predictions={'output_name': np.array([0, 0.5, 0.3, 0.9])},
        example_weights={'output_name': np.array([1.0])},
    )
    iterator = metric_util.to_label_prediction_example_weight(
        example,
        output_name='output_name',
        sub_key=metric_types.SubKey(top_k=2, class_id=1),
    )

    expected_label = 1.0
    expected_prediction = 0.5
    got_label, got_pred, got_example_weight = next(iterator)
    self.assertAllClose(got_label, np.array([expected_label]))
    self.assertAllClose(got_pred, np.array([expected_prediction]))
    self.assertAllClose(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsWithTopKAndAggregationTypeToNumpy(self):
    example = metric_types.StandardMetricInputs(
        labels={'output_name': np.array([1])},
        predictions={'output_name': np.array([0, 0.5, 0.3, 0.9])},
        example_weights={'output_name': np.array([1.0])})
    iterator = metric_util.to_label_prediction_example_weight(
        example,
        output_name='output_name',
        sub_key=metric_types.SubKey(top_k=2),
        aggregation_type=metric_types.AggregationType(micro_average=True))

    for expected_label, expected_prediction in zip((1.0, 0.0), (0.5, 0.9)):
      got_label, got_pred, got_example_weight = next(iterator)
      self.assertAllClose(got_label, np.array([expected_label]))
      self.assertAllClose(got_pred, np.array([expected_prediction]))
      self.assertAllClose(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsWithTopKToNumpyWithoutFlatten(self):
    example = metric_types.StandardMetricInputs(
        labels={'output_name': np.array([1, 2])},
        predictions={
            'output_name': np.array([[0, 0.5, 0.3, 0.9], [0.1, 0.4, 0.2, 0.3]])
        },
        example_weights={'output_name': np.array([1.0])})
    got_label, got_pred, got_example_weight = next(
        metric_util.to_label_prediction_example_weight(
            example,
            output_name='output_name',
            sub_key=metric_types.SubKey(top_k=2),
            flatten=False))

    self.assertAllClose(got_label, np.array([1, 2]))
    self.assertAllClose(
        got_pred,
        np.array([[float('-inf'), 0.5, float('-inf'), 0.9],
                  [float('-inf'), 0.4, float('-inf'), 0.3]]))
    self.assertAllClose(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsWithClassWeights(self):
    example = metric_types.StandardMetricInputs(
        labels={'output_name': np.array([2])},
        predictions={'output_name': np.array([0, 0.5, 0.3, 0.9])},
        example_weights={'output_name': np.array([1.0])})
    iterator = metric_util.to_label_prediction_example_weight(
        example,
        output_name='output_name',
        aggregation_type=metric_types.AggregationType(micro_average=True),
        class_weights={
            0: 1.0,
            1: 0.5,
            2: 0.25,
            3: 1.0
        },
        flatten=True)

    for expected_label, expected_prediction, expected_weight in zip(
        (0.0, 0.0, 1.0, 0.0), (0.0, 0.5, 0.3, 0.9), (1.0, 0.5, 0.25, 1.0)):
      got_label, got_pred, got_example_weight = next(iterator)
      self.assertAllClose(got_label, np.array([expected_label]))
      self.assertAllClose(got_pred, np.array([expected_prediction]))
      self.assertAllClose(got_example_weight, np.array([expected_weight]))

  def testStandardMetricInputsWithClassWeightsRaisesErrorWithoutFlatten(self):
    with self.assertRaises(ValueError):
      example = metric_types.StandardMetricInputs(
          labels=np.array([2]),
          predictions=np.array([0, 0.5, 0.3, 0.9]),
          example_weights=np.array([1.0]))
      next(
          metric_util.to_label_prediction_example_weight(
              example, class_weights={
                  1: 0.5,
                  2: 0.25
              }, flatten=False))

  def testStandardMetricInputsWithCustomLabelKeys(self):
    example = metric_types.StandardMetricInputs(
        labels={
            'custom_label': np.array([2]),
            'other_label': np.array([0])
        },
        predictions={'custom_prediction': np.array([0, 0.5, 0.3, 0.9])},
        example_weights=np.array([1.0]))
    eval_config = config_pb2.EvalConfig(model_specs=[
        config_pb2.ModelSpec(
            label_key='custom_label', prediction_key='custom_prediction')
    ])
    iterator = metric_util.to_label_prediction_example_weight(
        example, eval_config=eval_config)

    for expected_label, expected_prediction in zip((0.0, 0.0, 1.0, 0.0),
                                                   (0.0, 0.5, 0.3, 0.9)):
      got_label, got_pred, got_example_weight = next(iterator)
      self.assertAllClose(got_label, np.array([expected_label]), atol=0, rtol=0)
      self.assertAllClose(
          got_pred, np.array([expected_prediction]), atol=0, rtol=0)
      self.assertAllClose(got_example_weight, np.array([1.0]), atol=0, rtol=0)

  def testStandardMetricInputsWithMissingStringLabel(self):
    example = metric_types.StandardMetricInputs(
        label=np.array(['d']),
        prediction={
            'scores': np.array([0.2, 0.7, 0.1]),
            'classes': np.array(['a', 'b', 'c'])
        },
        example_weight=np.array([1.0]))
    iterator = metric_util.to_label_prediction_example_weight(example)

    for expected_label, expected_prediction in zip((0.0, 0.0, 0.0),
                                                   (0.2, 0.7, 0.1)):
      got_label, got_pred, got_example_weight = next(iterator)
      self.assertAllClose(got_label, np.array([expected_label]), atol=0, rtol=0)
      self.assertAllClose(
          got_pred, np.array([expected_prediction]), atol=0, rtol=0)
      self.assertAllClose(got_example_weight, np.array([1.0]), atol=0, rtol=0)

  def testStandardMetricInputsWithoutLabels(self):
    example = metric_types.StandardMetricInputs(
        label={'output_name': np.array([])},
        prediction={'output_name': np.array([0, 0.5, 0.3, 0.9])},
        example_weight={'output_name': np.array([1.0])})
    iterator = metric_util.to_label_prediction_example_weight(
        example, output_name='output_name')

    for expected_prediction in (0.0, 0.5, 0.3, 0.9):
      got_label, got_pred, got_example_weight = next(iterator)
      self.assertAllEqual(got_label, np.array([]))
      self.assertAllClose(got_pred, np.array([expected_prediction]))
      self.assertAllClose(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsWithoutPredictions(self):
    example = metric_types.StandardMetricInputs(
        label={'output_name': np.array([0, 0.5, 0.3, 0.9])},
        prediction={'output_name': np.array([])},
        example_weight={'output_name': np.array([1.0])})
    iterator = metric_util.to_label_prediction_example_weight(
        example, output_name='output_name')

    for expected_label in (0.0, 0.5, 0.3, 0.9):
      got_label, got_pred, got_example_weight = next(iterator)
      self.assertAllClose(got_label, np.array([expected_label]))
      self.assertAllEqual(got_pred, np.array([]))
      self.assertAllClose(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsWithMultipleOutputs(self):
    example = metric_types.StandardMetricInputs(
        label={
            'output1': np.array([0, 1]),
            'output2': np.array([1, 1])
        },
        prediction={
            'output1': np.array([0, 0.5]),
            'output2': np.array([0.2, 0.8])
        },
        example_weight={
            'output1': np.array([0.5]),
            'output2': np.array([1.0])
        })

    for output in ('output1', 'output2'):
      iterator = metric_util.to_label_prediction_example_weight(
          example, output_name=output, flatten=False, example_weighted=True)
      got_label, got_pred, got_example_weight = next(iterator)
      self.assertAllClose(got_label, example.label[output])
      self.assertAllEqual(got_pred, example.prediction[output])
      self.assertAllClose(got_example_weight, example.example_weight[output])

  def testStandardMetricInputsWithMultipleOutputsNotExampleWeighted(self):
    example = metric_types.StandardMetricInputs(
        label={
            'output1': np.array([0, 1]),
            'output2': np.array([1, 1])
        },
        prediction={
            'output1': np.array([0, 0.5]),
            'output2': np.array([0.2, 0.8])
        },
        example_weight={
            'output1': np.array([0.5]),
            'output2': np.array([1.0])
        })

    for output in ('output1', 'output2'):
      iterator = metric_util.to_label_prediction_example_weight(
          example, output_name=output, flatten=False, example_weighted=False)
      got_label, got_pred, got_example_weight = next(iterator)
      self.assertAllClose(got_label, example.label[output])
      self.assertAllEqual(got_pred, example.prediction[output])
      self.assertAllClose(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsWithMissingLabelsAndExampleWeights(self):
    example = metric_types.StandardMetricInputs(prediction={
        'output1': np.array([0, 0.5]),
        'output2': np.array([0.2, 0.8])
    })

    for output in ('output1', 'output2'):
      iterator = metric_util.to_label_prediction_example_weight(
          example, output_name=output, flatten=False, allow_none=True)
      got_label, got_pred, got_example_weight = next(iterator)
      self.assertAllEqual(got_label, np.array([]))
      self.assertAllEqual(got_pred, example.prediction[output])
      self.assertAllEqual(got_example_weight, np.array([1.0]))

  def testStandardMetricInputsWithMissingLabelKeyRaisesError(self):
    example = metric_types.StandardMetricInputs(
        label={'output2': np.array([1, 1])},
        prediction={
            'output1': np.array([0.5]),
            'output2': np.array([0.8])
        },
        example_weight={
            'output1': np.array([0.5]),
            'output2': np.array([1.0])
        })
    with self.assertRaisesRegex(
        ValueError, 'unable to prepare label for metric computation.*'):
      next(
          metric_util.to_label_prediction_example_weight(
              example, output_name='output1'))

  def testStandardMetricInputsWithMissingPredictionRaisesError(self):
    example = metric_types.StandardMetricInputs(
        label={
            'output1': np.array([0, 1]),
            'output2': np.array([1, 1])
        },
        prediction={'output2': np.array([0.8])},
        example_weight={
            'output1': np.array([0.5]),
            'output2': np.array([1.0])
        })
    with self.assertRaisesRegex(ValueError, '"output1" key not found.*'):
      next(
          metric_util.to_label_prediction_example_weight(
              example, output_name='output1'))

  def testStandardMetricInputsWithMissingExampleWeightKeyRaisesError(self):
    example = metric_types.StandardMetricInputs(
        label={
            'output1': np.array([0, 1]),
            'output2': np.array([1, 1])
        },
        prediction={
            'output1': np.array([0.5]),
            'output2': np.array([0.8])
        },
        example_weight={'output2': np.array([1.0])})
    with self.assertRaisesRegex(
        ValueError,
        'unable to prepare example_weight for metric computation.*'):
      next(
          metric_util.to_label_prediction_example_weight(
              example, output_name='output1', example_weighted=True))

  def testStandardMetricInputsWithNonScalarWeights(self):
    example = metric_types.StandardMetricInputs(
        label={'output_name': np.array([2])},
        prediction={'output_name': np.array([0, 0.5, 0.3, 0.9])},
        example_weight={'output_name': np.array([1.0, 0.0, 1.0, 1.0])})
    iterable = metric_util.to_label_prediction_example_weight(
        example,
        output_name='output_name',
        example_weighted=True,
        require_single_example_weight=False)

    for expected_label, expected_prediction, expected_weight in zip(
        (0.0, 0.0, 1.0, 0.0), (0.0, 0.5, 0.3, 0.9), (1.0, 0.0, 1.0, 1.0)):
      got_label, got_pred, got_example_weight = next(iterable)
      self.assertAllClose(got_label, np.array([expected_label]))
      self.assertAllEqual(got_pred, np.array([expected_prediction]))
      self.assertAllClose(got_example_weight, np.array([expected_weight]))

  def testStandardMetricInputsWithNonScalarWeightsNoFlatten(self):
    example = metric_types.StandardMetricInputs(
        label=np.array([2]),
        prediction=np.array([0, 0.5, 0.3, 0.9]),
        example_weight=np.array([1.0, 0.0, 1.0, 1.0]))
    got_label, got_pred, got_example_weight = next(
        metric_util.to_label_prediction_example_weight(
            example,
            flatten=False,
            example_weighted=True,
            require_single_example_weight=False))
    self.assertAllClose(got_label, np.array([2]))
    self.assertAllEqual(got_pred, np.array([0, 0.5, 0.3, 0.9]))
    self.assertAllClose(got_example_weight, np.array([1.0, 0.0, 1.0, 1.0]))

  def testStandardMetricInputsWithMismatchedExampleWeightsRaisesError(self):
    with self.assertRaises(ValueError):
      example = metric_types.StandardMetricInputs(
          labels=np.array([2]),
          predictions=np.array([0, 0.5, 0.3, 0.9]),
          example_weights=np.array([1.0, 0.0]))
      next(
          metric_util.to_label_prediction_example_weight(
              example,
              flatten=True,
              example_weighted=True,
              require_single_example_weight=False))

  def testStandardMetricInputsRequiringSingleExampleWeightRaisesError(self):
    with self.assertRaises(ValueError):
      example = metric_types.StandardMetricInputs(
          labels=np.array([2]),
          predictions=np.array([0, 0.5, 0.3, 0.9]),
          example_weights=np.array([1.0, 0.0]))
      next(
          metric_util.to_label_prediction_example_weight(
              example,
              example_weighted=True,
              require_single_example_weight=True))

  def testPrepareLabelsAndPredictions(self):
    labels = [0]
    preds = {
        'logistic': np.array([0.8]),
    }
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([0]))
    self.assertAllClose(got_preds, np.array([0.8]))

  def testPrepareLabelsAndPredictionsClassNotFound(self):
    labels = ['d']
    preds = {
        'scores': np.array([0.2, 0.7, 0.1]),
        'all_classes': np.array(['a', 'b', 'c'])
    }
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([0, 0, 0]))
    self.assertAllClose(got_preds, np.array([0.2, 0.7, 0.1]))

  def testPrepareLabelsAndPredictionsBatched(self):
    labels = [['b']]
    preds = {
        'logistic': np.array([[0.8]]),
        'all_classes': np.array([['a', 'b', 'c']])
    }
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([[1]]))
    self.assertAllClose(got_preds, np.array([[0.8]]))

  def testPrepareLabelsAndPredictionsMixedBatching(self):
    labels = np.array([1])
    preds = {
        'predictions': np.array([[0.8]]),
    }
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([1]))
    self.assertAllClose(got_preds, np.array([[0.8]]))

  def testPrepareMultipleLabelsAndPredictions(self):
    labels = np.array(['b', 'c', 'a'])
    preds = {
        'scores': np.array([0.2, 0.7, 0.1]),
        'classes': np.array(['a', 'b', 'c'])
    }
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([1, 2, 0]))
    self.assertAllClose(got_preds, np.array([0.2, 0.7, 0.1]))

  def testPrepareMultipleLabelsAndPredictionsPythonList(self):
    labels = ['b', 'c', 'a']
    preds = {'probabilities': [0.2, 0.7, 0.1], 'all_classes': ['a', 'b', 'c']}
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([1, 2, 0]))
    self.assertAllClose(got_preds, np.array([0.2, 0.7, 0.1]))

  def testPrepareLabelsAndPredictionsSparseTensorValue(self):
    labels = types.SparseTensorValue(
        indices=np.array([1, 2]),
        values=np.array([1, 1]),
        dense_shape=np.array([1, 2]))
    preds = {'probabilities': [0.2, 0.7, 0.1], 'all_classes': ['a', 'b', 'c']}
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([0, 1, 1]))
    self.assertAllClose(got_preds, np.array([0.2, 0.7, 0.1]))

  def testPrepareLabelsAndPredictionsEmptySparseTensorValue(self):
    labels = types.SparseTensorValue(
        values=np.array([]), indices=np.array([]), dense_shape=np.array([0, 2]))
    preds = {'probabilities': [0.2, 0.7, 0.1], 'all_classes': ['a', 'b', 'c']}
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([0, 0, 0]))
    self.assertAllClose(got_preds, np.array([0.2, 0.7, 0.1]))

  def testPrepareLabelsAndPredictionsSparseTensorValueWithBatching(self):
    labels = types.SparseTensorValue(
        indices=np.array([1, 2]),
        values=np.array([1, 1]),
        dense_shape=np.array([1, 2]))
    preds = {
        'probabilities': [[0.2, 0.7, 0.1]],
        'all_classes': [['a', 'b', 'c']]
    }
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([[0, 1, 1]]))
    self.assertAllClose(got_preds, np.array([[0.2, 0.7, 0.1]]))

  def testPrepareMultipleLabelsAndPredictionsMultiDimension(self):
    labels = [[0], [1]]
    preds = {'probabilities': [[0.2, 0.8], [0.3, 0.7]]}
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([[0], [1]]))
    self.assertAllClose(got_preds, np.array([[0.2, 0.8], [0.3, 0.7]]))

  def testPrepareLabelsAndPredictionsEmpty(self):
    labels = []
    preds = {'logistic': [], 'all_classes': ['a', 'b', 'c']}
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([]))
    self.assertAllClose(got_preds, np.array([]))

  def testPrepareLabelsAndPredictionsWithVocab(self):
    labels = np.array(['e', 'f'])
    preds = {'probabilities': [0.2, 0.8], 'all_classes': ['a', 'b', 'c']}
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds, label_vocabulary=['e', 'f'])

    self.assertAllClose(got_labels, np.array([0, 1]))
    self.assertAllClose(got_preds, np.array([0.2, 0.8]))

  def testPrepareLabelsAndPredictionsWithVocabUsingObjectType(self):
    labels = np.array(['e', 'f'], dtype=object)
    preds = {'probabilities': [0.2, 0.8], 'all_classes': ['a', 'b', 'c']}
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds, label_vocabulary=['e', 'f'])

    self.assertAllClose(got_labels, np.array([0, 1]))
    self.assertAllClose(got_preds, np.array([0.2, 0.8]))

  def testPrepareLabelsAndPredictionsSparseTensorValueAndVocab(self):
    labels = types.SparseTensorValue(
        indices=np.array([0, 2]),
        values=np.array(['c', 'a']),
        dense_shape=np.array([1, 2]))
    preds = {'probabilities': [0.2, 0.7, 0.1], 'all_classes': ['a', 'b', 'c']}
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([1, 0, 1]))
    self.assertAllClose(got_preds, np.array([0.2, 0.7, 0.1]))

  def testPrepareLabelsAndPredictionsUsingBinaryScores(self):
    labels = np.array([[0], [1]])
    preds = {
        'scores': np.array([[0.9, 0.2], [0.3, 0.7]]),
        'classes': np.array([['a', 'b'], ['a', 'b']])
    }
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([[0], [1]]))
    self.assertAllClose(got_preds, np.array([[0.9, 0.2], [0.3, 0.7]]))

  def testPrepareLabelsAndPredictionsUsingBinaryScoresSparse(self):
    labels = np.array([1, 0])
    preds = {
        'scores': np.array([[0.9, 0.2], [0.3, 0.7]]),
        'classes': np.array([['a', 'b'], ['a', 'b']])
    }
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([1, 0]))
    self.assertAllClose(got_preds, np.array([[0.9, 0.2], [0.3, 0.7]]))

  def testPrepareLabelsAndPredictionsUsingBinaryScoresUnbatched(self):
    labels = np.array([1])
    preds = {'scores': np.array([0.3, 0.7]), 'classes': np.array(['a', 'b'])}
    got_labels, got_preds = metric_util.prepare_labels_and_predictions(
        labels, preds)

    self.assertAllClose(got_labels, np.array([1]))
    self.assertAllClose(got_preds, np.array([0.3, 0.7]))

  def testSelectClassIDSparse(self):
    labels = np.array([2])
    preds = np.array([0.2, 0.7, 0.1])
    got_labels, got_preds = metric_util.select_class_id(1, labels, preds)

    self.assertAllClose(got_labels, np.array([0]))
    self.assertAllClose(got_preds, np.array([0.7]))

  def testSelectClassIDSparseNoShape(self):
    labels = np.array(2)
    preds = np.array([0.2, 0.7, 0.1])
    got_labels, got_preds = metric_util.select_class_id(1, labels, preds)

    self.assertAllClose(got_labels, np.array([0]))
    self.assertAllClose(got_preds, np.array([0.7]))

  def testSelectClassIDSparseWithMultipleValues(self):
    labels = np.array([0, 2, 1])
    preds = np.array([[0.2, 0.7, 0.1], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7]])
    got_labels, got_preds = metric_util.select_class_id(1, labels, preds)

    self.assertAllClose(got_labels, np.array([[0], [0], [1]]))
    self.assertAllClose(got_preds, np.array([[0.7], [0.6], [0.2]]))

  def testSelectClassIDSparseBatched(self):
    labels = np.array([[0], [2], [1]])
    preds = np.array([[0.2, 0.7, 0.1], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7]])
    got_labels, got_preds = metric_util.select_class_id(1, labels, preds)

    self.assertAllClose(got_labels, np.array([[0], [0], [1]]))
    self.assertAllClose(got_preds, np.array([[0.7], [0.6], [0.2]]))

  def testSelectClassIDSparseMultiDim(self):
    labels = np.array([[[0]], [[2]], [[1]]])
    preds = np.array([[[0.2, 0.7, 0.1]], [[0.3, 0.6, 0.1]], [[0.1, 0.2, 0.7]]])
    got_labels, got_preds = metric_util.select_class_id(1, labels, preds)

    self.assertAllClose(got_labels, np.array([[[0]], [[0]], [[1]]]))
    self.assertAllClose(got_preds, np.array([[[0.7]], [[0.6]], [[0.2]]]))

  def testRaisesErrorForInvalidSparseSettings(self):
    with self.assertRaises(ValueError):
      labels = np.array([[0, 0, 1]])
      preds = np.array([[0.2, 0.7, 0.1]])
      metric_util.select_class_id(1, labels, preds, sparse_labels=True)

  def testSelectClassID(self):
    labels = np.array([0, 0, 1])
    preds = np.array([0.2, 0.7, 0.1])
    got_labels, got_preds = metric_util.select_class_id(1, labels, preds)

    self.assertAllClose(got_labels, np.array([0]))
    self.assertAllClose(got_preds, np.array([0.7]))

  def testSelectClassIDWithMultipleValues(self):
    labels = np.array([[0, 0, 1], [0, 0, 1], [0, 1, 0]])
    preds = np.array([[0.2, 0.7, 0.1], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7]])
    got_labels, got_preds = metric_util.select_class_id(1, labels, preds)

    self.assertAllClose(got_labels, np.array([[0], [0], [1]]))
    self.assertAllClose(got_preds, np.array([[0.7], [0.6], [0.2]]))

  def testSelectClassIDBatched(self):
    labels = np.array([[0, 0, 1]])
    preds = np.array([[0.2, 0.7, 0.1]])
    got_labels, got_preds = metric_util.select_class_id(1, labels, preds)

    self.assertAllClose(got_labels, np.array([[0]]))
    self.assertAllClose(got_preds, np.array([[0.7]]))

  def testSelectClassIDMultiDim(self):
    labels = np.array([[[0, 0, 1]]])
    preds = np.array([[[0.2, 0.7, 0.1]]])
    got_labels, got_preds = metric_util.select_class_id(1, labels, preds)

    self.assertAllClose(got_labels, np.array([[[0]]]))
    self.assertAllClose(got_preds, np.array([[[0.7]]]))

  def testRaisesErrorForInvalidNonSparseSettings(self):
    with self.assertRaises(ValueError):
      labels = np.array([5])
      preds = np.array([0.2, 0.7, 0.1])
      metric_util.select_class_id(1, labels, preds, sparse_labels=False)

  def testSelectClassIDEmpty(self):
    labels = np.array(np.array([]))
    preds = np.array(np.array([]))
    got_labels, got_preds = metric_util.select_class_id(1, labels, preds)

    self.assertAllClose(got_labels, np.array([]))
    self.assertAllClose(got_preds, np.array([]))

  def testTopKIndices(self):
    scores = np.array([0.4, 0.1, 0.2, 0.3])
    got = metric_util.top_k_indices(2, scores)
    # Indices could be in any order, test by overwritting the original scores
    scores[got] = -1.0
    self.assertAllClose(scores, np.array([-1.0, 0.1, 0.2, -1.0]))

  def testTopKIndicesSorted(self):
    scores = np.array([0.1, 0.3, 0.4, 0.2])
    got = metric_util.top_k_indices(2, scores, sort=True)
    self.assertAllClose(got, np.array([2, 1]))
    self.assertAllClose(scores[got], np.array([0.4, 0.3]))

  def testTopKIndices2D(self):
    scores = np.array([[0.4, 0.1, 0.2, 0.3], [0.1, 0.2, 0.1, 0.6]])
    got = metric_util.top_k_indices(2, scores)
    scores[got] = -1.0
    self.assertAllClose(
        scores, np.array([[-1.0, 0.1, 0.2, -1.0], [0.1, -1.0, 0.1, -1.0]]))

  def testTopKIndices2DSorted(self):
    scores = np.array([[0.3, 0.1, 0.4, 0.2], [0.1, 0.2, 0.3, 0.6]])
    got = metric_util.top_k_indices(2, scores, sort=True)
    # Indices are in ([row_index,...], [col_index, ...]) format.
    self.assertAllClose(got, (np.array([0, 0, 1, 1]), np.array([2, 0, 3, 2])))
    self.assertAllClose(scores[got], np.array([0.4, 0.3, 0.6, 0.3]))

  def testTopKIndicesWithBinaryClassification(self):
    scores = np.array([0.2, 0.8])
    got = metric_util.top_k_indices(1, scores)
    self.assertAllClose(got, np.array([1]))
    self.assertAllClose(scores[got], np.array([0.8]))


