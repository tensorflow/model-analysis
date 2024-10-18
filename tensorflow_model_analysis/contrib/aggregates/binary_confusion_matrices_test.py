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
"""Tests for binary confusion matrix."""

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_model_analysis.contrib.aggregates import binary_confusion_matrices
from tensorflow_model_analysis.utils import test_util


class BinaryConfusionMatricesTest(
    test_util.TensorflowModelAnalysisTest, parameterized.TestCase
):

  @parameterized.parameters(
      dict(
          thresholds=[0, 0.5, 1],
          example_ids_count=100,
          example_weights=(None, None, None, None),
          example_ids=(None, None, None, None),
          expected_result={
              0: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=2.0, tn=1.0, fp=1.0, fn=0.0
                  ),
                  tp_examples=[],
                  tn_examples=[],
                  fp_examples=[],
                  fn_examples=[],
              ),
              0.5: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=1.0, tn=2.0, fp=0.0, fn=1.0
                  ),
                  tp_examples=[],
                  tn_examples=[],
                  fp_examples=[],
                  fn_examples=[],
              ),
              1: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=0.0, tn=2.0, fp=0.0, fn=2.0
                  ),
                  tp_examples=[],
                  tn_examples=[],
                  fp_examples=[],
                  fn_examples=[],
              ),
          },
      ),
      dict(
          thresholds=[0.1, 0.9],
          example_ids_count=1,
          example_weights=(1, 1, 1, 1),
          example_ids=('id_1', 'id_2', 'id_3', 'id_4'),
          expected_result={
              0.1: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=2.0, tn=1.0, fp=1.0, fn=0.0
                  ),
                  tp_examples=['id_3'],
                  tn_examples=['id_1'],
                  fp_examples=['id_2'],
                  fn_examples=[],
              ),
              0.9: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=0.0, tn=2.0, fp=0.0, fn=2.0
                  ),
                  tp_examples=[],
                  tn_examples=['id_1'],
                  fp_examples=[],
                  fn_examples=['id_3'],
              ),
          },
      ),
      dict(
          thresholds=[0.1, 0.9],
          example_ids_count=2,
          example_weights=(1, 1, 1, 1),
          example_ids=('id_1', 'id_2', 'id_3', 'id_4'),
          expected_result={
              0.1: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=2.0, tn=1.0, fp=1.0, fn=0.0
                  ),
                  tp_examples=['id_3', 'id_4'],
                  tn_examples=['id_1'],
                  fp_examples=['id_2'],
                  fn_examples=[],
              ),
              0.9: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=0.0, tn=2.0, fp=0.0, fn=2.0
                  ),
                  tp_examples=[],
                  tn_examples=['id_1', 'id_2'],
                  fp_examples=[],
                  fn_examples=['id_3', 'id_4'],
              ),
          },
      ),
      dict(
          thresholds=[0.25, 0.75],
          example_ids_count=100,
          example_weights=(0.2, 0.3, 0.5, 0.7),
          example_ids=(None, None, None, None),
          expected_result={
              0.25: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=1.2, tn=0.2, fp=0.3, fn=0.0
                  ),
                  tp_examples=[],
                  tn_examples=[],
                  fp_examples=[],
                  fn_examples=[],
              ),
              0.75: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=0.7, tn=0.5, fp=0.0, fn=0.5
                  ),
                  tp_examples=[],
                  tn_examples=[],
                  fp_examples=[],
                  fn_examples=[],
              ),
          },
      ),
  )
  def testBinaryConfusionMatricesPerRow(
      self,
      thresholds,
      example_ids_count,
      example_weights,
      example_ids,
      expected_result,
  ):
    labels = (0, 0, 1, 1)
    predictions = (0, 0.5, 0.3, 0.9)

    confusion_matrix = binary_confusion_matrices.BinaryConfusionMatrices(
        thresholds=thresholds,
        example_ids_count=example_ids_count,
    )
    accumulator = confusion_matrix.create_accumulator()
    for label, prediction, example_weight, example_id in zip(
        labels, predictions, example_weights, example_ids
    ):
      accumulator = confusion_matrix.add_input(
          accumulator=accumulator,
          labels=[label],
          predictions=[prediction],
          example_weights=[example_weight] if example_weight else None,
          example_id=example_id,
      )
    self.assertDictEqual(accumulator, expected_result)

  @parameterized.parameters(
      dict(
          thresholds=[0, 0.5, 1],
          example_ids_count=100,
          example_weights=(None, None, None, None),
          example_ids=(None, None, None, None),
          expected_result={
              0: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=2.0, tn=1.0, fp=1.0, fn=0.0
                  ),
                  tp_examples=[],
                  tn_examples=[],
                  fp_examples=[],
                  fn_examples=[],
              ),
              0.5: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=1.0, tn=2.0, fp=0.0, fn=1.0
                  ),
                  tp_examples=[],
                  tn_examples=[],
                  fp_examples=[],
                  fn_examples=[],
              ),
              1: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=0.0, tn=2.0, fp=0.0, fn=2.0
                  ),
                  tp_examples=[],
                  tn_examples=[],
                  fp_examples=[],
                  fn_examples=[],
              ),
          },
      ),
      dict(
          thresholds=[0.1, 0.9],
          example_ids_count=1,
          example_weights=(1, 1, 1, 1),
          example_ids=('id_1', 'id_2', 'id_3', 'id_4'),
          expected_result={
              0.1: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=2.0, tn=1.0, fp=1.0, fn=0.0
                  ),
                  tp_examples=['id_3'],
                  tn_examples=['id_1'],
                  fp_examples=['id_2'],
                  fn_examples=[],
              ),
              0.9: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=0.0, tn=2.0, fp=0.0, fn=2.0
                  ),
                  tp_examples=[],
                  tn_examples=['id_1'],
                  fp_examples=[],
                  fn_examples=['id_3'],
              ),
          },
      ),
      dict(
          thresholds=[0.1, 0.9],
          example_ids_count=2,
          example_weights=(1, 1, 1, 1),
          example_ids=('id_1', 'id_2', 'id_3', 'id_4'),
          expected_result={
              0.1: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=2.0, tn=1.0, fp=1.0, fn=0.0
                  ),
                  tp_examples=['id_3', 'id_4'],
                  tn_examples=['id_1'],
                  fp_examples=['id_2'],
                  fn_examples=[],
              ),
              0.9: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=0.0, tn=2.0, fp=0.0, fn=2.0
                  ),
                  tp_examples=[],
                  tn_examples=['id_1', 'id_2'],
                  fp_examples=[],
                  fn_examples=['id_3', 'id_4'],
              ),
          },
      ),
      dict(
          thresholds=[0.25, 0.75],
          example_ids_count=100,
          example_weights=(0.2, 0.3, 0.5, 0.7),
          example_ids=(None, None, None, None),
          expected_result={
              0.25: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=1.2, tn=0.2, fp=0.3, fn=0.0
                  ),
                  tp_examples=[],
                  tn_examples=[],
                  fp_examples=[],
                  fn_examples=[],
              ),
              0.75: binary_confusion_matrices._ThresholdEntry(
                  matrix=binary_confusion_matrices.Matrix(
                      tp=0.7, tn=0.5, fp=0.0, fn=0.5
                  ),
                  tp_examples=[],
                  tn_examples=[],
                  fp_examples=[],
                  fn_examples=[],
              ),
          },
      ),
  )
  def testBinaryConfusionMatricesInProcess(
      self,
      thresholds,
      example_ids_count,
      example_weights,
      example_ids,
      expected_result,
  ):
    labels = (0, 0, 1, 1)
    predictions = (0, 0.5, 0.3, 0.9)

    confusion_matrix = binary_confusion_matrices.BinaryConfusionMatrices(
        thresholds=thresholds,
        example_ids_count=example_ids_count,
    )
    actual = confusion_matrix(labels, predictions, example_weights, example_ids)
    self.assertDictEqual(actual, expected_result)


