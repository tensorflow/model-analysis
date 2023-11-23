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
from absl.testing import absltest
from absl.testing import parameterized
from tensorflow_model_analysis.experimental.metrics.aggregates import _classification


class ClassificationTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="binary_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          average="binary",
          expected=_classification.ConfusionMatrixAcc(
              tp=2, tn=2, fp=1, fn=2, config=dict(average="binary")
          ),
      ),
      dict(
          testcase_name="binary_micro_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          average="micro",
          expected=_classification.ConfusionMatrixAcc(
              tp=4, tn=4, fp=3, fn=3, config=dict(average="micro")
          ),
      ),
      dict(
          testcase_name="binary_macro_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          average="macro",
          expected=_classification.ConfusionMatrixAcc(
              tp=[2, 2],
              tn=[2, 2],
              fp=[1, 2],
              fn=[2, 1],
              config=dict(average="macro"),
          ),
      ),
      dict(
          testcase_name="micro_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          average="micro",
          expected=_classification.ConfusionMatrixAcc(
              tp=4, tn=4, fp=3, fn=3, config=dict(average="micro")
          ),
      ),
      dict(
          testcase_name="macro_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          average="macro",
          expected=_classification.ConfusionMatrixAcc(
              tp=[2, 2],
              tn=[2, 2],
              fp=[1, 2],
              fn=[2, 1],
              config=dict(average="macro"),
          ),
      ),
      dict(
          testcase_name="samples_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          average="samples",
          expected=_classification.ConfusionMatrixAcc(
              tp=[1, 0, 0, 1, 1, 1, 0],
              tn=[1, 0, 0, 1, 1, 1, 0],
              fp=[0, 1, 1, 0, 0, 0, 1],
              fn=[0, 1, 1, 0, 0, 0, 1],
              config=dict(average="samples"),
          ),
      ),
  ])
  def test_confusion_matrix_binary_encoding(
      self, y_true, y_pred, input_type, average, expected
  ):
    confusion_matrix = _classification.ConfusionMatrix()
    self.assertEqual(
        expected,
        confusion_matrix(
            y_true, y_pred, input_type=input_type, average=average
        ),
    )

  def test_confusion_matrix_merge(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = _classification.ConfusionMatrix()
    baseline = _classification.ConfusionMatrixAcc(
        tp=1, tn=1, fp=1, fn=1, config=dict(average="binary")
    )
    acc = confusion_matrix.add_inputs(baseline, y_true, y_pred)
    acc_other = confusion_matrix.add_inputs(None, y_true, y_pred)
    actual = confusion_matrix.merge_accumulators([acc, acc_other])
    expected = _classification.ConfusionMatrixAcc(
        tp=5, tn=5, fp=3, fn=5, config=dict(average="binary")
    )
    self.assertEqual(expected, actual)

  def test_confusion_matrix_add(self):
    a = _classification.ConfusionMatrixAcc(tp=4, tn=4, fp=2, fn=4)
    b = _classification.ConfusionMatrixAcc(tp=1, tn=6, fp=9, fn=2)
    expected = _classification.ConfusionMatrixAcc(tp=5, tn=10, fp=11, fn=6)
    self.assertEqual(a + b, expected)

  def test_confusion_matrix_invalid_config(self):
    a = _classification.ConfusionMatrixAcc(
        tp=4, tn=4, fp=2, fn=4, config=dict(average="micro")
    )
    b = _classification.ConfusionMatrixAcc(
        tp=1, tn=6, fp=9, fn=2, config=dict(average="macro")
    )
    with self.assertRaisesRegex(ValueError, "Incompatile config: "):
      a + b  # pylint: disable=pointless-statement

  def test_confusion_matrix_trues_positives(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = _classification.ConfusionMatrix()
    cm = confusion_matrix(y_true, y_pred)
    self.assertEqual(3, cm.p)
    self.assertEqual(4, cm.t)

  def test_confusion_matrix_invalid_input_type(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = _classification.ConfusionMatrix()
    with self.assertRaisesRegex(NotImplementedError, "is not supported"):
      confusion_matrix(y_true, y_pred, input_type="invalid")

  def test_confusion_matrix_invalid_average_type(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = _classification.ConfusionMatrix()
    with self.assertRaisesRegex(NotImplementedError, " is not supported"):
      confusion_matrix(y_true, y_pred, average="invalid")

  def test_confusion_matrix_unknown_tn(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = _classification.ConfusionMatrix(unknown_tn=True)
    actual = confusion_matrix(y_true, y_pred)
    expected = _classification.ConfusionMatrixAcc(
        tp=2, tn=float("inf"), fp=1, fn=2, config=dict(average="binary")
    )
    self.assertEqual(expected, actual)


if __name__ == "__main__":
  absltest.main()
