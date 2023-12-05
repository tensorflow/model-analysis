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
import numpy as np
from tensorflow_model_analysis.experimental.metrics.aggregates import _classification


InputType = _classification.InputType
AverageType = _classification.AverageType
ConfusionMatrixMetric = _classification.ConfusionMatrixMetric


class ClassificationTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="binary_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          average="binary",
          expected=_classification._ConfusionMatrix(tp=2, tn=2, fp=1, fn=2),
      ),
      dict(
          testcase_name="multiclass_indicator_binary_average",
          y_pred=[[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1]],
          y_true=[[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0]],
          input_type="multiclass-indicator",
          average="binary",
          expected=_classification._ConfusionMatrix(tp=2, tn=2, fp=1, fn=2),
      ),
      dict(
          testcase_name="binary_string_binary_average",
          y_pred=["Y", "N", "Y", "N", "Y", "N", "N"],
          y_true=["Y", "Y", "N", "N", "Y", "N", "Y"],
          input_type="binary",
          pos_label="Y",
          average="binary",
          expected=_classification._ConfusionMatrix(tp=2, tn=2, fp=1, fn=2),
      ),
      dict(
          testcase_name="binary_micro_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          average="micro",
          expected=_classification._ConfusionMatrix(tp=4, tn=4, fp=3, fn=3),
      ),
      dict(
          testcase_name="multiclass_indicator_micro_average",
          y_pred=[[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1]],
          y_true=[[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0]],
          input_type="multiclass-indicator",
          average="micro",
          expected=_classification._ConfusionMatrix(tp=4, tn=4, fp=3, fn=3),
      ),
      dict(
          testcase_name="multiclass_micro_average",
          y_pred=["y", "n", "y", "n", "y", "n", "n", "u"],
          y_true=["y", "y", "n", "n", "y", "n", "y", "u"],
          input_type="multiclass",
          average="micro",
          expected=_classification._ConfusionMatrix(tp=5, tn=13, fp=3, fn=3),
      ),
      dict(
          testcase_name="multiclass_multioutput_micro_average",
          y_pred=[["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]],
          y_true=[["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]],
          input_type="multiclass-multioutput",
          average="micro",
          expected=_classification._ConfusionMatrix(tp=6, tn=12, fp=3, fn=3),
      ),
      dict(
          testcase_name="binary_macro_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          average="macro",
          expected=_classification._ConfusionMatrix(
              tp=[2, 2],
              tn=[2, 2],
              fp=[1, 2],
              fn=[2, 1],
          ),
      ),
      dict(
          testcase_name="multiclass_indicator_macro_average",
          y_pred=[[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1]],
          y_true=[[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0]],
          input_type="multiclass-indicator",
          average="macro",
          expected=_classification._ConfusionMatrix(
              tp=[2, 2],
              tn=[2, 2],
              fp=[1, 2],
              fn=[2, 1],
          ),
      ),
      dict(
          testcase_name="multiclass_macro_average",
          y_pred=["y", "n", "y", "n", "y", "n", "n", "u"],
          y_true=["y", "y", "n", "n", "y", "n", "y", "u"],
          input_type="multiclass",
          # A vocab is required to align class index for macro.
          vocab={"y": 0, "n": 1, "u": 2},
          average="macro",
          expected=_classification._ConfusionMatrix(
              tp=[2, 2, 1],
              tn=[3, 3, 7],
              fp=[1, 2, 0],
              fn=[2, 1, 0],
          ),
      ),
      dict(
          testcase_name="multiclass_multioutput_macro_average",
          y_pred=[["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]],
          y_true=[["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]],
          input_type="multiclass-multioutput",
          # A vocab is required to align class index for macro.
          vocab={"y": 0, "n": 1, "u": 2},
          average="macro",
          expected=_classification._ConfusionMatrix(
              tp=[3, 2, 1],
              tn=[3, 2, 7],
              fp=[1, 2, 0],
              fn=[1, 2, 0],
          ),
      ),
  ])
  def test_confusion_matrix_metrics_callable(
      self,
      y_true,
      y_pred,
      input_type,
      average,
      expected,
      vocab=None,
      pos_label=1,
  ):
    confusion_matrix = _classification.ConfusionMatrixAggFn(
        input_type=input_type,
        average=average,
        vocab=vocab,
        pos_label=pos_label,
    )
    self.assertEqual((expected,), confusion_matrix(y_true, y_pred))

  @parameterized.named_parameters([
      dict(
          testcase_name="binary_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          average="binary",
          metrics=["precision", "recall", "f1_score", "accuracy"],
          expected=(2 / 3, 2 / 4, 4 / 7, 1),
      ),
      dict(
          testcase_name="macro_average",
          y_pred=[1, 0, 1, 0, 1, 0, 0],
          y_true=[1, 1, 0, 0, 1, 0, 1],
          input_type="binary",
          average="macro",
          metrics=["precision", "recall", "f1_score", "accuracy"],
          expected=((2 / 3 + 2 / 4) / 2, (2 / 3 + 2 / 4) / 2, 4 / 7, 1),
      ),
      dict(
          testcase_name="multiclass_indicator_binary_average",
          y_pred=[[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1]],
          y_true=[[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0]],
          input_type="multiclass-indicator",
          average="binary",
          metrics=["precision", "recall", "f1_score", "accuracy"],
          expected=(2 / 3, 2 / 4, 4 / 7, 1),
      ),
      dict(
          testcase_name="binary_string_binary_average",
          y_pred=["Y", "N", "Y", "N", "Y", "N", "N"],
          y_true=["Y", "Y", "N", "N", "Y", "N", "Y"],
          input_type="binary",
          pos_label="Y",
          average="binary",
          metrics=["precision", "recall", "f1_score", "accuracy"],
          expected=(2 / 3, 2 / 4, 4 / 7, 1),
      ),
  ])
  def test_confusion_matrix_metric(
      self,
      y_true,
      y_pred,
      input_type,
      average,
      expected,
      metrics,
      vocab=None,
      pos_label=1,
  ):
    confusion_matrix = _classification.ConfusionMatrixAggFn(
        input_type=input_type,
        average=average,
        vocab=vocab,
        pos_label=pos_label,
        metrics=metrics,
    )
    np.testing.assert_allclose(
        expected, confusion_matrix(y_true, y_pred), atol=1e-6
    )

  def test_confusion_matrix_metric_invalid_average_type(self):
    confusion_matrix = _classification.ConfusionMatrixAggFn(
        metrics=(ConfusionMatrixMetric.PRECISION,),
        average=AverageType.WEIGHTED,
    )
    with self.assertRaisesRegex(
        NotImplementedError, '"weighted" average is not supported'
    ):
      confusion_matrix([0, 1, 0], [1, 0, 0])

  def test_confusion_matrix_metric_invalid_metric(self):
    confusion_matrix = _classification.ConfusionMatrixAggFn(
        metrics=(ConfusionMatrixMetric.MEAN_AVERAGE_PRECISION,),
    )
    with self.assertRaisesRegex(NotImplementedError, "metric is not supported"):
      confusion_matrix([0, 1, 0], [1, 0, 0])

  @parameterized.named_parameters([
      dict(
          testcase_name="multiclass_indicator",
          y_pred=[[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
          y_true=[[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0]],
          input_type="multiclass-indicator",
          metrics=["precision", "recall", "f1_score", "accuracy"],
          # precision = mean([1, 0, 0, 1, 1, 0]) = 3 / 6 = 0.5
          # recall = mean([1, 0, 0, 1, 1, 0]) = 3 / 6 = 0.5
          # f1_score = mean(2 * precision * recall / (precision + recall)) = 0.5
          # accuracy = mean([1, 0, 0, 1, 1, 0]) = 0.5
          expected=(0.5, 0.5, 0.5, 0.5),
      ),
      dict(
          testcase_name="multiclass_samples",
          y_pred=["y", "n", "y", "n", "y", "n", "n", "u"],
          y_true=["y", "y", "n", "n", "y", "n", "y", "u"],
          input_type="multiclass",
          metrics=["precision", "recall", "f1_score", "accuracy"],
          # precision = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8 = 0.625
          # recall = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8 = 0.625
          # f1_score = 2 * precision * recall / (precision + recall) = 0.625
          # accuracy = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8 = 0.625
          expected=(0.625, 0.625, 0.625, 0.625),
      ),
      dict(
          testcase_name="multiclass_multioutput",
          y_pred=[["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]],
          y_true=[["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]],
          input_type="multiclass-multioutput",
          # A vocab is required to align class index for macro.
          # vocab={"y": 0, "n": 1, "u": 2},
          metrics=["precision", "recall", "f1_score", "accuracy"],
          # precision = mean([1, 0.5, 0, 1, 1, 1, 0, 1]) = 5.5/8 = 0.6875
          # recall = mean([1, 1, 0, 1, 0.5, 1, 0, 1]) = 5.5 / 8 = 0.6875
          # f1_score = mean(preceision * recall / (precision + recall) )= 0.6667
          # accuracy = mean([1, 1, 0, 1, 1, 1, 0, 1]) = 6 / 8 = 0.75
          expected=(0.6875, 0.6875, 0.6666667, 0.75),
      ),
  ])
  def test_samplewise_confusion_matrix_metric(
      self,
      y_true,
      y_pred,
      metrics,
      input_type,
      expected,
      vocab=None,
  ):
    confusion_matrix = _classification.SamplewiseConfusionMatrixAggFn(
        metrics=metrics,
        input_type=input_type,
        vocab=vocab,
    )
    np.testing.assert_allclose(expected, confusion_matrix(y_true, y_pred))

  def test_confusion_matrix_merge(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = _classification.ConfusionMatrixAggFn()
    baseline = _classification._ConfusionMatrix(tp=1, tn=1, fp=1, fn=1)
    acc = confusion_matrix.add_inputs(baseline, y_true, y_pred)
    acc_other = confusion_matrix.add_inputs(None, y_true, y_pred)
    actual = confusion_matrix.merge_accumulators([acc, acc_other])
    expected = _classification._ConfusionMatrix(tp=5, tn=5, fp=3, fn=5)
    self.assertEqual(expected, actual)

  def test_samplewise_confusion_matrix_merge(self):
    y_pred = ["dog", "cat", "cat"]
    y_true = ["dog", "cat", "bird"]
    confusion_matrix = _classification.SamplewiseConfusionMatrixAggFn(
        metrics=(ConfusionMatrixMetric.PRECISION,),
        input_type=InputType.MULTICLASS,
    )
    acc = confusion_matrix.add_inputs(
        confusion_matrix.create_accumulator(), y_true, y_pred
    )
    actual = confusion_matrix.add_inputs(acc, y_true, y_pred)
    expected = {"precision": _classification.MeanState(4, 6)}
    self.assertDictEqual(expected, actual)

  def test_confusion_matrix_add(self):
    a = _classification._ConfusionMatrix(tp=4, tn=4, fp=2, fn=4)
    b = _classification._ConfusionMatrix(tp=1, tn=6, fp=9, fn=2)
    expected = _classification._ConfusionMatrix(tp=5, tn=10, fp=11, fn=6)
    self.assertEqual(a + b, expected)

  def test_confusion_matrix_trues_positives(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = _classification.ConfusionMatrixAggFn()
    cm = confusion_matrix(y_true, y_pred)[0]
    self.assertEqual(3, cm.p)
    self.assertEqual(4, cm.t)

  def test_confusion_matrix_invalid_input_type(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = _classification.ConfusionMatrixAggFn(
        input_type=InputType.CONTINUOUS
    )
    with self.assertRaisesRegex(NotImplementedError, "is not supported"):
      confusion_matrix(y_true, y_pred)

  def test_samplewise_confusion_matrix_invalid_input_type(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = _classification.SamplewiseConfusionMatrixAggFn(
        metrics=(), input_type=InputType.CONTINUOUS
    )
    with self.assertRaisesRegex(NotImplementedError, "is not supported"):
      confusion_matrix(y_true, y_pred)

  def test_confusion_matrix_invalid_average_type(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = _classification.ConfusionMatrixAggFn(
        average=AverageType.WEIGHTED
    )
    with self.assertRaisesRegex(NotImplementedError, " is not supported"):
      confusion_matrix(y_true, y_pred)

  def test_confusion_matrix_binary_average_invalid_input(self):
    y_pred = ["dog", "cat", "cat", "bird", "tiger"]
    y_true = ["dog", "cat", "bird", "cat", "tiger"]
    confusion_matrix = _classification.ConfusionMatrixAggFn(
        input_type=InputType.MULTICLASS_MULTIOUTPUT,
        average=AverageType.BINARY,
    )
    with self.assertRaisesRegex(ValueError, "input is not supported"):
      confusion_matrix(y_true, y_pred)

  def test_confusion_matrix_multiclass_indicator_incorrect_input_shape(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = _classification.ConfusionMatrixAggFn(
        input_type=InputType.MULTICLASS_INDICATOR
    )
    with self.assertRaisesRegex(ValueError, "needs to be 2D array"):
      confusion_matrix(y_true, y_pred)

  def test_confusion_matrix_multiclass_macro_vocab_reqruired(self):
    confusion_matrix = _classification.ConfusionMatrixAggFn(
        average=AverageType.MACRO
    )
    acc = _classification._ConfusionMatrix(tp=1, tn=1, fp=1, fn=1)
    with self.assertRaisesRegex(ValueError, "Global vocab is needed"):
      confusion_matrix.merge_accumulators([acc])

  def test_confusion_matrix_unknown_tn(self):
    y_pred = [1, 0, 1, 0, 1, 0, 0]
    y_true = [1, 1, 0, 0, 1, 0, 1]
    confusion_matrix = _classification.ConfusionMatrixAggFn(unknown_tn=True)
    actual = confusion_matrix(y_true, y_pred)
    expected = _classification._ConfusionMatrix(
        tp=2, tn=float("inf"), fp=1, fn=2
    )
    self.assertEqual((expected,), actual)

  def test_confusion_matrix_samples_average_disallowed(self):
    with self.assertRaisesRegex(ValueError, "average is unsupported,"):
      _classification.ConfusionMatrixAggFn(average=AverageType.SAMPLES)

  def test_confusion_matrix_binary_input_samples_average_disallowed(self):
    with self.assertRaisesRegex(
        ValueError, "Samples average is not available for Binary"
    ):
      _classification.SamplewiseConfusionMatrixAggFn(
          metrics=(), input_type=InputType.BINARY
      )


if __name__ == "__main__":
  absltest.main()
