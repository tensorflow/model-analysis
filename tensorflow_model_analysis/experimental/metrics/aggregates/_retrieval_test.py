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
from tensorflow_model_analysis.experimental.metrics.aggregates import _retrieval


InputType = _retrieval.InputType


class ClassificationTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="multiclass",
          y_pred=["y", "n", "y", "n", "y", "n", "n", "u"],
          y_true=["y", "y", "n", "n", "y", "n", "y", "u"],
          input_type=InputType.MULTICLASS,
          metrics=[_retrieval.RetrievalMetric.PRECISION],
          k_list=[1, 2],
          # precision@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8 = 0.625
          # precision@2 = precision@1 since there is only one output per
          # example in multiclass case.
          expected=([[0.625, 0.625]]),
      ),
      dict(
          testcase_name="multiclass_multioutput",
          y_pred=[["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]],
          y_true=[["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]],
          input_type=InputType.MULTICLASS_MULTIOUTPUT,
          metrics=[_retrieval.RetrievalMetric.PRECISION],
          k_list=[1, 2],
          # precision@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5/8
          # precision@2 = mean([1, 1/2, 0, 1, 1, 1, 0, 1]) = 5.5/8
          expected=([[5 / 8, 5.5 / 8]]),
      ),
      dict(
          testcase_name="multiclass_multioutput_infinity_k",
          y_pred=[["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]],
          y_true=[["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]],
          input_type=InputType.MULTICLASS_MULTIOUTPUT,
          metrics=[_retrieval.RetrievalMetric.PRECISION],
          k_list=None,
          # precision = mean([1, 1/2, 0, 1, 0, 1, 0, 1]) = 5.5 / 8
          expected=([[5.5 / 8]]),
      ),
  ])
  def test_compute_retrieval_metric(
      self,
      y_true,
      y_pred,
      input_type,
      metrics,
      k_list,
      expected,
  ):
    confusion_matrix = _retrieval.TopKRetrievalAggFn(
        metrics=metrics,
        k_list=k_list,
        input_type=input_type,
    )
    np.testing.assert_allclose(expected, confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
  absltest.main()
