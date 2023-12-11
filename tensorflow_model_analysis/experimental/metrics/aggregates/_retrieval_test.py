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
RetrievalMetric = _retrieval.RetrievalMetric


class ClassificationTest(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="multiclass",
          y_pred=["y", "n", "y", "n", "y", "n", "n", "u"],
          y_true=["y", "y", "n", "n", "y", "n", "y", "u"],
          input_type=InputType.MULTICLASS,
          metrics=[
              RetrievalMetric.FOWLKES_MALLOWS_INDEX,
              RetrievalMetric.THREAT_SCORE,
              RetrievalMetric.DCG_SCORE,
              RetrievalMetric.NDCG_SCORE,
              RetrievalMetric.MEAN_RECIPROCAL_RANK,
              RetrievalMetric.PRECISION,
              RetrievalMetric.FALSE_DISCOVERY_RATE,
              RetrievalMetric.MEAN_AVERAGE_PRECISION,
              RetrievalMetric.RECALL,
              RetrievalMetric.MISS_RATE,
              RetrievalMetric.F1_SCORE,
              RetrievalMetric.ACCURACY,
          ],
          k_list=[1, 2],
          expected=([
              # FMI@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # FMI@2 = FMI@1 because there is only one output.
              [5 / 8, 5 / 8],
              # threat_score@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # threat_score@2 = threat_score@1 since there is only one output.
              [5 / 8, 5 / 8],
              # DCG@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # DCG@2 = DCG@1 since there is only one output.
              [5 / 8, 5 / 8],
              # NDCG@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # NDCG@2 = NDCG@1 since there is only one output.
              [5 / 8, 5 / 8],
              # mRR@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # mRR@2 = mRR@1 since there is only one output.
              [5 / 8, 5 / 8],
              # precision@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # precision@2 = precision@1 since there is only one output.
              [5 / 8, 5 / 8],
              # FDR@1 = mean([0, 1, 1, 0, 0, 0, 1, 0]) = 3 / 8
              # FDR@2 = FDR@1 since there is only one output.
              [3 / 8, 3 / 8],
              # mAP@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # mAP@2 = mAP@1 since there is only one output.
              [5 / 8, 5 / 8],
              # recall@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # recall@2 = recall@1 since there is only one output.
              [5 / 8, 5 / 8],
              # miss_rate@1 = mean([0, 1, 1, 0, 0, 0, 1, 0]) = 3 / 8
              # miss_rate@2 = miss_rate@1 since there is only one output.
              [3 / 8, 3 / 8],
              # f1_score@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # f1_score@2 = f1_score@1 since there is only one output.
              [5 / 8, 5 / 8],
              # accuracy@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # accuracy@2 = accuracy@1 since there is only one output.
              [5 / 8, 5 / 8],
          ]),
      ),
      dict(
          testcase_name="multiclass_multioutput",
          y_pred=[["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]],
          y_true=[["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]],
          input_type=InputType.MULTICLASS_MULTIOUTPUT,
          metrics=[
              RetrievalMetric.FOWLKES_MALLOWS_INDEX,
              RetrievalMetric.THREAT_SCORE,
              RetrievalMetric.DCG_SCORE,
              RetrievalMetric.NDCG_SCORE,
              RetrievalMetric.MEAN_RECIPROCAL_RANK,
              RetrievalMetric.PRECISION,
              RetrievalMetric.FALSE_DISCOVERY_RATE,
              RetrievalMetric.MEAN_AVERAGE_PRECISION,
              RetrievalMetric.RECALL,
              RetrievalMetric.MISS_RATE,
              RetrievalMetric.F1_SCORE,
              RetrievalMetric.ACCURACY,
          ],
          k_list=[1, 2],
          expected=([
              # FMI@1 = mean(sqrt([1, 0, 0, 1, 0.5, 1, 0, 1]))
              # FMI@2 = mean(sqrt([1, 0.5, 0, 1, 0.5, 1, 0, 1]))
              [
                  np.sqrt([1, 0, 0, 1, 0.5, 1, 0, 1]).mean(),
                  np.sqrt([1, 0.5, 0, 1, 0.5, 1, 0, 1]).mean(),
              ],
              # threat_score@1 = mean([1, 0, 0, 1, 0.5, 1, 0, 1]) = 4.5 / 8
              # threat_score@2 = mean([0.5, 0.5, 0, 0.5, 1/3, 0.5, 0, 0.5]) =
              #       = (2.5 + 1/3) / 8
              [4.5 / 8, (2.5 + 1 / 3) / 8],
              # DCG@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # DCG@2 = mean([1, 1/log2(3), 0, 1, 1, 1, 0, 1])
              #       = (5 + 1 / log2(3)) / 8
              [5 / 8, (1 / np.log2(3) + 5) / 8],
              # NDCG@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # NDCG@2 = mean([1, 1/log2(3), 0, 1, 1/(1+1/log2(3)), 1, 0, 1])
              #        = (4 + 1 / log2(3) + 1/(1+1/log2(3))) / 8
              [5 / 8, (4 + 1 / np.log2(3) + 1 / (1 + 1 / np.log2(3))) / 8],
              # mRR@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5 / 8
              # mRR@2 = mean([1, 1/2, 0, 1, 1, 1, 0, 1]) = 5.5 / 8
              [5 / 8, 5.5 / 8],
              # precision@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5/8
              # precision@2 = mean([1, 1/2, 0, 1, 1, 1, 0, 1]) = 5.5/8
              [5 / 8, 5.5 / 8],
              # FDR@1 = mean([0, 1, 1, 0, 0, 0, 1, 0]) = 3/8
              # FDR@2 = mean([0, 1/2, 1, 0, 0, 0, 1, 0]) = 2.5/8
              [3 / 8, 2.5 / 8],
              # mAP@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5/8
              # mAP@2 = mean([1, 1/2, 0, 1, 1/2, 1, 0, 1]) = 5/8
              [5 / 8, 5 / 8],
              # recall@1 = mean([1, 0, 0, 1, 1/2, 1, 0, 1]) = 4.5/8
              # recall@2 = mean([1, 1, 0, 1, 1/2, 1, 0, 1]) = 5.5/8
              [4.5 / 8, 5.5 / 8],
              # miss_rate@1 = mean([0, 1, 1, 0, 1/2, 0, 1, 0]) = 3.5/8
              # miss_rate@2 = mean([0, 0, 1, 0, 1/2, 0, 1, 0]) = 2.5/8
              [3.5 / 8, 2.5 / 8],
              # f1_score@1 = mean([1, 0, 0, 1, 1/1.5, 1, 0, 1]) = (4+2/3)/8
              # f1_score@2 = mean([1, 1/1.5, 0, 1, 1/1.5, 1, 0, 1]) = (4+4/3)/8
              [(4 + 2 / 3) / 8, (4 + 4 / 3) / 8],
              # accuracy@1 = mean([1, 0, 0, 1, 1, 1, 0, 1]) = 5/8
              # accuracy@2 = mean([1, 1, 0, 1, 1, 1, 0, 1]) = 6/8
              [5 / 8, 6 / 8],
          ]),
      ),
      dict(
          testcase_name="multiclass_multioutput_infinity_k",
          y_pred=[["y"], ["n", "y"], ["y"], ["n"], ["y"], ["n"], ["n"], ["u"]],
          y_true=[["y"], ["y"], ["n"], ["n"], ["y", "n"], ["n"], ["y"], ["u"]],
          input_type=InputType.MULTICLASS_MULTIOUTPUT,
          metrics=[RetrievalMetric.PRECISION],
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
