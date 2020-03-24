# Lint as: python3
# Copyright 2018 Google LLC
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
"""Test for evaluator."""

from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.extractors import extractor


class EvaluatorTest(testutil.TensorflowModelAnalysisTest):

  def testVerifyEvaluatorRaisesValueError(self):
    extractors = [
        extractor.Extractor(stage_name='ExtractorThatExists', ptransform=None)
    ]
    evaluator.verify_evaluator(
        evaluator.Evaluator(
            stage_name='EvaluatorWithoutError',
            run_after='ExtractorThatExists',
            ptransform=None), extractors)

    with self.assertRaises(ValueError):
      evaluator.verify_evaluator(
          evaluator.Evaluator(
              stage_name='EvaluatorWithError',
              run_after='ExtractorThatDoesNotExist',
              ptransform=None), extractors)


if __name__ == '__main__':
  tf.test.main()
