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
"""Public API for creating analysis table."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function



import apache_beam as beam
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.types_compat import Text


def AnalysisTableEvaluator(  # pylint: disable=invalid-name
    key = constants.ANALYSIS_KEY,
    run_after = extractor.LAST_EXTRACTOR_STAGE_NAME
):
  """Creates an Evaluator for returning Extracts data for analysis.

  Args:
    key: Name to use for key in Evaluation output.
    run_after: Extractor to run after (None means before any extractors).

  Returns:
    Evaluator for collecting analysis data. The output is stored under the key
    'analysis'.
  """
  # pylint: disable=no-value-for-parameter
  return evaluator.Evaluator(
      stage_name='EvaluateExtracts',
      run_after=run_after,
      ptransform=EvaluateExtracts(key=key))
  # pylint: enable=no-value-for-parameter


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.typehints.Any)
@beam.typehints.with_output_types(evaluator.Evaluation)
def EvaluateExtracts(  # pylint: disable=invalid-name
    extracts,
    key = constants.ANALYSIS_KEY):
  """Creates Evaluation output for extracts.

  Args:
    extracts: PCollection of Extracts.
    key: Name to use for key in Evaluation output.

  Returns:
    Evaluation containing PCollection of Extracts.
  """
  return {key: extracts}
