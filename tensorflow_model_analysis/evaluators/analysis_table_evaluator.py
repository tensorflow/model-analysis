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
"""Public API for creating analysis table."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Any, Dict, Iterable, Optional, Text, Union

import apache_beam as beam
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.extractors import extractor


def AnalysisTableEvaluator(  # pylint: disable=invalid-name
    key: Text = constants.ANALYSIS_KEY,
    run_after: Text = extractor.LAST_EXTRACTOR_STAGE_NAME,
    include: Optional[Union[Iterable[Text], Dict[Text, Any]]] = None,
    exclude: Optional[Union[Iterable[Text],
                            Dict[Text, Any]]] = None) -> evaluator.Evaluator:
  """Creates an Evaluator for returning Extracts data for analysis.

  If both include and exclude are None then tfma.INPUT_KEY extracts will be
  excluded by default.

  Args:
    key: Name to use for key in Evaluation output.
    run_after: Extractor to run after (None means before any extractors).
    include: List or map of keys to include in output. Keys starting with '_'
      are automatically filtered out at write time. If a map of keys is passed
      then the keys and sub-keys that exist in the map will be included in the
      output. An empty dict behaves as a wildcard matching all keys or the value
      itself. Since matching on feature values is not currently supported, an
      empty dict must be used to represent the leaf nodes.
      For example: {'key1': {'key1-subkey': {}}, 'key2': {}}.
    exclude: List or map of keys to exclude from output. If a map of keys is
      passed then the keys and sub-keys that exist in the map will be excluded
      from the output. An empty dict behaves as a wildcard matching all keys or
      the value itself. Since matching on feature values is not currently
      supported, an empty dict must be used to represent the leaf nodes.
      For example: {'key1': {'key1-subkey': {}}, 'key2': {}}.

  Returns:
    Evaluator for collecting analysis data. The output is stored under the key
    'analysis'.

  Raises:
    ValueError: If both include and exclude are used.
  """
  # pylint: disable=no-value-for-parameter
  return evaluator.Evaluator(
      stage_name='EvaluateExtracts',
      run_after=run_after,
      ptransform=EvaluateExtracts(key=key, include=include, exclude=exclude))
  # pylint: enable=no-value-for-parameter


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(Any)
def EvaluateExtracts(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    key: Text = constants.ANALYSIS_KEY,
    include: Optional[Union[Iterable[Text], Dict[Text, Any]]] = None,
    exclude: Optional[Union[Iterable[Text],
                            Dict[Text, Any]]] = None) -> evaluator.Evaluation:
  """Creates Evaluation output for extracts.

  If both include and exclude are None then tfma.INPUT_KEY extracts will be
  excluded by default.

  Args:
    extracts: PCollection of Extracts.
    key: Name to use for key in Evaluation output.
    include: List or map of keys to include in output. Keys starting with '_'
      are automatically filtered out at write time. If a map of keys is passed
      then the keys and sub-keys that exist in the map will be included in the
      output. An empty dict behaves as a wildcard matching all keys or the value
      itself. Since matching on feature values is not currently supported, an
      empty dict must be used to represent the leaf nodes.
      For example: {'key1': {'key1-subkey': {}}, 'key2': {}}.
    exclude: List or map of keys to exclude from output. If a map of keys is
      passed then the keys and sub-keys that exist in the map will be excluded
      from the output. An empty dict behaves as a wildcard matching all keys or
      the value itself. Since matching on feature values is not currently
      supported, an empty dict must be used to represent the leaf nodes.
      For example: {'key1': {'key1-subkey': {}}, 'key2': {}}.

  Returns:
    Evaluation containing PCollection of Extracts.
  """
  if include is None and exclude is None:
    exclude = [constants.INPUT_KEY]
  filtered = extracts
  if include or exclude:
    filtered = extracts | extractor.Filter(include=include, exclude=exclude)
  return {key: filtered}
