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
"""API for Tensorflow Model Analysis model validation."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Any, Dict, List, Text

import apache_beam as beam
from tensorflow_model_analysis import types
from tensorflow_model_analysis.validators import validator


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(Any)
def Validate(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection, alternatives: Dict[Text,
                                                          beam.PTransform],
    validators: List[validator.Validator]) -> validator.Validation:
  """Performs validation of alternative evaluations.

  Args:
    extracts: PCollection of extracts.
    alternatives: Dict of PTransforms (Extracts -> Evaluation) whose output will
      be compared for validation purposes (e.g. 'baseline' vs 'candidate').
    validators: List of validators for validating the output from running the
      alternatives. The Validation outputs produced by the validators will be
      merged into a single output. If there are overlapping output keys, later
      outputs will replace earlier outputs sharing the same key.

  Returns:
    Validation dict.
  """
  evaluations = {}
  for key in alternatives:
    evaluations[key] = extracts | 'Evaluate(%s)' % key >> alternatives[key]

  validation = {}
  for v in validators:
    validation.update(evaluations | v.stage_name >> v.ptransform)
  return validation
