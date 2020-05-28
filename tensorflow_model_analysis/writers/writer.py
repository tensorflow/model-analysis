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
"""Writer types."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Any, NamedTuple, Text, Union

import apache_beam as beam
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.validators import validator

# A writer is a PTransform that takes Evaluation or Validation output as input
# and serializes the associated PCollections of data to a sink.
Writer = NamedTuple(
    'Writer',
    [
        ('stage_name', Text),
        # PTransform Evaluation -> PDone or Validation -> PDone
        ('ptransform', beam.PTransform)
    ])


@beam.ptransform_fn
@beam.typehints.with_input_types(Any)
@beam.typehints.with_output_types(beam.pvalue.PDone)
def Write(evaluation_or_validation: Union[evaluator.Evaluation,
                                          validator.Validation], key: Text,
          ptransform: beam.PTransform) -> beam.pvalue.PDone:
  """Writes given Evaluation or Validation data using given writer PTransform.

  Args:
    evaluation_or_validation: Evaluation or Validation data.
    key: Key for Evaluation or Validation output to write. It is valid for the
      key to not exist in the dict (in which case the write is a no-op).
    ptransform: PTransform to use for writing.

  Raises:
    ValueError: If Evaluation or Validation is empty. The key does not need to
      exist in the Evaluation or Validation, but the dict must not be empty.

  Returns:
    beam.pvalue.PDone.
  """
  if not evaluation_or_validation:
    raise ValueError('Evaluations and Validations cannot be empty')
  if key in evaluation_or_validation:
    return evaluation_or_validation[key] | ptransform
  return beam.pvalue.PDone(list(evaluation_or_validation.values())[0].pipeline)
