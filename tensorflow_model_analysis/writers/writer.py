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

from typing import NamedTuple, Optional, Union

import apache_beam as beam
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.validators import validator

# A writer is a PTransform that takes Evaluation or Validation output as input
# and serializes the associated PCollections of data to a sink.
Writer = NamedTuple(
    'Writer',
    [
        ('stage_name', str),
        # PTransform (Evaluation|Validation) -> Beam write result
        ('ptransform', beam.PTransform)
    ])


@beam.ptransform_fn
def Write(evaluation_or_validation: Union[evaluator.Evaluation,
                                          validator.Validation], key: str,
          ptransform: beam.PTransform) -> Optional[beam.PCollection]:
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
    The result of the underlying beam write PTransform. This makes it possible
    for interactive environments to execute your writer, as well as for
    downstream Beam stages to make use of the files that are written.
  """
  if not evaluation_or_validation:
    raise ValueError('Evaluations and Validations cannot be empty')
  if key in evaluation_or_validation:
    return evaluation_or_validation[key] | ptransform
  return None
