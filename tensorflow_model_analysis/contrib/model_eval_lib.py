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
"""Contrib Experimental API for Tensorflow Model Analysis, subject to change."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function


import apache_beam as beam

from tensorflow_model_analysis import types
from tensorflow_model_analysis.api.impl import evaluate
from tensorflow_model_analysis.slicer import slicer
from tensorflow_model_analysis.types_compat import List, Optional


@beam.ptransform_fn
@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(beam.typehints.Any)
def BuildDiagnosticTable(  # pylint: disable=invalid-name
    examples,
    eval_shared_model,
    slice_spec = None,
    desired_batch_size = None):
  """Public API version of evaluate.BuildDiagnosticTable.

  Use this function to build an example-oriented PCollection containing, for
  each example, an ExampleAndExtracts, useful for debugging models.

  Args:
    examples: PCollection of input examples. Can be any format the model accepts
      (e.g. string containing CSV row, TensorFlow.Example, etc).
    eval_shared_model: Shared model parameters for EvalSavedModel.
    slice_spec: Optional list of SingleSliceSpec specifying the slices to slice
      the data into. If None, defaults to the overall slice.
    desired_batch_size: Optional batch size for batching in Predict and
      Aggregate.

  Returns:
    beam.pvalue.PCollection of ExampleAndExtracts. The caller is responsible for
    committing to file for now.
  """
  if slice_spec is None:
    slice_spec = [slicer.SingleSliceSpec()]
  return (examples
          | 'BuildDiagnosticTable' >> evaluate.BuildDiagnosticTable(
              eval_shared_model, slice_spec, desired_batch_size))
