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
"""Public API for performing evaluations using the EvalSavedModel."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function



import apache_beam as beam

from tensorflow_model_analysis import types
from tensorflow_model_analysis.api.impl import aggregate
from tensorflow_model_analysis.api.impl import api_types
from tensorflow_model_analysis.api.impl import slice as slice_api
from tensorflow_model_analysis.extractors import feature_extractor
from tensorflow_model_analysis.extractors import predict_extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.slicer import slicer
from tensorflow_model_analysis.types_compat import List, Optional

_METRICS_NAMESPACE = 'tensorflow_model_analysis'


@beam.ptransform_fn
@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(beam.typehints.Any)
def ToExampleAndExtracts(examples):
  """Converts an example to ExampleAndExtracts with empty extracts."""
  return (examples
          |
          beam.Map(lambda x: types.ExampleAndExtracts(example=x, extracts={})))


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.typehints.Any)
@beam.typehints.with_output_types(beam.typehints.Any)
def Extract(examples_and_extracts,
            extractors):
  """Performs Extractions serially in provided order."""
  augmented = examples_and_extracts

  for extractor in extractors:
    augmented = augmented | extractor.stage_name >> extractor.ptransform

  return augmented


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.typehints.Any)
# No typehint for output type, since it's a multi-output DoFn result that
# Beam doesn't support typehints for yet (BEAM-3280).
def Evaluate(  # pylint: disable=invalid-name
    examples_and_extracts,
    eval_shared_model,
    desired_batch_size = None,
):
  """Evaluate the given EvalSavedModel on the given examples.

  This is for TFMA use only. Users should call
  tfma.ExtractEvaluateAndWriteResults instead of this function.

  Args:
    examples_and_extracts: PCollection of ExampleAndExtracts. The extracts MUST
      contain a FeaturesPredictionsLabels extract with key 'fpl' and a list of
      SliceKeyType extracts with key 'slice_keys'. Typically these will be added
      by calling the default_extractors function.
    eval_shared_model: Shared model parameters for EvalSavedModel including any
      additional metrics (see EvalSharedModel for more information on how to
      configure additional metrics).
    desired_batch_size: Optional batch size for batching in Aggregate.

  Returns:
    DoOutputsTuple. The tuple entries are
    PCollection of (slice key, metrics) and
    PCollection of (slice key, plot metrics).
  """
  # pylint: disable=no-value-for-parameter
  return (
      examples_and_extracts

      # Input: one example at a time, with slice keys in extracts.
      # Output: one fpl example per slice key (notice that the example turns
      #         into n, replicated once per applicable slice key)
      | 'FanoutSlices' >> slice_api.FanoutSlices()

      # Each slice key lands on one shard where metrics are computed for all
      # examples in that shard -- the "map" and "reduce" parts of the
      # computation happen within this shard.
      # Output: Multi-outputs, a dict of slice key to computed metrics, and
      # plots if applicable.
      | 'ComputePerSliceMetrics' >> aggregate.ComputePerSliceMetrics(
          eval_shared_model=eval_shared_model,
          desired_batch_size=desired_batch_size))
  # pylint: enable=no-value-for-parameter


@beam.ptransform_fn
@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(beam.typehints.Any)
def BuildDiagnosticTable(  # pylint: disable=invalid-name
    examples,
    eval_shared_model,
    slice_spec = None,
    desired_batch_size = None,
    extractors = None
):
  """Build diagnostics for the spacified EvalSavedModel and example collection.

  Args:
    examples: PCollection of input examples. Can be any format the model accepts
      (e.g. string containing CSV row, TensorFlow.Example, etc).
    eval_shared_model: Shared model parameters for EvalSavedModel.
    slice_spec: Optional list of SingleSliceSpec specifying the slices to slice
      the data into. If None, defaults to the overall slice.
    desired_batch_size: Optional batch size for batching in Predict and
      Aggregate.
    extractors: Optional list of Extractors to execute prior to slicing and
      aggregating the metrics. If not provided, a default set will be run.

  Returns:
    PCollection of ExampleAndExtracts
  """

  if not extractors:
    extractors = [
        predict_extractor.PredictExtractor(eval_shared_model,
                                           desired_batch_size),
        feature_extractor.FeatureExtractor(),
        slice_key_extractor.SliceKeyExtractor(slice_spec)
    ]

  # pylint: disable=no-value-for-parameter
  return (examples
          | 'ToExampleAndExtracts' >> ToExampleAndExtracts()
          | Extract(extractors=extractors))
  # pylint: enable=no-value-for-parameter
