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
# Standard __future__ imports
from __future__ import print_function

# Standard Imports

import apache_beam as beam
from tensorflow_model_analysis import types
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.evaluators import analysis_table_evaluator
from tensorflow_model_analysis.evaluators import evaluator
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.extractors import feature_extractor
from tensorflow_model_analysis.extractors import predict_extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.slicer import slicer
from typing import List, Optional


@beam.ptransform_fn
@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(evaluator.Evaluation)
def BuildAnalysisTable(  # pylint: disable=invalid-name
    examples: beam.pvalue.PCollection,
    eval_shared_model: types.EvalSharedModel,
    slice_spec: Optional[List[slicer.SingleSliceSpec]] = None,
    desired_batch_size: Optional[int] = None,
    extractors: Optional[List[extractor.Extractor]] = None,
    evaluators: Optional[List[evaluator.Evaluator]] = None
) -> beam.pvalue.PCollection:
  """Builds an analysis table from data extracted from the input.

  Use this function to build an example-oriented PCollection of output data
  useful for debugging models.

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
    evaluators: Optional list of Evaluators for evaluating Extracts. If not
      provided a default set will be used..

  Returns:
    beam.pvalue.PCollection of Extracts. The caller is responsible for
    committing to file for now.
  """
  if slice_spec is None:
    slice_spec = [slicer.SingleSliceSpec()]

  if not extractors:
    extractors = [
        predict_extractor.PredictExtractor(eval_shared_model,
                                           desired_batch_size),
        feature_extractor.FeatureExtractor(),
        slice_key_extractor.SliceKeyExtractor(slice_spec)
    ]
  if not evaluators:
    evaluators = [analysis_table_evaluator.AnalysisTableEvaluator()]

  # pylint: disable=no-value-for-parameter
  return (examples
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | model_eval_lib.ExtractAndEvaluate(
              extractors=extractors, evaluators=evaluators))
  # pylint: enable=no-value-for-parameter
