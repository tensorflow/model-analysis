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
# Standard __future__ imports
from __future__ import print_function

import copy
# Standard Imports

import apache_beam as beam

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import constants as eval_saved_model_constants
from tensorflow_model_analysis.eval_saved_model import dofn
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.extractors import feature_extractor
from typing import Generator, List, Optional

PREDICT_EXTRACTOR_STAGE_NAME = 'Predict'


def PredictExtractor(eval_shared_model: types.EvalSharedModel,
                     desired_batch_size: Optional[int] = None,
                     materialize: Optional[bool] = True) -> extractor.Extractor:
  """Creates an Extractor for TFMAPredict.

  The extractor's PTransform loads and runs the eval_saved_model against every
  example yielding a copy of the Extracts input with an additional extract
  of type FeaturesPredictionsLabels keyed by
  tfma.FEATURES_PREDICTIONS_LABELS_KEY.

  Args:
    eval_shared_model: Shared model parameters for EvalSavedModel.
    desired_batch_size: Optional batch size for batching in Aggregate.
    materialize: True to call the FeatureExtractor to add MaterializedColumn
      entries for the features, predictions, and labels.

  Returns:
    Extractor for extracting features, predictions, labels, and other tensors
    during predict.
  """
  # pylint: disable=no-value-for-parameter
  return extractor.Extractor(
      stage_name=PREDICT_EXTRACTOR_STAGE_NAME,
      ptransform=_TFMAPredict(
          eval_shared_model=eval_shared_model,
          desired_batch_size=desired_batch_size,
          materialize=materialize))
  # pylint: enable=no-value-for-parameter


@beam.typehints.with_input_types(beam.typehints.List[types.Extracts])
@beam.typehints.with_output_types(types.Extracts)
class _TFMAPredictionDoFn(dofn.EvalSavedModelDoFn):
  """A DoFn that loads the model and predicts."""

  def __init__(self, eval_shared_model: types.EvalSharedModel) -> None:
    super(_TFMAPredictionDoFn, self).__init__(eval_shared_model)
    self._predict_batch_size = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'predict_batch_size')
    self._num_instances = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_instances')

  def process(self, element: List[types.Extracts]
             ) -> Generator[types.Extracts, None, None]:
    batch_size = len(element)
    self._predict_batch_size.update(batch_size)
    self._num_instances.inc(batch_size)
    serialized_examples = [x[constants.INPUT_KEY] for x in element]

    # Compute FeaturesPredictionsLabels for each serialized_example
    for fetched in self._eval_saved_model.predict_list(serialized_examples):
      element_copy = copy.copy(element[fetched.input_ref])
      element_copy[constants.FEATURES_PREDICTIONS_LABELS_KEY] = (
          self._eval_saved_model.as_features_predictions_labels([fetched])[0])
      for key in fetched.values:
        if key in (eval_saved_model_constants.FEATURES_NAME,
                   eval_saved_model_constants.LABELS_NAME,
                   eval_saved_model_constants.PREDICTIONS_NAME):
          continue
        element_copy[key] = fetched.values[key]
      yield element_copy


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _TFMAPredict(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    eval_shared_model: types.EvalSharedModel,
    desired_batch_size: Optional[int] = None,
    materialize: Optional[bool] = True) -> beam.pvalue.PCollection:
  """A PTransform that adds predictions to Extracts.

  Args:
    extracts: PCollection of Extracts containing a serialized example to be fed
      to the model.
    eval_shared_model: Shared model parameters for EvalSavedModel.
    desired_batch_size: Optional. Desired batch size for prediction.
    materialize: True to call the FeatureExtractor to add MaterializedColumn
      entries for the features, predictions, and labels.

  Returns:
    PCollection of Extracts, where the extracts contains the features,
    predictions, labels retrieved.
  """
  batch_args = {}
  if desired_batch_size:
    batch_args = dict(
        min_batch_size=desired_batch_size, max_batch_size=desired_batch_size)

  # We don't actually need to add the add_metrics_callbacks to do Predict,
  # but because if we want to share the model between Predict and subsequent
  # stages (i.e. we use same shared handle for this and subsequent stages),
  # then if we don't add the metrics callbacks here, they won't be present
  # in the model in the later stages if we reuse the model from this stage.
  extracts = (
      extracts
      | 'Batch' >> beam.BatchElements(**batch_args)
      | 'Predict' >> beam.ParDo(
          _TFMAPredictionDoFn(eval_shared_model=eval_shared_model)))

  if materialize:
    return extracts | 'ExtractFeatures' >> feature_extractor._ExtractFeatures(  # pylint: disable=protected-access
        additional_extracts=eval_shared_model.additional_fetches)

  return extracts
