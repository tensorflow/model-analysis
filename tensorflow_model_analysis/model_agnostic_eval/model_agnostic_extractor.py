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
"""Public API for extracting FeaturesPredictionsLabels without eval model."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import copy
import datetime

# Standard Imports

import apache_beam as beam
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_metrics_graph import eval_metrics_graph
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.model_agnostic_eval import model_agnostic_predict as agnostic_predict
from tfx_bsl.beam import shared
from typing import Generator, List, Optional


def ModelAgnosticGetFPLFeedConfig(
    model_agnostic_config: agnostic_predict.ModelAgnosticConfig
) -> eval_metrics_graph.FPLFeedConfig:
  """Creates an FPLFeedConfig from the input ModelAgnosticConfig.

  Creates the placeholder ops based on the input ModelAgnosticConfig. The
  feature_spec is used to determine Tensor vs SparseTensor and dtype for the
  placeholder op.

  Args:
    model_agnostic_config: The config to use to generate the placeholder ops.

  Returns:
    An eval_metrics_graph.FPLFeedConfig which can be used to instantiate the
    infeed on a metric graph.

  Raises:
    ValueError: Supplied ModelAgnosticConfig is invalid.
  """
  features = {}
  predictions = {}
  labels = {}

  for key, value in model_agnostic_config.feature_spec.items():
    placeholder = None
    if isinstance(value, tf.io.FixedLenFeature):
      placeholder = (constants.PLACEHOLDER, value.dtype)
    elif isinstance(value, tf.io.VarLenFeature):
      placeholder = (constants.SPARSE_PLACEHOLDER, value.dtype)
    else:
      raise ValueError('Unsupported type %s in feature_spec.' % value)

    if key in model_agnostic_config.prediction_keys:
      predictions[key] = placeholder
    elif key in model_agnostic_config.label_keys:
      labels[key] = placeholder
    else:
      features[key] = placeholder
  return eval_metrics_graph.FPLFeedConfig(
      features=features, predictions=predictions, labels=labels)


# pylint: disable=no-value-for-parameter
def ModelAgnosticExtractor(
    model_agnostic_config: agnostic_predict.ModelAgnosticConfig,
    desired_batch_size: Optional[int] = None) -> extractor.Extractor:
  """Creates an Extractor for ModelAgnosticEval.

  The extractor's PTransform creates and runs ModelAgnosticEval against every
  example yielding a copy of the Extracts input with an additional extract of
  type FeaturesPredictionsLabels keyed by tfma.FEATURES_PREDICTIONS_LABELS_KEY.

  Args:
    model_agnostic_config: The config to use to be able to generate Features,
      Predictions, and Labels dict. This can be done through explicit labeling
      of keys in the input tf.Example.
    desired_batch_size: Optional batch size for batching in Predict.

  Returns:
    Extractor for extracting features, predictions, and labels during predict.

  Raises:
    ValueError: Supplied ModelAgnosticConfig is invalid.
  """
  return extractor.Extractor(
      stage_name='ModelAgnosticExtractor',
      ptransform=ModelAgnosticExtract(
          model_agnostic_config=model_agnostic_config,
          desired_batch_size=desired_batch_size))


@beam.typehints.with_input_types(beam.typehints.List[types.Extracts])
@beam.typehints.with_output_types(types.Extracts)
class _ModelAgnosticExtractDoFn(beam.DoFn):
  """A DoFn that extracts the FPL from the examples."""

  def __init__(self, model_agnostic_config: agnostic_predict.ModelAgnosticConfig
              ) -> None:
    self._model_agnostic_config = model_agnostic_config
    # TODO(b/140805724): It's odd that shared_handle is not passed as an
    # argument to the constructor. Logically, it seems to have a 1-1
    # correspondence with the model_agnostic_config, so it should be passed with
    # it.
    self._shared_handle = shared.Shared()
    self._model_agnostic_wrapper = None
    self._model_load_seconds = None
    self._model_load_seconds_distribution = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'model_load_seconds')

  def _make_construct_fn(  # pylint: disable=invalid-name
      self, model_agnostic_config: agnostic_predict.ModelAgnosticConfig):
    """Returns construct func for Shared for constructing ModelAgnosticEval."""

    def construct():  # pylint: disable=invalid-name
      """Function for constructing a EvalSavedModel."""
      start_time = datetime.datetime.now()
      model_agnostic_wrapper = agnostic_predict.ModelAgnosticPredict(
          model_agnostic_config)
      end_time = datetime.datetime.now()
      self._model_load_seconds = int((end_time - start_time).total_seconds())
      return model_agnostic_wrapper

    return construct

  def setup(self):
    self._model_agnostic_wrapper = self._shared_handle.acquire(
        self._make_construct_fn(self._model_agnostic_config))

  def process(self, element: List[types.Extracts]
             ) -> Generator[types.Extracts, None, None]:
    serialized_examples = [x[constants.INPUT_KEY] for x in element]

    # Compute FeaturesPredictionsLabels for each serialized_example using
    # the constructed model_agnostic_wrapper.
    for fpl in self._model_agnostic_wrapper.get_fpls_from_examples(
        serialized_examples):
      element_copy = copy.copy(element[fpl.input_ref])
      element_copy[constants.FEATURES_PREDICTIONS_LABELS_KEY] = fpl
      yield element_copy

  def finish_bundle(self):
    # Must update distribution in finish_bundle instead of setup
    # because Beam metrics are not supported in setup.
    if self._model_load_seconds is not None:
      self._model_load_seconds_distribution.update(self._model_load_seconds)
      self._model_load_seconds = None


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def ModelAgnosticExtract(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    model_agnostic_config: agnostic_predict.ModelAgnosticConfig,
    desired_batch_size: Optional[int] = None) -> beam.pvalue.PCollection:
  """A PTransform that generates features, predictions, labels.

  Args:
    extracts: PCollection of Extracts containing a serialized example to be fed
      to the model.
    model_agnostic_config: A config specifying how to extract
      FeaturesPredictionsLabels from the input input Extracts.
    desired_batch_size: Optional batch size for batching in Aggregate.

  Returns:
    PCollection of Extracts, where the extracts contains the features,
    predictions, labels retrieved.
  """
  batch_args = {}
  if desired_batch_size:
    batch_args = dict(
        min_batch_size=desired_batch_size, max_batch_size=desired_batch_size)
  return (extracts
          | 'Batch' >> beam.BatchElements(**batch_args)
          | 'ModelAgnosticExtract' >> beam.ParDo(
              _ModelAgnosticExtractDoFn(
                  model_agnostic_config=model_agnostic_config)))
