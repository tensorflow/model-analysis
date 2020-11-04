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
"""Implements API for extracting features from an example."""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import copy

from typing import Any, Dict, List, Optional, Text

from absl import logging
import apache_beam as beam
import numpy as np
import tensorflow as tf

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis import util
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.extractors import extractor

_FEATURE_EXTRACTOR_STAGE_NAME = 'ExtractFeatures'


def FeatureExtractor(
    additional_extracts: Optional[List[Text]] = None,
    excludes: Optional[List[bytes]] = None,
    extract_source: Text = constants.FEATURES_PREDICTIONS_LABELS_KEY,
    extract_dest: Text = constants.MATERIALIZE_COLUMNS):
  # pylint: disable=no-value-for-parameter
  return extractor.Extractor(
      stage_name=_FEATURE_EXTRACTOR_STAGE_NAME,
      ptransform=_ExtractFeatures(
          additional_extracts=additional_extracts,
          excludes=excludes,
          source=extract_source,
          dest=extract_dest))
  # pylint: enable=no-value-for-parameter


def _AugmentExtracts(data: Dict[Text, Any], prefix: Text, excludes: List[bytes],
                     extracts: types.Extracts) -> None:
  """Augments the Extracts with FeaturesPredictionsLabels.

  Args:
    data: Data dictionary returned by PredictExtractor.
    prefix: Prefix to use in column naming (e.g. 'features', 'labels', etc).
    excludes: List of strings containing features, predictions, or labels to
      exclude from materialization.
    extracts: The Extracts to be augmented. This is mutated in-place.

  Raises:
    TypeError: If the FeaturesPredictionsLabels is corrupt.
  """
  for name, val in data.items():
    if excludes is not None and name in excludes:
      continue
    # If data originated from FeaturesPredictionsLabels, then the value will be
    # stored under a 'node' key.
    if isinstance(val, dict) and encoding.NODE_SUFFIX in val:
      val = val.get(encoding.NODE_SUFFIX)

    if name in (prefix, util.KEY_SEPARATOR + prefix):
      col_name = prefix
    elif prefix not in ('features', 'predictions', 'labels'):
      # Names used by additional extracts should be properly escaped already so
      # avoid escaping the name a second time by manually combining the prefix.
      col_name = prefix + util.KEY_SEPARATOR + name
    else:
      col_name = util.compound_key([prefix, name])

    if isinstance(val, tf.compat.v1.SparseTensorValue):
      extracts[col_name] = types.MaterializedColumn(
          name=col_name, value=val.values)

    elif isinstance(val, np.ndarray) or isinstance(val, list):
      # Only support first dim for now
      val = val[0] if len(val) > 0 else []  # pylint: disable=g-explicit-length-test
      extracts[col_name] = types.MaterializedColumn(name=col_name, value=val)

    else:
      raise TypeError(
          'Dictionary item with key %s, value %s had unexpected type %s' %
          (name, val, type(val)))


def _ParseExample(extracts: types.Extracts,
                  materialize_columns: bool = True) -> None:
  """Feature extraction from serialized tf.Example."""
  # Deserialize the example.
  example = tf.train.Example()
  try:
    example.ParseFromString(extracts[constants.INPUT_KEY])
  except:  # pylint: disable=bare-except
    logging.warning('Could not parse tf.Example from the input source.')

  features = {}
  if constants.FEATURES_PREDICTIONS_LABELS_KEY in extracts:
    features = extracts[constants.FEATURES_PREDICTIONS_LABELS_KEY].features

  for name in example.features.feature:
    if materialize_columns or name not in features:
      key = util.compound_key(['features', name])
      value = example.features.feature[name]
      if value.HasField('bytes_list'):
        values = [v for v in value.bytes_list.value]
      elif value.HasField('float_list'):
        values = [v for v in value.float_list.value]
      elif value.HasField('int64_list'):
        values = [v for v in value.int64_list.value]
      if materialize_columns:
        extracts[key] = types.MaterializedColumn(name=key, value=values)
      if name not in features:
        features[name] = {encoding.NODE_SUFFIX: np.array([values])}


def _MaterializeFeatures(
    extracts: types.Extracts,
    additional_extracts: Optional[List[Text]] = None,
    excludes: Optional[List[bytes]] = None,
    source: Text = constants.FEATURES_PREDICTIONS_LABELS_KEY,
    dest: Text = constants.MATERIALIZE_COLUMNS,
) -> types.Extracts:
  """Converts FeaturesPredictionsLabels into MaterializedColumn in the extract.

  It must be the case that the PredictExtractor was called before calling this
  function.

  Args:
    extracts: The Extracts to be augmented.
    additional_extracts: Optional list of additional extracts to include along
      with the features, predictions, and labels.
    excludes: Optional list of strings containing features, predictions, or
      labels to exclude from materialization.
    source: Source for extracting features. Currently it supports extracting
      features from FPLs and input tf.Example protos.
    dest: Destination for extracted features. Currently supported are adding
      materialized columns, or the features dict of the FPLs.

  Returns:
    Returns Extracts (which is a deep copy of the original Extracts, so the
      original isn't mutated) with features populated.

  Raises:
    RuntimeError: When tfma.FEATURES_PREDICTIONS_LABELS_KEY key is not populated
      by PredictExtractor for FPL source or incorrect extraction source given.
  """
  # Make a deep copy, so we don't mutate the original.
  result = copy.deepcopy(extracts)

  if additional_extracts:
    for key in additional_extracts:
      if key in result:
        _AugmentExtracts(result[key], key, excludes, result)

  if source == constants.FEATURES_PREDICTIONS_LABELS_KEY:
    fpl = result.get(constants.FEATURES_PREDICTIONS_LABELS_KEY)
    if not fpl:
      raise RuntimeError('FPL missing. Ensure PredictExtractor was called.')

    if not isinstance(fpl, types.FeaturesPredictionsLabels):
      raise TypeError(
          'Expected FPL to be instance of FeaturesPredictionsLabel. FPL was: %s'
          'of type %s' % (str(fpl), type(fpl)))

    # We disable pytyping here because we know that 'fpl' key corresponds to a
    # non-materialized column.
    # pytype: disable=attribute-error
    _AugmentExtracts(fpl.features, constants.FEATURES_KEY, excludes, result)
    _AugmentExtracts(fpl.predictions, constants.PREDICTIONS_KEY, excludes,
                     result)
    _AugmentExtracts(fpl.labels, constants.LABELS_KEY, excludes, result)
    # pytype: enable=attribute-error
    return result
  elif source == constants.INPUT_KEY:
    serialized_example = result.get(constants.INPUT_KEY)
    if not serialized_example:
      raise RuntimeError('tf.Example missing. Ensure extracts contain '
                         'serialized tf.Example.')
    materialize_columns = (dest == constants.MATERIALIZE_COLUMNS)
    _ParseExample(result, materialize_columns)
    return result
  else:
    raise RuntimeError('Unsupported feature extraction source.')


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ExtractFeatures(
    extracts: beam.pvalue.PCollection,
    additional_extracts: Optional[List[Text]] = None,
    excludes: Optional[List[bytes]] = None,
    source: Text = constants.FEATURES_PREDICTIONS_LABELS_KEY,
    dest: Text = constants.MATERIALIZE_COLUMNS) -> beam.pvalue.PCollection:
  """Builds MaterializedColumn extracts from FPL created in evaluate.Predict().

  It must be the case that the PredictExtractor was called before calling this
  function.

  Args:
    extracts: PCollection containing the Extracts that will have
      MaterializedColumn added to.
    additional_extracts: Optional list of additional extracts to include along
      with the features, predictions, and labels.
    excludes: Optional list of strings containing features, predictions, or
      labels to exclude from materialization.
    source: Source for extracting features. Currently it supports extracting
      features from FPLs and input tf.Example protos.
    dest: Destination for extracted features. Currently supported are adding
      materialized columns, or the features dict of the FPLs.

  Returns:
    PCollection of Extracts
  """
  return extracts | 'MaterializeFeatures' >> beam.Map(
      _MaterializeFeatures,
      additional_extracts=additional_extracts,
      excludes=excludes,
      source=source,
      dest=dest)
