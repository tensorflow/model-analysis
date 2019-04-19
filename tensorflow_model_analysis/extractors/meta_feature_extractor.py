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
"""Extractor for creating new features from existing features.

For example usage, see the tests associated with this file.
"""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import copy

# Standard Imports

import apache_beam as beam
import numpy as np
import tensorflow as tf

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import encoding
from typing import Any, Callable, Text


def get_feature_value(fpl: types.FeaturesPredictionsLabels,
                      feature_key: Text) -> Any:
  """Helper to get value from FPL dict."""
  node_value = fpl.features[feature_key][encoding.NODE_SUFFIX]
  if isinstance(node_value, tf.compat.v1.SparseTensorValue):
    return node_value.values
  return node_value


def _set_feature_value(features: types.DictOfFetchedTensorValues,
                       feature_key: Text,
                       feature_value: Any) -> types.DictOfFetchedTensorValues:
  """Helper to set feature in FPL dict."""
  if not isinstance(feature_value, np.ndarray) and not isinstance(
      feature_value, tf.compat.v1.SparseTensorValue):
    feature_value = np.array([feature_value])
  features[feature_key] = {encoding.NODE_SUFFIX: feature_value}
  return features  # pytype: disable=bad-return-type


def get_fpl_copy(extracts: types.Extracts) -> types.FeaturesPredictionsLabels:
  """Get a copy of the FPL in the extracts of extracts."""
  fpl_orig = extracts.get(constants.FEATURES_PREDICTIONS_LABELS_KEY)
  if not fpl_orig:
    raise RuntimeError('FPL missing, Please ensure _Predict() was called.')

  # We must make a copy of the FPL tuple as well, so that we don't mutate the
  # original which is disallowed by Beam.
  fpl_copy = types.FeaturesPredictionsLabels(
      features=copy.copy(fpl_orig.features),
      labels=fpl_orig.labels,
      predictions=fpl_orig.predictions,
      input_ref=fpl_orig.input_ref)
  return fpl_copy


def update_fpl_features(fpl: types.FeaturesPredictionsLabels,
                        new_features: types.DictOfFetchedTensorValues):
  """Add new features to the FPL."""
  for key, value in new_features.items():
    # if the key already exists in the dictionary, throw an error.
    if key in fpl.features:
      raise ValueError('Modification of existing keys is not allowed.')
    _set_feature_value(fpl.features, key, value)


def _ExtractMetaFeature(  # pylint: disable=invalid-name
    extracts: types.Extracts,
    new_features_fn: Callable[[types.FeaturesPredictionsLabels], types
                              .DictOfFetchedTensorValues]) -> types.Extracts:
  """Augments FPL dict with new feature(s)."""
  # Create a new feature from existing ones.
  fpl_copy = get_fpl_copy(extracts)
  new_features = new_features_fn(fpl_copy)

  # Add the new features to the existing ones.
  update_fpl_features(fpl_copy, new_features)

  result = copy.copy(extracts)
  result[constants.FEATURES_PREDICTIONS_LABELS_KEY] = fpl_copy
  return result


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def ExtractMetaFeature(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    new_features_fn: Callable[[types.FeaturesPredictionsLabels], types
                              .DictOfFetchedTensorValues]
) -> beam.pvalue.PCollection:
  """Extracts meta-features derived from existing features.

  It must be the case that the PredictExtractor was called before calling this
  function.

  Args:
    extracts: PCollection containing the Extracts that will have
      MaterializedColumn added to its extracts.
    new_features_fn: A function that adds new features. Must take a
      FeaturesPredictionsLabel tuple as an argument, and return a a dict of new
      features to add, where the keys are new feature names and the values are
      the associated values.Only adding new features is permitted to prevent
      inadvertently removing useful data.

  Returns:
    PCollection of Extracts
  """
  return (
      extracts
      | 'ExtractMetaFeature' >> beam.Map(_ExtractMetaFeature, new_features_fn))
