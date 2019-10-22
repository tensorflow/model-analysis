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
"""Public API for extracting slice keys."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import copy
# Standard Imports

import apache_beam as beam

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.slicer import slicer_lib as slicer

from typing import List, Optional

SLICE_KEY_EXTRACTOR_STAGE_NAME = 'ExtractSliceKeys'


def SliceKeyExtractor(slice_spec: Optional[List[slicer.SingleSliceSpec]] = None,
                      materialize: Optional[bool] = True
                     ) -> extractor.Extractor:
  """Creates an extractor for extracting slice keys.

  The incoming Extracts must contain a FeaturesPredictionsLabels extract keyed
  by tfma.FEATURES_PREDICTIONS_LABELS_KEY. Typically this will be obtained by
  calling the PredictExtractor.

  The extractor's PTransform yields a copy of the Extracts input with an
  additional extract pointing at the list of SliceKeyType values keyed by
  tfma.SLICE_KEY_TYPES_KEY. If materialize is True then a materialized version
  of the slice keys will be added under the key tfma.MATERIALZED_SLICE_KEYS_KEY.

  Args:
    slice_spec: Optional list of SingleSliceSpec specifying the slices to slice
      the data into. If None, defaults to the overall slice.
    materialize: True to add MaterializedColumn entries for the slice keys.

  Returns:
    Extractor for slice keys.
  """
  if not slice_spec:
    slice_spec = [slicer.SingleSliceSpec()]
  return extractor.Extractor(
      stage_name=SLICE_KEY_EXTRACTOR_STAGE_NAME,
      ptransform=_ExtractSliceKeys(slice_spec, materialize))


@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
class _ExtractSliceKeysFn(beam.DoFn):
  """A DoFn that extracts slice keys that apply per example."""

  def __init__(self, slice_spec: List[slicer.SingleSliceSpec],
               materialize: bool):
    self._slice_spec = slice_spec
    self._materialize = materialize

  def process(self, element: types.Extracts) -> List[types.Extracts]:
    features = None
    if constants.FEATURES_PREDICTIONS_LABELS_KEY in element:
      fpl = element[constants.FEATURES_PREDICTIONS_LABELS_KEY]
      if not isinstance(fpl, types.FeaturesPredictionsLabels):
        raise TypeError(
            'Expected FPL to be instance of FeaturesPredictionsLabel. FPL was: '
            '%s of type %s' % (str(fpl), type(fpl)))
      features = fpl.features
    elif constants.FEATURES_KEY in element:
      features = element[constants.FEATURES_KEY]
    if not features:
      raise RuntimeError(
          'Features missing, Please ensure Predict() was called.')
    slices = list(
        slicer.get_slices_for_features_dict(features, self._slice_spec))

    # Make a a shallow copy, so we don't mutate the original.
    element_copy = copy.copy(element)

    element_copy[constants.SLICE_KEY_TYPES_KEY] = slices
    # Add a list of stringified slice keys to be materialized to output table.
    if self._materialize:
      element_copy[constants.SLICE_KEYS_KEY] = types.MaterializedColumn(
          name=constants.SLICE_KEYS_KEY,
          value=(list(
              slicer.stringify_slice_key(x).encode('utf-8') for x in slices)))
    return [element_copy]


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _ExtractSliceKeys(extracts: beam.pvalue.PCollection,
                      slice_spec: List[slicer.SingleSliceSpec],
                      materialize: bool = True) -> beam.pvalue.PCollection:
  return extracts | beam.ParDo(_ExtractSliceKeysFn(slice_spec, materialize))
