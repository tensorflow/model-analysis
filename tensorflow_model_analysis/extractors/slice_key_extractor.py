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

from __future__ import print_function



import apache_beam as beam

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.api.impl import api_types
from tensorflow_model_analysis.slicer import slicer

from tensorflow_model_analysis.types_compat import List, Optional


def SliceKeyExtractor(
    slice_spec = None,
    materialize = True):
  """Creates an extractor for extracting slice keys.

  The incoming ExampleAndExtracts must contain a FeaturesPredictionsLabels
  extract keyed by 'fpl'. Typically this will be obtained by calling the
  PredictExtractor.

  The extractor's PTransform yields a copy of the ExampleAndExtracts input with
  an additional 'slice_keys' extract pointing at the list of SliceKeyType
  values. If materialize is True then a materialized version of the slice keys
  will be added under the key 'materialized_slice_keys'.

  Args:
    slice_spec: Optional list of SingleSliceSpec specifying the slices to slice
      the data into. If None, defaults to the overall slice.
    materialize: True to add MaterializedColumn entries for the slice keys.

  Returns:
    Extractor for slice keys.
  """
  if slice_spec is None:
    slice_spec = [slicer.SingleSliceSpec()]
  return api_types.Extractor(
      stage_name='ExtractSliceKeys',
      ptransform=ExtractSliceKeys(slice_spec, materialize))


@beam.typehints.with_input_types(beam.typehints.Any)
@beam.typehints.with_output_types(beam.typehints.Any)
class _ExtractSliceKeys(beam.DoFn):
  """A DoFn that extracts slice keys that apply per example."""

  def __init__(self, slice_spec,
               materialize):
    self._slice_spec = slice_spec
    self._materialize = materialize

  def process(self, element
             ):
    fpl = element.extracts.get(constants.FEATURES_PREDICTIONS_LABELS_KEY)
    if not fpl:
      raise RuntimeError('FPL missing, Please ensure Predict() was called.')
    if not isinstance(fpl, api_types.FeaturesPredictionsLabels):
      raise TypeError(
          'Expected FPL to be instance of FeaturesPredictionsLabel. FPL was: '
          '%s of type %s' % (str(fpl), type(fpl)))
    features = fpl.features
    slices = list(
        slicer.get_slices_for_features_dict(features, self._slice_spec))

    # Make a a shallow copy, so we don't mutate the original.
    element_copy = (element.create_copy_with_shallow_copy_of_extracts())

    element_copy.extracts[constants.SLICE_KEYS] = slices
    # Add a list of stringified slice keys to be materialized to output table.
    if self._materialize:
      element_copy.extracts[
          constants.SLICE_KEYS_MATERIALIZED] = types.MaterializedColumn(
              name=constants.SLICE_KEYS_MATERIALIZED,
              value=(list(
                  slicer.stringify_slice_key(x).encode('utf-8')
                  for x in slices)))
    return [element_copy]


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.typehints.Any)
@beam.typehints.with_output_types(beam.typehints.Any)
def ExtractSliceKeys(examples_and_extracts,
                     slice_spec,
                     materialize = True):
  return examples_and_extracts | beam.ParDo(
      _ExtractSliceKeys(slice_spec, materialize))
