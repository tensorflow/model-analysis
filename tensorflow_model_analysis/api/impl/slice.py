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

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.api.impl import api_types
from tensorflow_model_analysis.slicer import slicer
from tensorflow_model_analysis.types_compat import Generator, Text, Tuple

# For use in Beam type annotations, because Beam's support for Python types
# in Beam type annotations is not complete.
_BeamSliceKeyType = beam.typehints.Tuple[  # pylint: disable=invalid-name
    beam.typehints.Tuple[Text, beam.typehints.Union[bytes, int, float]], Ellipsis]

_METRICS_NAMESPACE = 'tensorflow_model_analysis'


@beam.typehints.with_input_types(beam.typehints.Any)
@beam.typehints.with_output_types(
    beam.typehints.Tuple[_BeamSliceKeyType, api_types.FeaturesPredictionsLabels]
)
class _FanoutSlicesDoFn(beam.DoFn):
  """A DoFn that performs per-slice key fanout prior to computing aggregates."""

  def __init__(self):
    self._num_slices_generated_per_instance = beam.metrics.Metrics.distribution(
        _METRICS_NAMESPACE, 'num_slices_generated_per_instance')
    self._post_slice_num_instances = beam.metrics.Metrics.counter(
        _METRICS_NAMESPACE, 'post_slice_num_instances')

  def process(self, element
             ):
    fpl = element.extracts.get(constants.FEATURES_PREDICTIONS_LABELS_KEY)
    if not fpl:
      raise RuntimeError('FPL missing, Please ensure Predict() was called.')
    if not isinstance(fpl, api_types.FeaturesPredictionsLabels):
      raise TypeError(
          'Expected FPL to be instance of FeaturesPredictionsLabel. FPL was: '
          '%s of type %s' % (str(fpl), type(fpl)))

    slices = element.extracts.get(constants.SLICE_KEYS)

    slice_count = 0
    for slice_key in slices:
      slice_count += 1
      yield (slice_key, fpl)

    self._num_slices_generated_per_instance.update(slice_count)
    self._post_slice_num_instances.inc(slice_count)


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.typehints.Any)
@beam.typehints.with_output_types(
    beam.typehints.Tuple[_BeamSliceKeyType, api_types.FeaturesPredictionsLabels]
)  # pylint: disable=invalid-name
def FanoutSlices(pcoll):
  """Fan out examples based on the slice keys."""
  result = pcoll | 'DoSlicing' >> beam.ParDo(_FanoutSlicesDoFn())


  return result
