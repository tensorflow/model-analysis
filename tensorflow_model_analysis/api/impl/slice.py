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
from tensorflow_model_analysis.eval_saved_model import load
from tensorflow_model_analysis.slicer import slicer
from tensorflow_model_analysis.types_compat import Generator, List, Tuple


# For use in Beam type annotations, because Beam's support for Python types
# in Beam type annotations is not complete.
_BeamSliceKeyType = beam.typehints.Tuple[  # pylint: disable=invalid-name
    beam.typehints.Tuple[bytes, beam.typehints.Union[bytes, int, float]], Ellipsis]

_METRICS_NAMESPACE = 'tensorflow_model_analysis'


@beam.typehints.with_input_types(beam.typehints.Any)
@beam.typehints.with_output_types(
    beam.typehints.Tuple[_BeamSliceKeyType, beam.typehints.Any])
class _SliceDoFn(beam.DoFn):
  """A DoFn that performs slicing."""

  def __init__(self, slice_spec):
    self._slice_spec = slice_spec
    self._num_slices_generated_per_instance = beam.metrics.Metrics.distribution(
        _METRICS_NAMESPACE, 'num_slices_generated_per_instance')
    self._post_slice_num_instances = beam.metrics.Metrics.counter(
        _METRICS_NAMESPACE, 'post_slice_num_instances')

  def process(self, element):
    fpl = element.extracts[constants.FEATURES_PREDICTIONS_LABELS_KEY]
    features = fpl.features
    slice_count = 0
    for slice_key in slicer.get_slices_for_features_dict(
        features, self._slice_spec):
      slice_count += 1
      yield (slice_key, fpl)

    self._num_slices_generated_per_instance.update(slice_count)
    self._post_slice_num_instances.inc(slice_count)


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.typehints.Any)
@beam.typehints.with_output_types(
    beam.typehints.Tuple[_BeamSliceKeyType, beam.typehints.Any])  # pylint: disable=invalid-name
def Slice(intro_result,
          slice_spec):
  return intro_result | beam.ParDo(_SliceDoFn(slice_spec))
