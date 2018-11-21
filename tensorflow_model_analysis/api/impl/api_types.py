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
"""Types used in the public API."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import apache_beam as beam
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.slicer import slicer

from tensorflow_model_analysis.types_compat import Any, Dict, List, NamedTuple, Optional, Text, Tuple

EvalConfig = NamedTuple(  # pylint: disable=invalid-name
    'EvalConfig',
    [
        ('model_location',
         Text),  # The location of the model used for this evaluation
        ('data_location',
         Text),  # The location of the data used for this evaluation
        ('slice_spec', Optional[List[slicer.SingleSliceSpec]]
        ),  # The corresponding slice spec
        ('example_weight_metric_key',
         Text),  # The name of the metric that contains example weight
    ])

EvalResult = NamedTuple(  # pylint: disable=invalid-name
    'EvalResult',
    [('slicing_metrics', List[Tuple[slicer.SliceKeyType, Dict[Text, Any]]]),
     ('plots', List[Tuple[slicer.SliceKeyType, Dict[Text, Any]]]),
     ('config', EvalConfig)])

SUPPORTED_MODES = [
    constants.DATA_CENTRIC_MODE,
    constants.MODEL_CENTRIC_MODE,
]

Extractor = NamedTuple(  # pylint: disable=invalid-name
    'Extractor', [('stage_name', Text), ('ptransform', beam.PTransform)])

FeaturesPredictionsLabels = NamedTuple(  # pylint: disable=invalid-name
    'FeaturesPredictionsLabels',
    [('example_ref', int), ('features', types.DictOfFetchedTensorValues),
     ('predictions', types.DictOfFetchedTensorValues),
     ('labels', types.DictOfFetchedTensorValues)])


class EvalResults(object):
  """Class for results from multiple model analysis run."""

  def __init__(self,
               results,
               mode = constants.UNKNOWN_EVAL_MODE):
    if mode not in SUPPORTED_MODES:
      raise ValueError('Mode ' + mode + ' must be one of ' +
                       Text(SUPPORTED_MODES))

    self._results = results
    self._mode = mode

  def get_results(self):
    return self._results

  def get_mode(self):
    return self._mode
