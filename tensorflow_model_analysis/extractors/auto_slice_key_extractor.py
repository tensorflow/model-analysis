# Copyright 2019 Google LLC
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
"""Public API for Auto Slicing."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import itertools

from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.slicer import slicer_lib as slicer

from typing import List, Optional

from tensorflow_metadata.proto.v0 import statistics_pb2

SLICE_KEY_EXTRACTOR_STAGE_NAME = 'AutoExtractSliceKeys'


def AutoSliceKeyExtractor(
    statistics: statistics_pb2.DatasetFeatureStatisticsList,
    materialize: Optional[bool] = True
) -> extractor.Extractor:
  """Creates an extractor for automatically extracting slice keys.

  The incoming Extracts must contain a FeaturesPredictionsLabels extract keyed
  by tfma.FEATURES_PREDICTIONS_LABELS_KEY. Typically this will be obtained by
  calling the PredictExtractor.

  The extractor's PTransform yields a copy of the Extracts input with an
  additional extract pointing at the list of SliceKeyType values keyed by
  tfma.SLICE_KEY_TYPES_KEY. If materialize is True then a materialized version
  of the slice keys will be added under the key tfma.MATERIALZED_SLICE_KEYS_KEY.

  Args:
    statistics: Data statistics.
    materialize: True to add MaterializedColumn entries for the slice keys.

  Returns:
    Extractor for slice keys.
  """
  slice_spec = slice_spec_from_stats(statistics)
  return extractor.Extractor(
      stage_name=SLICE_KEY_EXTRACTOR_STAGE_NAME,
      ptransform=slice_key_extractor._ExtractSliceKeys(slice_spec, materialize))  # pylint: disable=protected-access


# TODO(pachristopher): Slice numeric features based on quantile buckets.
def slice_spec_from_stats(  # pylint: disable=invalid-name
    statistics: statistics_pb2.DatasetFeatureStatisticsList,
    categorical_uniques_threshold: int = 100,
    max_cross_size: int = 2
) -> List[slicer.SingleSliceSpec]:
  """Generates slicing spec from statistics.

  Args:
    statistics: Data statistics.
    categorical_uniques_threshold: Maximum number of unique values beyond which
      we don't slice on that categorical feature.
    max_cross_size: Maximum size feature crosses to consider.

  Returns:
    List of slice specs.
  """
  columns_to_consider = []
  for feature in statistics.datasets[0].features:
    if len(feature.path.step) != 1:
      continue
    stats_type = feature.WhichOneof('stats')
    if stats_type == 'string_stats':
      # TODO(pachristopher): Consider slicing on top-K values for features
      # with high cardinality.
      if 0 < feature.string_stats.unique <= categorical_uniques_threshold:
        columns_to_consider.append(feature)

  result = []
  for i in range(1, max_cross_size+1):
    for cross in itertools.combinations(columns_to_consider, i):
      result.append(slicer.SingleSliceSpec(
          columns=[feature.path.step[0] for feature in cross]))
  result.append(slicer.SingleSliceSpec())
  return result
