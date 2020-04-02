# Lint as: python3
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

import bisect
import copy
import itertools

from typing import Dict, List, Optional, Text, Tuple

import apache_beam as beam
import numpy as np
from tensorflow_model_analysis import types
from tensorflow_model_analysis import util
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.slicer import slicer_lib as slicer

from tensorflow_metadata.proto.v0 import statistics_pb2

SLICE_KEY_EXTRACTOR_STAGE_NAME = 'AutoExtractSliceKeys'
TRANSFORMED_FEATURE_PREFIX = 'transformed_'


def AutoSliceKeyExtractor(  # pylint: disable=invalid-name
    statistics: statistics_pb2.DatasetFeatureStatisticsList,
    materialize: Optional[bool] = True) -> extractor.Extractor:
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
      ptransform=_AutoExtractSliceKeys(slice_spec, statistics, materialize))


def _get_quantile_boundaries(
    statistics: statistics_pb2.DatasetFeatureStatisticsList
) -> Dict[Text, List[float]]:
  """Get quantile bucket boundaries from statistics proto."""
  result = {}
  for feature in _get_slicable_numeric_features(statistics):
    boundaries = None
    for histogram in feature.num_stats.histograms:
      if histogram.type == statistics_pb2.Histogram.QUANTILES:
        boundaries = [bucket.low_value for bucket in histogram.buckets]
        boundaries.append(histogram.buckets[-1].high_value)
        break
    assert boundaries is not None
    result[feature.path.step[0]] = boundaries
  return result


def _bin_value(value: float, boundaries: List[float]) -> int:
  """Bin value based the bucket boundaries."""
  return bisect.bisect_left(boundaries, value)


def _get_bucket_boundary(bucket: int,
                         boundaries: List[float]) -> Tuple[float, float]:
  """Given bucket index, return the bucket boundary."""
  if bucket == len(boundaries):
    end = float('inf')
  else:
    end = boundaries[bucket]
  if bucket == 0:
    start = float('-inf')
  else:
    start = boundaries[bucket - 1]
  return (start, end)


@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
class _BucketizeNumericFeaturesFn(beam.DoFn):
  """A DoFn that extracts slice keys that apply per example."""

  def __init__(self, statistics: statistics_pb2.DatasetFeatureStatisticsList):
    # Get bucket boundaries for numeric features
    self._bucket_boundaries = _get_quantile_boundaries(statistics)

  def process(self, element: types.Extracts) -> List[types.Extracts]:
    # Make a a shallow copy, so we don't mutate the original.
    element_copy = copy.copy(element)
    features = util.get_features_from_extracts(element_copy)
    for feature_name, boundaries in self._bucket_boundaries.items():
      if feature_name in features:
        transformed_values = []
        for value in features[feature_name]:
          transformed_values.append(_bin_value(value, boundaries))
        features[TRANSFORMED_FEATURE_PREFIX +
                 feature_name] = np.array(transformed_values)
    return [element_copy]


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def _AutoExtractSliceKeys(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    slice_spec: List[slicer.SingleSliceSpec],
    statistics: statistics_pb2.DatasetFeatureStatisticsList,
    materialize: bool = True) -> beam.pvalue.PCollection:
  return (extracts
          | 'BucketizeNumericFeatures' >> beam.ParDo(
              _BucketizeNumericFeaturesFn(statistics))
          | 'ExtractSliceKeys' >> slice_key_extractor.ExtractSliceKeys(
              slice_spec, materialize))


def _get_slicable_numeric_features(
    statistics: statistics_pb2.DatasetFeatureStatisticsList
) -> List[statistics_pb2.FeatureNameStatistics]:
  """Get numeric features to slice on."""
  result = []
  for feature in statistics.datasets[0].features:
    if len(feature.path.step) != 1:
      continue
    stats_type = feature.WhichOneof('stats')
    if stats_type == 'num_stats':
      # Ignore features which have the same value in all examples.
      if feature.num_stats.min == feature.num_stats.max:
        continue
      result.append(feature)
  return result


def _get_slicable_categorical_features(
    statistics: statistics_pb2.DatasetFeatureStatisticsList,
    categorical_uniques_threshold: int = 100,
) -> List[statistics_pb2.FeatureNameStatistics]:
  """Get categorical features to slice on."""
  result = []
  for feature in statistics.datasets[0].features:
    # TODO(pachristopher): Consider structured features once TFMA supports
    # slicing on structured features.
    if len(feature.path.step) != 1:
      continue
    stats_type = feature.WhichOneof('stats')
    if stats_type == 'string_stats':
      # TODO(pachristopher): Consider slicing on top-K values for features
      # with high cardinality.
      if 1 < feature.string_stats.unique <= categorical_uniques_threshold:
        result.append(feature)
  return result


# TODO(pachristopher): Slice numeric features based on quantile buckets.
def slice_spec_from_stats(  # pylint: disable=invalid-name
    statistics: statistics_pb2.DatasetFeatureStatisticsList,
    categorical_uniques_threshold: int = 100,
    max_cross_size: int = 2) -> List[slicer.SingleSliceSpec]:
  """Generates slicing spec from statistics.

  Args:
    statistics: Data statistics.
    categorical_uniques_threshold: Maximum number of unique values beyond which
      we don't slice on that categorical feature.
    max_cross_size: Maximum size feature crosses to consider.

  Returns:
    List of slice specs.
  """
  slicable_column_names = []
  for feature in _get_slicable_categorical_features(
      statistics, categorical_uniques_threshold):
    slicable_column_names.append(feature.path.step[0])
  for feature in _get_slicable_numeric_features(statistics):
    # We would bucketize the feature based on the quantiles boundaries.
    slicable_column_names.append(TRANSFORMED_FEATURE_PREFIX +
                                 feature.path.step[0])

  result = []
  for i in range(1, max_cross_size + 1):
    for cross in itertools.combinations(slicable_column_names, i):
      result.append(
          slicer.SingleSliceSpec(
              columns=[feature_name for feature_name in cross]))
  result.append(slicer.SingleSliceSpec())
  return result
