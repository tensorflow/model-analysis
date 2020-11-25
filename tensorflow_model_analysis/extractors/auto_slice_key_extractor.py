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

from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Text, Tuple, Union

import apache_beam as beam
import numpy as np
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis import util
from tensorflow_model_analysis.extractors import extractor
from tensorflow_model_analysis.slicer import slicer_lib as slicer

from tensorflow_metadata.proto.v0 import statistics_pb2

SLICE_KEY_EXTRACTOR_STAGE_NAME = 'AutoExtractSliceKeys'
TRANSFORMED_FEATURE_PREFIX = 'transformed_'


def AutoSliceKeyExtractor(  # pylint: disable=invalid-name
    statistics: Union[beam.pvalue.PCollection,
                      statistics_pb2.DatasetFeatureStatisticsList],
    categorical_uniques_threshold: int = 100,
    max_cross_size: int = 2,
    allowlist_features: Optional[Set[Text]] = None,
    denylist_features: Optional[Set[Text]] = None,
    materialize: bool = True) -> extractor.Extractor:
  """Creates an extractor for automatically extracting slice keys.

  The incoming Extracts must contain a FeaturesPredictionsLabels extract keyed
  by tfma.FEATURES_PREDICTIONS_LABELS_KEY. Typically this will be obtained by
  calling the PredictExtractor.

  The extractor's PTransform yields a copy of the Extracts input with an
  additional extract pointing at the list of SliceKeyType values keyed by
  tfma.SLICE_KEY_TYPES_KEY. If materialize is True then a materialized version
  of the slice keys will be added under the key tfma.MATERIALZED_SLICE_KEYS_KEY.

  Args:
    statistics: PCollection of data statistics proto or actual data statistics
      proto. Note that when passed a PCollection, it would be matrialized and
      passed as a side input.
    categorical_uniques_threshold: Maximum number of unique values beyond which
      we don't slice on that categorical feature.
    max_cross_size: Maximum size feature crosses to consider.
    allowlist_features: Set of features to be used for slicing.
    denylist_features: Set of features to ignore for slicing.
    materialize: True to add MaterializedColumn entries for the slice keys.

  Returns:
    Extractor for slice keys.
  """
  assert not allowlist_features or not denylist_features

  return extractor.Extractor(
      stage_name=SLICE_KEY_EXTRACTOR_STAGE_NAME,
      ptransform=_AutoExtractSliceKeys(statistics,
                                       categorical_uniques_threshold,
                                       max_cross_size, allowlist_features,
                                       denylist_features, materialize))


def get_quantile_boundaries(
    statistics: statistics_pb2.DatasetFeatureStatisticsList
) -> Dict[Text, List[float]]:
  """Get quantile bucket boundaries from statistics proto."""
  result = {}
  for feature in _get_slicable_numeric_features(
      list(statistics.datasets[0].features)):
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


def get_bucket_boundary(bucket: int,
                        boundaries: List[float]) -> Tuple[float, float]:
  """Given bucket index, return the bucket boundary.

  Note that the bucket index was computed using bisect_left [1].

  [1] https://docs.python.org/3/library/bisect.html#bisect.bisect_left

  Args:
    bucket: Bucket index.
    boundaries: List of boundaries.

  Returns:
    A tuple (start, end] representing bucket boundary.
  """
  if bucket == len(boundaries):
    end = float('inf')
  else:
    end = boundaries[bucket]
  if bucket == 0:
    start = float('-inf')
  else:
    start = boundaries[bucket - 1]
  return (start, end)


@beam.typehints.with_input_types(types.Extracts,
                                 statistics_pb2.DatasetFeatureStatisticsList)
@beam.typehints.with_output_types(types.Extracts)
class _BucketizeNumericFeaturesFn(beam.DoFn):
  """A DoFn that bucketizes numeric features using the quantiles."""

  def __init__(self):
    self._bucket_boundaries = None

  def process(
      self, element: types.Extracts,
      statistics: statistics_pb2.DatasetFeatureStatisticsList
  ) -> List[types.Extracts]:
    if self._bucket_boundaries is None:
      self._bucket_boundaries = get_quantile_boundaries(statistics)
    # Make a deep copy, so we don't mutate the original.
    element_copy = copy.deepcopy(element)
    features = util.get_features_from_extracts(element_copy)
    for feature_name, boundaries in self._bucket_boundaries.items():
      if (feature_name in features and features[feature_name] is not None and
          features[feature_name].size > 0):
        transformed_values = []
        for value in features[feature_name]:
          transformed_values.append(_bin_value(value, boundaries))
        features[TRANSFORMED_FEATURE_PREFIX +
                 feature_name] = np.array(transformed_values)
    return [element_copy]


@beam.typehints.with_input_types(types.Extracts, List[slicer.SingleSliceSpec])
@beam.typehints.with_output_types(types.Extracts)
class _ExtractSliceKeysFn(beam.DoFn):
  """A DoFn that extracts slice keys that apply per example.

  This is a fork of slice_key_extractor.ExtractSliceKeys but taking the
  slice_spec as a side input.
  """

  def __init__(self, materialize: bool):
    self._materialize = materialize

  def process(self, element: types.Extracts,
              slice_spec: List[slicer.SingleSliceSpec]) -> List[types.Extracts]:
    features = util.get_features_from_extracts(element)
    # There are no transformed features so only search raw features for slices.
    slices = list(
        slicer.get_slices_for_features_dicts([], features, slice_spec))

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


@beam.typehints.with_input_types(Any,
                                 statistics_pb2.DatasetFeatureStatisticsList)
@beam.typehints.with_output_types(List[slicer.SingleSliceSpec])
class _SliceSpecFromStatsFn(beam.DoFn):
  """A DoFn that creates slice spec from statistics."""

  def __init__(self, categorical_uniques_threshold: int, max_cross_size: int,
               allowlist_features: Optional[Set[Text]],
               denylist_features: Optional[Set[Text]]):
    self._categorical_uniques_threshold = categorical_uniques_threshold
    self._max_cross_size = max_cross_size
    self._allowlist_features = allowlist_features
    self._denylist_features = denylist_features

  def process(
      self, unused_element: Any,
      statistics: statistics_pb2.DatasetFeatureStatisticsList
  ) -> Iterable[List[slicer.SingleSliceSpec]]:
    yield slice_spec_from_stats(
        statistics,
        categorical_uniques_threshold=self._categorical_uniques_threshold,
        max_cross_size=self._max_cross_size,
        allowlist_features=self._allowlist_features,
        denylist_features=self._denylist_features)


@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
class _AutoExtractSliceKeys(beam.PTransform):
  """Extract slice keys."""

  def __init__(self,
               statistics: Union[beam.pvalue.PCollection,
                                 statistics_pb2.DatasetFeatureStatisticsList],
               categorical_uniques_threshold: int = 100,
               max_cross_size: int = 2,
               allowlist_features: Optional[Set[Text]] = None,
               denylist_features: Optional[Set[Text]] = None,
               materialize: bool = True) -> beam.pvalue.PCollection:
    if isinstance(statistics, beam.pvalue.PCollection):
      self._statistics = beam.pvalue.AsSingleton(statistics)
    else:
      self._statistics = statistics
    self._categorical_uniques_threshold = categorical_uniques_threshold
    self._max_cross_size = max_cross_size
    self._allowlist_features = allowlist_features
    self._denylist_features = denylist_features
    self._materialize = materialize

  def expand(self, extracts):
    slice_spec = (
        extracts.pipeline
        | beam.Create([None])
        | 'SliceSpecFromStats' >> beam.ParDo(
            _SliceSpecFromStatsFn(
                categorical_uniques_threshold=self
                ._categorical_uniques_threshold,
                max_cross_size=self._max_cross_size,
                allowlist_features=self._allowlist_features,
                denylist_features=self._denylist_features,
            ),
            statistics=self._statistics))
    return (extracts
            | 'BucketizeNumericFeatures' >> beam.ParDo(
                _BucketizeNumericFeaturesFn(), statistics=self._statistics)
            | 'ExtractSliceKeys' >> beam.ParDo(
                _ExtractSliceKeysFn(materialize=self._materialize),
                slice_spec=beam.pvalue.AsSingleton(slice_spec)))


def _get_slicable_numeric_features(
    features: List[statistics_pb2.FeatureNameStatistics]
) -> Iterator[statistics_pb2.FeatureNameStatistics]:
  """Get numeric features to slice on."""
  for feature in features:
    stats_type = feature.WhichOneof('stats')
    if stats_type == 'num_stats':
      # Ignore features which have the same value in all examples.
      if feature.num_stats.min == feature.num_stats.max:
        continue
      yield feature


def _get_slicable_categorical_features(
    features: List[statistics_pb2.FeatureNameStatistics],
    categorical_uniques_threshold: int = 100,
) -> Iterator[statistics_pb2.FeatureNameStatistics]:
  """Get categorical features to slice on."""
  for feature in features:
    stats_type = feature.WhichOneof('stats')
    if stats_type == 'string_stats':
      # TODO(pachristopher): Consider slicing on top-K values for features
      # with high cardinality.
      if 1 < feature.string_stats.unique <= categorical_uniques_threshold:
        yield feature


# TODO(pachristopher): Slice numeric features based on quantile buckets.
def slice_spec_from_stats(  # pylint: disable=invalid-name
    statistics: statistics_pb2.DatasetFeatureStatisticsList,
    categorical_uniques_threshold: int = 100,
    max_cross_size: int = 2,
    allowlist_features: Optional[Set[Text]] = None,
    denylist_features: Optional[Set[Text]] = None) -> List[
        slicer.SingleSliceSpec]:
  """Generates slicing spec from statistics.

  Args:
    statistics: Data statistics.
    categorical_uniques_threshold: Maximum number of unique values beyond which
      we don't slice on that categorical feature.
    max_cross_size: Maximum size feature crosses to consider.
    allowlist_features: Set of features to be used for slicing.
    denylist_features: Set of features to ignore for slicing.

  Returns:
    List of slice specs.
  """
  features_to_consider = []
  for feature in statistics.datasets[0].features:
    # TODO(pachristopher): Consider structured features once TFMA supports
    # slicing on structured features.
    if (len(feature.path.step) != 1 or
        (allowlist_features and feature.path.step[0] not in allowlist_features)
        or (denylist_features and feature.path.step[0] in denylist_features)):
      continue
    features_to_consider.append(feature)

  slicable_column_names = []
  for feature in _get_slicable_categorical_features(
      features_to_consider, categorical_uniques_threshold):
    slicable_column_names.append(feature.path.step[0])
  for feature in _get_slicable_numeric_features(features_to_consider):
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
