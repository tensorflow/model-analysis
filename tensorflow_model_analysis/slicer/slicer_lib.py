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
"""Slicer library.

Use this library for generating slices from a specification
(List[SingleSliceSpec]) and input features.
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import itertools

from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Text, Tuple, Union

import apache_beam as beam
import numpy as np
import six
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer import slice_accessor

# FeatureValueType represents a value that a feature could take.
FeatureValueType = Union[Text, int, float]  # pylint: disable=invalid-name

# SingletonSliceKeyType is a tuple, where the first element is the key of the
# feature, and the second element is its value. This describes a single
# feature-value pair.
SingletonSliceKeyType = Tuple[Text, FeatureValueType]  # pylint: disable=invalid-name

# SliceKeyType is a either the empty tuple (for the overal slice) or a tuple of
# SingletonSliceKeyType. This completely describes a single slice.
SliceKeyType = Union[Tuple[()], Tuple[SingletonSliceKeyType, ...]]  # pylint: disable=invalid-name

# CrossSliceKeyType is a tuple, where first and second element is SliceKeyType.
CrossSliceKeyType = Tuple[SliceKeyType, SliceKeyType]  # pylint: disable=invalid-name

SliceKeyOrCrossSliceKeyType = Union[SliceKeyType, CrossSliceKeyType]  # pylint: disable=invalid-name

OVERALL_SLICE_NAME = 'Overall'
# The slice key for the slice that includes all of the data.
OVERALL_SLICE_KEY = ()


class SingleSliceSpec(object):
  """Specification for a single slice.

  This is intended to be an immutable class that specifies a single slice.
  Use this in conjunction with get_slices_for_features_dicts to generate slices
  for dictionaries of features.

  Examples:
    - columns = ['age'], features = []
      This means to slice by the 'age' column.
    - columns = ['age'], features = [('gender', 'female')]
      This means to slice by the 'age' column if the 'gender' is 'female'.
    - For more examples, refer to the tests in slicer_test.py.
  """

  def __eq__(self, other: 'SingleSliceSpec'):
    # Need access to other's protected fields for comparison.
    if isinstance(other, self.__class__):
      # pylint: disable=protected-access
      return (self._columns == other._columns and
              self._features == other._features)
      # pylint: enable=protected-access
    else:
      return False

  def __ne__(self, other: 'SingleSliceSpec'):
    return not self.__eq__(other)

  def __hash__(self):
    return hash((self._columns, self._features))

  def __init__(self,
               columns: Iterable[Text] = (),
               features: Iterable[Tuple[Text, FeatureValueType]] = (),
               spec: config.SlicingSpec = None):
    """Initialises a SingleSliceSpec.

    Args:
      columns: an iterable of column names to slice on.
      features: a iterable of features to slice on. Each feature is a (key,
        value) tuple. Note that strings representing ints and floats will be
        automatically converted to ints and floats respectively and will be
        compared against both the string versions and int or float versions of
        the associated features.
      spec: Initializes slicing spec from proto. If not None, overrides any
        values passed in columns or features.

    Raises:
      ValueError: There was overlap between the columns specified in columns
        and those in features.
      ValueError: columns or features was a string: they should probably be a
        singleton list containing that string.
    """
    if isinstance(columns, six.string_types):
      raise ValueError('columns is a string: it should probably be a singleton '
                       'list containing that string')
    if isinstance(features, six.string_types):
      raise ValueError('features is a string: it should probably be a '
                       'singleton list containing that string')

    if spec is not None:
      columns = spec.feature_keys
      features = [(k, v) for k, v in spec.feature_values.items()]

    features = [(k, _to_type(v)) for (k, v) in features]

    self._columns = frozenset(columns)
    self._features = frozenset(features)

    # We build this up as an instance variable, instead of building it each
    # time we call generate_slices, for efficiency reasons.
    #
    # This is a flat list of SingletonSliceKeyTypes,
    # i.e. List[SingletonSliceKeyType].
    self._value_matches = []

    for (key, value) in self._features:
      if not isinstance(value, six.string_types) and not isinstance(value, int):
        raise NotImplementedError('Only string and int values are supported '
                                  'as the slice value.')
      if key in self._columns:
        raise ValueError('Columns specified in columns and in features should '
                         'not overlap, but %s was specified in both.' % key)
      self._value_matches.append((key, value))
    self._value_matches = sorted(self._value_matches)

  def __repr__(self):
    return 'SingleSliceSpec(columns=%s, features=%s)' % (self._columns,
                                                         self._features)

  def to_proto(self) -> config.SlicingSpec:
    feature_values = {k: str(v) for (k, v) in self._features}
    return config.SlicingSpec(
        feature_keys=self._columns, feature_values=feature_values)

  def is_overall(self):
    """Returns True if this specification represents the overall slice."""
    return not self._columns and not self._features

  def is_slice_applicable(self, slice_key: SliceKeyType):
    """Determines if this slice spec is applicable to a slice of data.

    Args:
      slice_key: The slice as a SliceKeyType

    Returns:
      True if the slice_spec is applicable to the given slice, False otherwise.
    """
    columns = list(self._columns)
    features = list(self._features)
    for singleton_slice_key in slice_key:
      # Convert to internal representation of slice (i.e. str -> float, etc).
      if len(singleton_slice_key) == 2:
        singleton_slice_key = (singleton_slice_key[0],
                               _to_type(singleton_slice_key[1]))
      if singleton_slice_key in features:
        features.remove(singleton_slice_key)
      elif singleton_slice_key[0] in columns:
        columns.remove(singleton_slice_key[0])
      else:
        return False
    return not features and not columns

  def generate_slices(
      self, accessor: slice_accessor.SliceAccessor
  ) -> Generator[SliceKeyType, None, None]:
    """Generates all slices that match this specification from the data.

    Should only be called within this file.

    Examples:
      - columns = [], features = [] (the overall slice case)
        slice accessor has features age=[5], gender=['f'], interest=['knitting']
        returns [()]
      - columns = ['age'], features = [('gender', 'f')]
        slice accessor has features age=[5], gender=['f'], interest=['knitting']
        returns [[('age', 5), ('gender, 'f')]]
      - columns = ['interest'], features = [('gender', 'f')]
        slice accessor has features age=[5], gender=['f'],
        interest=['knitting', 'games']
        returns [[('gender', 'f'), ('interest, 'knitting')],
                 [('gender', 'f'), ('interest, 'games')]]

    Args:
      accessor: slice accessor.

    Yields:
      A SliceKeyType for each slice that matches this specification. Nothing
      will be yielded if there no slices matched this specification. The entries
      in the yielded SliceKeyTypes are guaranteed to be sorted by key names (and
      then values, if necessary), ascending.
    """
    # Check all the value matches (where there's a specific value specified).
    for (key, value) in self._features:
      if not accessor.has_key(key):
        return

      accessor_values = accessor.get(key)
      if value not in accessor_values:
        if isinstance(value, str):
          if value.encode() not in accessor_values:  # For Python3.
            return
        # Check that string version of int/float not in values.
        elif str(value) not in accessor_values:
          return

    # Get all the column matches (where we're matching only the column).
    #
    # For each column, we generate a List[SingletonSliceKeyType] containing
    # all pairs (column, value) for all values of the column. So this will be
    # a List[List[SingletonSliceKeyType]].
    #
    # For example, column_matches might be:
    # [[('gender', 'f'), ('gender', 'm')], [('age', 4), ('age', 5)]]
    column_matches = []
    for column in self._columns:
      # If a column to slice on doesn't appear in the example, then there will
      # be no applicable slices, so return.
      if not accessor.has_key(column):
        return

      column_match = []
      for value in accessor.get(column):
        if isinstance(value, bytes):
          column_match.append((column, tf.compat.as_text(value)))
        else:
          column_match.append((column, value))
      column_matches.append(column_match)

    # We can now take the Cartesian product of the column_matches, and append
    # the value matches to each element of that, to generate the final list of
    # slices. Note that for the overall slice case the column_matches is [] and
    # the Cartesian product of [] is ().
    for column_part in itertools.product(*column_matches):
      yield tuple(sorted(self._value_matches + list(column_part)))


def serialize_slice_key(
    slice_key: SliceKeyType) -> metrics_for_slice_pb2.SliceKey:
  """Converts SliceKeyType to SliceKey proto.

  Args:
    slice_key: The slice key in the format of SliceKeyType.

  Returns:
    The slice key in the format of SliceKey proto.

  Raises:
    TypeError: If the evaluate type is unrecognized.
  """
  result = metrics_for_slice_pb2.SliceKey()

  for (col, val) in slice_key:
    single_slice_key = result.single_slice_keys.add()
    single_slice_key.column = col
    if isinstance(val, (six.binary_type, six.text_type)):
      single_slice_key.bytes_value = tf.compat.as_bytes(val)
    elif isinstance(val, six.integer_types):
      single_slice_key.int64_value = val
    elif isinstance(val, float):
      single_slice_key.float_value = val
    else:
      raise TypeError('unrecognized type of type %s, value %s' %
                      (type(val), val))

  return result


def _to_type(v: FeatureValueType) -> FeatureValueType:
  """Converts string versions of ints and floats to respective values."""
  if isinstance(v, float) or isinstance(v, int):
    return v
  try:
    v = str(v)
    if '.' in v:
      return float(v)
    else:
      return int(v)
  except ValueError:
    return v


def serialize_cross_slice_key(
    cross_slice_key: CrossSliceKeyType) -> metrics_for_slice_pb2.CrossSliceKey:
  """Converts CrossSliceKeyType to CrossSliceKey proto."""
  result = metrics_for_slice_pb2.CrossSliceKey()
  baseline_slice_key, comparison_slice_key = cross_slice_key
  result.baseline_slice_key.CopyFrom(serialize_slice_key(baseline_slice_key))
  result.comparison_slice_key.CopyFrom(
      serialize_slice_key(comparison_slice_key))
  return result


def deserialize_slice_key(
    slice_key: metrics_for_slice_pb2.SliceKey) -> SliceKeyType:
  """Converts SliceKey proto to SliceKeyType.

  Args:
    slice_key: The slice key in the format of proto SliceKey.

  Returns:
    The slice key in the format of SliceKeyType.

  Raises:
    TypeError: If the evaluate type is unreconized.
  """
  result = []
  for elem in slice_key.single_slice_keys:
    if elem.HasField('bytes_value'):
      value = tf.compat.as_text(elem.bytes_value)
    elif elem.HasField('int64_value'):
      value = elem.int64_value
    elif elem.HasField('float_value'):
      value = elem.float_value
    else:
      raise TypeError('unrecognized type of type %s, value %s' %
                      (type(elem), elem))
    result.append((elem.column, value))
  return tuple(result)


def get_slices_for_features_dicts(
    features_dicts: Iterable[Union[types.DictOfTensorValue,
                                   types.DictOfFetchedTensorValues]],
    default_features_dict: Union[types.DictOfTensorValue,
                                 types.DictOfFetchedTensorValues],
    slice_spec: List[SingleSliceSpec]) -> Iterable[SliceKeyType]:
  """Generates the slice keys appropriate for the given features dictionaries.

  Args:
    features_dicts: Features dictionaries. For example a list of transformed
      features dictionaries.
    default_features_dict: Additional dict to search if a match is not found in
      features dictionaries. For example the raw features.
    slice_spec: slice specification.

  Yields:
    Slice keys appropriate for the given features dictionaries.
  """
  accessor = slice_accessor.SliceAccessor(features_dicts, default_features_dict)
  for single_slice_spec in slice_spec:
    for slice_key in single_slice_spec.generate_slices(accessor):
      yield slice_key


def stringify_slice_key(slice_key: SliceKeyType) -> Text:
  """Stringifies a slice key.

  The string representation of a SingletonSliceKeyType is "feature:value". When
  multiple columns / features are specified, the string representation of a
  SliceKeyType is "c1_X_c2_X_...:v1_X_v2_X_..." where c1, c2, ... are the column
  names and v1, v2, ... are the corresponding values For example,
  ('gender, 'f'), ('age', 5) befores age_X_gender:f_X_5. If no columns / feature
  specified, return "Overall".

  Note that we do not perform special escaping for slice values that contain
  '_X_'. This stringified representation is meant to be human-readbale rather
  than a reversible encoding.

  The columns will be in the same order as in SliceKeyType. If they are
  generated using SingleSliceSpec.generate_slices, they will be in sorted order,
  ascending.

  Technically float values are not supported, but we don't check for them here.

  Args:
    slice_key: Slice key to stringify. The constituent SingletonSliceKeyTypes
      should be sorted in ascending order.

  Returns:
    String representation of the slice key.
  """
  key_count = len(slice_key)
  if not key_count:
    return OVERALL_SLICE_NAME

  keys = []
  values = []
  separator = '_X_'

  for (feature, value) in slice_key:
    # Since this is meant to be a human-readable string, we assume that the
    # feature and feature value are valid UTF-8 strings (might not be true in
    # cases where people store serialised protos in the features for instance).
    keys.append(tf.compat.as_text(feature))
    # We need to call as_str_any to convert non-string (e.g. integer) values to
    # string first before converting to text.
    values.append(tf.compat.as_text(tf.compat.as_str_any(value)))

  # To use u'{}' instead of '{}' here to avoid encoding a unicode character with
  # ascii codec.
  return (separator.join([u'{}'.format(key) for key in keys]) + ':' +
          separator.join([u'{}'.format(value) for value in values]))


def is_cross_slice_applicable(
    cross_slice_key: CrossSliceKeyType,
    cross_slicing_spec: config.CrossSlicingSpec) -> bool:
  """Checks if CrossSlicingSpec is applicable to the CrossSliceKeyType."""
  baseline_slice_key, comparison_slice_key = cross_slice_key

  if not SingleSliceSpec(spec=cross_slicing_spec.baseline_spec
                        ).is_slice_applicable(baseline_slice_key):
    return False
  for comparison_slicing_spec in cross_slicing_spec.slicing_specs:
    if SingleSliceSpec(
        spec=comparison_slicing_spec).is_slice_applicable(comparison_slice_key):
      return True
  return False


def get_slice_key_type(
    slice_key: Union[SliceKeyType, CrossSliceKeyType]) -> Any:
  """Determines if the slice_key in SliceKeyType or CrossSliceKeyType format.

  Args:
    slice_key: The slice key which can be in SliceKeyType format or
      CrossSliceType format.

  Returns:
    SliceKeyType object or CrossSliceKeyType object.

  Raises:
    TypeError: If slice key is not recognized.
  """

  def is_singleton_slice_key_type(
      singleton_slice_key: SingletonSliceKeyType) -> bool:
    try:
      col, val = singleton_slice_key
    except ValueError:
      return False
    if (isinstance(col, (six.binary_type, six.text_type)) and
        (isinstance(val, (six.binary_type, six.text_type)) or
         isinstance(val, six.integer_types) or isinstance(val, float))):
      return True
    else:
      return False

  def is_slice_key_type(slice_key: SliceKeyType) -> bool:
    if not slice_key:
      return True

    for single_slice_key in slice_key:
      if not is_singleton_slice_key_type(single_slice_key):
        return False
    return True

  if is_slice_key_type(slice_key):
    return SliceKeyType

  try:
    baseline_slice, comparison_slice = slice_key
  except ValueError:
    raise TypeError('Unrecognized slice type. Neither SliceKeyType nor'
                    ' CrossSliceKeyType.')

  if (is_slice_key_type(baseline_slice) and
      is_slice_key_type(comparison_slice)):
    return CrossSliceKeyType
  else:
    raise TypeError('Unrecognized slice type. Neither SliceKeyType nor'
                    ' CrossSliceKeyType.')


def is_cross_slice_key(
    slice_key: Union[SliceKeyType, CrossSliceKeyType]) -> bool:
  """Returns whether slice_key is cross_slice or not."""
  return get_slice_key_type(slice_key) == CrossSliceKeyType


def _is_multi_dim_keys(slice_keys: SliceKeyType) -> bool:
  """Returns true if slice_keys are multi dimensional."""
  if isinstance(slice_keys, np.ndarray):
    return True
  if (isinstance(slice_keys, list) and slice_keys and
      isinstance(slice_keys[0], list)):
    return True
  return False


def slice_key_matches_slice_specs(
    slice_key: SliceKeyType, slice_specs: Iterable[SingleSliceSpec]) -> bool:
  """Checks whether a slice key matches any slice spec.

  In this setting, a slice key matches a slice spec if it could have been
  generated by that spec.

  Args:
    slice_key: The slice key to check for applicability against slice specs.
    slice_specs: Slice specs against which to check applicability of a slice
      key.

  Returns:
    True if the slice_key matches any slice specs, False otherwise.
  """
  for slice_spec in slice_specs:
    if slice_spec.is_slice_applicable(slice_key):
      return True
  return False


@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(Tuple[SliceKeyType, types.Extracts])
class _FanoutSlicesDoFn(beam.DoFn):
  """A DoFn that performs per-slice key fanout prior to computing aggregates."""

  def __init__(self, key_filter_fn: Callable[[Text], bool]):
    self._num_slices_generated_per_instance = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'num_slices_generated_per_instance')
    self._post_slice_num_instances = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'post_slice_num_instances')
    self._key_filter_fn = key_filter_fn

  def process(
      self,
      element: types.Extracts) -> List[Tuple[SliceKeyType, types.Extracts]]:
    key_filter_fn = self._key_filter_fn  # Local cache.
    filtered = {k: v for k, v in element.items() if key_filter_fn(k)}
    slice_keys = element.get(constants.SLICE_KEY_TYPES_KEY)
    # The query based evaluator will group slices into a multi-dimentional array
    # with an extra dimension representing the examples matching the query key.
    # We need to flatten and dedup the slice keys.
    if _is_multi_dim_keys(slice_keys):
      arr = np.array(slice_keys)
      unique_keys = set()
      for k in arr.flatten():
        unique_keys.add(k)
      if not unique_keys and arr.shape:
        # If only the empty overall slice is in array, it is removed by flatten
        unique_keys.add(())
      slice_keys = unique_keys
    result = [(slice_key, filtered) for slice_key in slice_keys]
    self._num_slices_generated_per_instance.update(len(result))
    self._post_slice_num_instances.inc(len(result))
    return result


# TODO(cyfoo): Possibly introduce the same telemetry in Lantern to help with
# evaluating importance of b/111353165 based on actual Lantern usage data.
@beam.ptransform_fn
@beam.typehints.with_input_types(Tuple[SliceKeyType, types.Extracts])
@beam.typehints.with_output_types(int)
def _TrackDistinctSliceKeys(  # pylint: disable=invalid-name
    slice_keys_and_values: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
  """Gathers slice key telemetry post slicing."""

  def increment_counter(element):  # pylint: disable=invalid-name
    num_distinct_slice_keys = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_distinct_slice_keys')
    num_distinct_slice_keys.inc(element)
    return element

  return (slice_keys_and_values
          | 'ExtractSliceKeys' >> beam.Keys()
          | 'RemoveDuplicates' >> beam.Distinct()
          | 'Size' >> beam.combiners.Count.Globally()
          | 'IncrementCounter' >> beam.Map(increment_counter))


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(Tuple[SliceKeyType, types.Extracts])
def FanoutSlices(  # pylint: disable=invalid-name
    pcoll: beam.pvalue.PCollection,
    include_slice_keys_in_output: Optional[bool] = False
) -> beam.pvalue.PCollection:  # pylint: disable=invalid-name
  """Fan out extracts based on slice keys (slice keys removed by default)."""
  if include_slice_keys_in_output:
    key_filter_fn = lambda k: True
  else:
    pruned_keys = (constants.SLICE_KEY_TYPES_KEY, constants.SLICE_KEYS_KEY)
    key_filter_fn = lambda k: k not in pruned_keys

  result = pcoll | 'DoSlicing' >> beam.ParDo(_FanoutSlicesDoFn(key_filter_fn))

  # pylint: disable=no-value-for-parameter
  _ = result | 'TrackDistinctSliceKeys' >> _TrackDistinctSliceKeys()
  # pylint: enable=no-value-for-parameter

  return result


# TFMA v1 uses Text for its keys while TFMA v2 uses MetricKey
_MetricsDict = Dict[Any, Any]


@beam.ptransform_fn
@beam.typehints.with_input_types(Tuple[SliceKeyType, _MetricsDict])
@beam.typehints.with_output_types(Tuple[SliceKeyType, _MetricsDict])
def FilterOutSlices(  # pylint: disable=invalid-name
    values: beam.pvalue.PCollection,
    slices_count: beam.pvalue.PCollection,
    min_slice_size: int,
    error_metric_key: Text = '__ERROR__') -> beam.pvalue.PCollection:
  """Filter out slices with examples count lower than k_anonymization_count.

  Since we might filter out certain slices to preserve privacy in the case of
  small slices, to make end users aware of this, we will append filtered out
  slice keys with empty data, and a debug message explaining the omission.

  Args:
    values: PCollection of aggregated data keyed at slice_key
    slices_count: PCollection of slice keys and their example count.
    min_slice_size: If the number of examples in a specific slice is less than
      min_slice_size, then an error will be returned for that slice. This will
      be useful to ensure privacy by not displaying the aggregated data for
      smaller number of examples.
    error_metric_key: The special metric key to indicate errors.

  Returns:
    A PCollection keyed at all the possible slice_key and aggregated data for
    slice keys with example count more than min_slice_size and error
    message for filtered out slices.
  """

  class FilterOutSmallSlicesDoFn(beam.DoFn):
    """DoFn to filter out small slices."""

    def __init__(self, error_metric_key: Text):
      self.error_metric_key = error_metric_key

    def process(
        self, element: Tuple[SliceKeyType, _MetricsDict]
    ) -> Generator[Tuple[SliceKeyType, _MetricsDict], None, None]:
      """Filter out small slices.

      For slices (excluding overall slice) with examples count lower than
      min_slice_size, it adds an error message.

      Args:
        element: Tuple containing slice key and a dictionary containing
          corresponding elements from merged pcollections.

      Yields:
        PCollection of (slice_key, aggregated_data or error message)
      """
      (slice_key, value) = element
      if value['values']:
        if (not slice_key or value['slices_count'][0] >= min_slice_size):
          yield (slice_key, value['values'][0])
        else:
          yield (
              slice_key,
              {
                  self.error_metric_key:  # LINT.IfChange
                      'Example count for this slice key is lower than the '
                      'minimum required value: %d. No data is aggregated for '
                      'this slice.' % min_slice_size
                  # LINT.ThenChange(../addons/fairness/frontend/fairness-metrics-board/fairness-metrics-board.js)
              })

  return ({
      'values': values,
      'slices_count': slices_count
  }
          | 'CoGroupingSlicesCountAndAggregatedData' >> beam.CoGroupByKey()
          | 'FilterOutSmallSlices' >> beam.ParDo(
              FilterOutSmallSlicesDoFn(error_metric_key)))
