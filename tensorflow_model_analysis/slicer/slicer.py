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

# Standard Imports
import apache_beam as beam
import six
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.post_export_metrics import metric_keys
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.slicer import slice_accessor
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Text, Tuple, Union

# FeatureValueType represents a value that a feature could take.
FeatureValueType = Union[bytes, int, float]  # pylint: disable=invalid-name

# SingletonSliceKeyType is a tuple, where the first element is the key of the
# feature, and the second element is its value. This describes a single
# feature-value pair.
SingletonSliceKeyType = Tuple[Text, FeatureValueType]  # pylint: disable=invalid-name

# SliceKeyType is a either the empty tuple (for the overal slice) or a tuple of
# SingletonSliceKeyType. This completely describes a single slice.
SliceKeyType = Union[Tuple[()], Tuple[SingletonSliceKeyType, ...]]  # pylint: disable=invalid-name

OVERALL_SLICE_NAME = 'Overall'


class SingleSliceSpec(object):
  """Specification for a single slice.

  This is intended to be an immutable class that specifies a single slice.
  Use this in conjunction with get_slices_for_features_dict to generate slices
  for a dictionary of features.

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
               features: Iterable[Tuple[Text, FeatureValueType]] = ()):
    """Initialises a SingleSliceSpec.

    Args:
      columns: an iterable of column names to slice on.
      features: a iterable of features to slice on. Each feature is a (key,
        value) tuple. Note that the value can be either a string or an int, and
        the type is taken into account when comparing values, so
        SingleSliceSpec(features=[('age', '5')]) will *not* match a slice with
        age=[5] (age is a string in the spec, but an int in the slice).

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

    if columns is None:
      columns = []

    if features is None:
      features = []

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
      if singleton_slice_key in features:
        features.remove(singleton_slice_key)
      elif singleton_slice_key[0] in columns:
        columns.remove(singleton_slice_key[0])
      else:
        return False
    return not features and not columns

  def generate_slices(self, accessor: slice_accessor.SliceAccessor
                     ) -> Generator[SliceKeyType, None, None]:
    """Generates all slices that match this specification from the data.

    Should only be called within this file.

    Examples:
      - columns = [], features = []
        slice accessor has features age=[5], gender=['f'], interest=['knitting']
        returns [[]]
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

      if value not in accessor.get(key):
        if isinstance(value, str):
          if value.encode() not in accessor.get(key):  # For Python3.
            return
        else:
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

      column_matches.append([(column, value) for value in accessor.get(column)])

    # We can now take the Cartesian product of the column_matches, and append
    # the value matches to each element of that, to generate the final list of
    # slices.
    for column_part in itertools.product(*column_matches):
      yield tuple(sorted(self._value_matches + list(column_part)))


def serialize_slice_key(slice_key: SliceKeyType
                       ) -> metrics_for_slice_pb2.SliceKey:
  """Converts SliceKeyType to SliceKey proto.

  Args:
    slice_key: The slice key in the format of SliceKeyType.

  Returns:
    The slice key in the format of SliceKey proto.

  Raises:
    TypeError: If the evaluate type is unreconized.
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


def deserialize_slice_key(slice_key: metrics_for_slice_pb2.SliceKey
                         ) -> SliceKeyType:
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
      value = elem.bytes_value
    elif elem.HasField('int64_value'):
      value = elem.int64_value
    elif elem.HasField('float_value'):
      value = elem.float_value
    else:
      raise TypeError('unrecognized type of type %s, value %s' %
                      (type(elem), elem))
    result.append((elem.column, value))
  return tuple(result)


def get_slices_for_features_dict(features_dict: types.DictOfFetchedTensorValues,
                                 slice_spec: List[SingleSliceSpec]
                                ) -> Iterable[SliceKeyType]:
  """Generates the slice keys appropriate for the given features dictionary.

  Args:
    features_dict: Features dictionary.
    slice_spec: slice specification.

  Yields:
    Slice keys appropriate for the given features dictionary.
  """
  accessor = slice_accessor.SliceAccessor(features_dict)
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

  def process(self, element: types.Extracts
             ) -> List[Tuple[SliceKeyType, types.Extracts]]:
    key_filter_fn = self._key_filter_fn  # Local cache.
    filtered = {k: v for k, v in element.items() if key_filter_fn(k)}
    result = [(slice_key, filtered)
              for slice_key in element.get(constants.SLICE_KEY_TYPES_KEY)]
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
def FanoutSlices(pcoll: beam.pvalue.PCollection,
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


@beam.ptransform_fn
@beam.typehints.with_input_types(Tuple[SliceKeyType, types.Extracts])
@beam.typehints.with_output_types(Tuple[SliceKeyType, types.Extracts])
def FilterOutSlices(  # pylint: disable=invalid-name
    values: beam.pvalue.PCollection, slices_count: beam.pvalue.PCollection,
    k_anonymization_count: int) -> beam.pvalue.PCollection:
  """Filter out slices with examples count lower than k_anonymization_count.

  Since we might filter out certain slices to preserve privacy in the case of
  small slices, to make end users aware of this, we will append filtered out
  slice keys with empty data, and a debug message explaining the omission.

  Args:
    values: PCollection of aggregated data keyed at slice_key
    slices_count: PCollection of slice keys and their example count.
    k_anonymization_count: If the number of examples in a specific slice is less
      than k_anonymization_count, then an error will be returned for that slice.
      This will be useful to ensure privacy by not displaying the aggregated
      data for smaller number of examples.

  Returns:
    A PCollection keyed at all the possible slice_key and aggregated data for
    slice keys with example count more than k_anonymization_count and error
    message for filtered out slices.
  """

  class FilterOutSmallSlicesDoFn(beam.DoFn):
    """DoFn to filter out small slices.

    For slices (excluding overall slice) with examples count lower than
    k_anonymization_count, it adds an error message.

    Args:
      element: Tuple containing slice key and a dictionary containing
        corresponding elements from merged pcollections.

    Returns:
      PCollection of (slice_key, aggregated_data or error message)
    """

    def process(self, element: Tuple[SliceKeyType, Dict[Text, Any]]
               ) -> Generator[Tuple[SliceKeyType, Dict[Text, Any]], None, None]:
      (slice_key, value) = element
      if value['values']:
        if (not slice_key or value['slices_count'][0] >= k_anonymization_count):
          yield (slice_key, value['values'][0])
        else:
          yield (slice_key, {
              metric_keys.ERROR_METRIC:
                  'Example count for this slice key is lower than '
                  'the minimum required value: %d. No data is aggregated for '
                  'this slice.' % k_anonymization_count
          })

  return ({
      'values': values,
      'slices_count': slices_count
  }
          | 'CoGroupingSlicesCountAndAggregatedData' >> beam.CoGroupByKey()
          | 'FilterOutSmallSlices' >> beam.ParDo(FilterOutSmallSlicesDoFn()))
