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

from __future__ import print_function

import itertools


import six
from tensorflow_model_analysis import types
from tensorflow_model_analysis.slicer import slice_accessor
from tensorflow_model_analysis.types_compat import Generator, Iterable, List, Tuple, Union

# FeatureValueType represents a value that a feature could take.
FeatureValueType = Union[bytes, int, float]  # pylint: disable=invalid-name

# SingletonSliceKeyType is a tuple, where the first element is the key of the
# feature, and the second element is its value. This describes a single
# feature-value pair.
SingletonSliceKeyType = Tuple[bytes, FeatureValueType]  # pylint: disable=invalid-name

# SliceKeyType is a tuple of SingletonSliceKeyType. This completely describes
# a single slice.
SliceKeyType = Tuple[SingletonSliceKeyType, Ellipsis]  # pylint: disable=invalid-name

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

  def __eq__(self, other):
    # Need access to other's protected fields for comparison.
    if isinstance(other, self.__class__):
      # pylint: disable=protected-access
      return (self._columns == other._columns and
              self._features == other._features)
      # pylint: enable=protected-access
    else:
      return False

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash((self._columns, self._features))

  def __init__(self,
               columns = (),
               features = ()):
    """Initialises a SingleSliceSpec.

    Args:
      columns: an iterable of column names to slice on.
      features: a iterable of features to slice on. Each feature is a
        (key, value) tuple. Note that the value can be either a string or an
        int, and the type is taken into account when comparing values, so
        SingleSliceSpec(features=[('age', '5')]) will *not* match a slice
        with age=[5] (age is a string in the spec, but an int in the slice).

    Raises:
      ValueError: There was overlap between the columns specified in columns
        and those in features.
      ValueError: columns or features was a string: they should probably be a
        singleton list containing that string.
    """
    if isinstance(columns, str):
      raise ValueError('columns is a string: it should probably be a singleton '
                       'list containing that string')
    if isinstance(features, str):
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

  def is_slice_applicable(self, slice_key):
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

  def generate_slices(self, accessor
                     ):
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


def get_slices_for_features_dict(
    features_dict,
    slice_spec):
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


def stringify_slice_key(slice_key):
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
    keys.append(feature)
    values.append(value)

  return separator.join([
      '{}'.format(key) for key in keys
  ]) + ':' + separator.join(['{}'.format(value) for value in values])
