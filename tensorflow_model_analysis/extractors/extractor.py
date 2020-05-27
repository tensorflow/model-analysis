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
"""Extractor type."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Any, Dict, Iterable, NamedTuple, Optional, Text, Union

import apache_beam as beam
from tensorflow_model_analysis import types

# Tag for the last extractor in list of extractors.
LAST_EXTRACTOR_STAGE_NAME = '<last-extractor>'

# An Extractor is a PTransform that takes Extracts as input and returns Extracts
# as output. A typical example is a PredictExtractor that receives an 'input'
# placeholder for input and adds additional 'features', 'labels', and
# 'predictions' extracts.
Extractor = NamedTuple(  # pylint: disable=invalid-name
    'Extractor',
    [
        ('stage_name', Text),
        # PTransform Extracts -> Extracts
        ('ptransform', beam.PTransform)
    ])


@beam.ptransform_fn
@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
def Filter(  # pylint: disable=invalid-name
    extracts: beam.pvalue.PCollection,
    include: Optional[Union[Iterable[Text], Dict[Text, Any]]] = None,
    exclude: Optional[Union[Iterable[Text],
                            Dict[Text,
                                 Any]]] = None) -> beam.pvalue.PCollection:
  """Filters extracts to include/exclude specified keys.

  Args:
    extracts: PCollection of extracts.
    include: List or map of keys to include in output. If a map of keys is
      passed then the keys and sub-keys that exist in the map will be included
      in the output. An empty dict behaves as a wildcard matching all keys or
      the value itself. Since matching on feature values is not currently
      supported, an empty dict must be used to represent the leaf nodes.
      For example: {'key1': {'key1-subkey': {}}, 'key2': {}}.
    exclude: List or map of keys to exclude from output. If a map of keys is
      passed then the keys and sub-keys that exist in the map will be excluded
      from the output. An empty dict behaves as a wildcard matching all keys or
      the value itself. Since matching on feature values is not currently
      supported, an empty dict must be used to represent the leaf nodes.
      For example: {'key1': {'key1-subkey': {}}, 'key2': {}}.

  Returns:
    Filtered PCollection of Extracts.

  Raises:
    ValueError: If both include and exclude are used.
  """
  if include and exclude:
    raise ValueError('only one of include or exclude should be used.')

  if not isinstance(include, dict):
    include = {k: {} for k in include or []}
  if not isinstance(exclude, dict):
    exclude = {k: {} for k in exclude or []}

  def filter_extracts(extracts: types.Extracts) -> types.Extracts:  # pylint: disable=invalid-name
    """Filters extracts."""
    if not include and not exclude:
      return extracts
    elif include:
      return _include_filter(include, extracts)
    else:
      return _exclude_filter(exclude, extracts)

  return extracts | beam.Map(filter_extracts)


def _include_filter(include, target):
  """Filters target to only include keys in include.

  Args:
    include: Dict of keys from target to include. An empty dict matches all
      values.
    target: Target dict to apply filter to.

  Returns:
    A new dict with values from target filtered out.
  """
  if not include:
    return target

  result = {}
  for key, subkeys in include.items():
    if key in target:
      if subkeys:
        result[key] = _include_filter(subkeys, target[key])
      else:
        result[key] = target[key]
  return result


def _exclude_filter(exclude, target):
  """Filters output to only include keys not in exclude.

  Args:
    exclude: Dict of keys from target to exclude. An empty dict matches all
      values.
    target: Target dict to apply filter to.

  Returns:
    A new dict with values from target filtered out.
  """
  result = {}
  for key, value in target.items():
    if key in exclude:
      if exclude[key]:
        value = _exclude_filter(exclude[key], target[key])
        if value:
          result[key] = value
    else:
      result[key] = value
  return result
