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

import apache_beam as beam
from tensorflow_model_analysis import types
from typing import List, NamedTuple, Optional, Text

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
def Filter(extracts: beam.pvalue.PCollection,
           include: Optional[List[Text]] = None,
           exclude: Optional[List[Text]] = None):
  """Filters extracts to include/exclude specified keys.

  Args:
    extracts: PCollection of extracts.
    include: Keys to include in output.
    exclude: Keys to exclude from output.

  Returns:
    Filtered PCollection of Extracts.

  Raises:
    ValueError: If both include and exclude are used.
  """
  if include and exclude:
    raise ValueError('only one of include or exclude should be used.')

  # Make into sets for lookup efficiency.
  include = frozenset(include or [])
  exclude = frozenset(exclude or [])

  def filter_extracts(extracts: types.Extracts) -> types.Extracts:  # pylint: disable=invalid-name
    """Filters extracts."""
    if not include and not exclude:
      return extracts
    elif include:
      return {k: v for k, v in extracts.items() if k in include}
    else:
      assert exclude
      return {k: v for k, v in extracts.items() if k not in exclude}

  return extracts | beam.Map(filter_extracts)
