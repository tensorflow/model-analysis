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
"""Implements API for extracting features from an example."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function



import apache_beam as beam
import numpy as np
import tensorflow as tf

from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.eval_saved_model import load
from tensorflow_model_analysis.types_compat import Any, Dict


# For now, we store only the first N sparse keys in our diagnostics table.
_MAX_SPARSE_FEATURES_PER_COLUMN = 10


def _AugmentExtracts(
    fpl_dict,
    example_and_extracts):
  """Augments the ExampleAndExtracts with FeaturesPredictionsLabels.

  Args:
    fpl_dict: The dictionary returned by evaluate._Predict()
    example_and_extracts: The ExampleAndExtracts to be augmented. This is
      mutated in-place.

  Raises:
    TypeError: If the FeaturesPredictionsLabels is corrupt.
  """
  for name, val in fpl_dict.items():
    val = val.get(encoding.NODE_SUFFIX)

    if isinstance(val, tf.SparseTensorValue):
      example_and_extracts.extracts[name] = types.MaterializedColumn(
          name=name,
          value=val.values[0:_MAX_SPARSE_FEATURES_PER_COLUMN])

    elif isinstance(val, np.ndarray):
      val = val[0]  # only support first dim for now.
      if not np.isscalar(val):
        val = val[0:_MAX_SPARSE_FEATURES_PER_COLUMN]
      example_and_extracts.extracts[name] = types.MaterializedColumn(
          name=name, value=val)

    else:
      raise TypeError(
          'Dictionary item with key %s, value %s had unexpected type %s' %
          (name, val, type(val)))


def _MaterializeFeatures(
    example_and_extracts):
  """Converts FeaturesPredictionsLabels into MaterializedColumn in the extract.

  It must be the case that evaluate._Predict() was called on the
  ExampleAndExtracts before calling this function.

  Args:
    example_and_extracts: The ExampleAndExtracts to be augmented

  Returns:
    Returns an augmented ExampleAndExtracts (which is a shallow copy of
    the original ExampleAndExtracts, so the original isn't mutated)

  Raises:
    RuntimeError: When _Predict() didn't populate the 'fpl' key.
  """
  # Make a a shallow copy, so we don't mutate the original.
  result = example_and_extracts.create_copy_with_shallow_copy_of_extracts()

  fpl = result.extracts.get(constants.FEATURES_PREDICTIONS_LABELS_KEY)
  if not fpl:
    raise RuntimeError('FPL missing, Please ensure _Predict() was called.')

  if not isinstance(fpl, load.FeaturesPredictionsLabels):
    raise TypeError(
        'Expected FPL to be instance of FeaturesPredictionsLabel. FPL was: %s '
        'of type %s' % (str(fpl), type(fpl)))

  # We disable pytyping here because we know that 'fpl' key corresponds to a
  # non-materialized column.
  # pytype: disable=attribute-error
  _AugmentExtracts(fpl.features, result)
  _AugmentExtracts(fpl.predictions, result)
  _AugmentExtracts(fpl.labels, result)
  return result
  # pytype: enable=attribute-error


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.typehints.Any)
@beam.typehints.with_output_types(beam.typehints.Any)
def ExtractFeatures(
    examples_and_extracts):
  """Builds MaterializedColumn extracts from FPL created in evaluate.Predict().

  It must be the case that evaluate._Predict() was called on the
  ExampleAndExtracts before calling this function.

  Args:
    examples_and_extracts: PCollection containing the ExampleAndExtracts that
                           will have MaterializedColumn added to its extracts.

  Returns:
    PCollection of ExampleAndExtracts
  """
  return (examples_and_extracts
          | 'MaterializeFeatures' >> beam.Map(_MaterializeFeatures))
