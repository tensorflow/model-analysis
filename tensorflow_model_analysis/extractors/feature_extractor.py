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
  """Augments The ExampleAndExtracts with FeaturesPredictionsLabels.

  Args:
    fpl_dict: The dictionary returned by evaluate._Predict()
    example_and_extracts: The ExampleAndExtracts to be augmented -- note that
      this variable modified (ie both an input and output)
  Raises:
    TypeError: if the FeaturesPredictionsLabels is corrupt.
  """
  for name, val in fpl_dict.iteritems():
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
      raise TypeError('Unexpected fpl type: %s' % str(val))


def _MaterializeFeatures(
    example_and_extracts):
  """Converts FeaturesPredictionsLabels into MaterializedColumn in the extract.

  It must be the case that evaluate._Predict() was called on the
  ExampleAndExtracts before calling this function.

  Args:
    example_and_extracts: The ExampleAndExtracts to be augmented

  Returns:
    Reference to augmented ExampleAndExtracts.

  Raises:
    RuntimeError: When _Predict() didn't populate the 'fpl' key.
  """
  fpl = example_and_extracts.extracts.get(
      constants.FEATURES_PREDICTIONS_LABELS_KEY)
  if not fpl:
    raise RuntimeError(
        'fpl missing, Please ensure _Predict() was called.')

  if not isinstance(fpl, load.FeaturesPredictionsLabels):
    raise RuntimeError(
        'Expected FPL to be instance of FeaturesPredictionsLabel. FPL was: %s'
        % str(fpl))

  # We disable pytyping here because we know that 'fpl' key corresponds to a
  # non-materialized column.
  # pytype: disable=attribute-error
  _AugmentExtracts(fpl.features, example_and_extracts)
  _AugmentExtracts(fpl.predictions, example_and_extracts)
  _AugmentExtracts(fpl.labels, example_and_extracts)
  return example_and_extracts
  # pytype: enable=attribute-error


@beam.typehints.with_input_types(beam.typehints.Any)
@beam.typehints.with_output_types(beam.typehints.Any)
@beam.ptransform_fn
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
