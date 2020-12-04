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
"""Utilities for writing tests."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import math
import tempfile

from typing import Dict, Iterable, List, Optional, Union, Sequence, Text, Tuple

import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import model_util
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import constants as eval_constants
from tensorflow_model_analysis.eval_saved_model import load
from tensorflow_model_analysis.eval_saved_model import util


class TensorflowModelAnalysisTest(tf.test.TestCase):
  """Test class that extends tf.test.TestCase with extra functionality."""

  def setUp(self) -> None:
    self.longMessage = True  # pylint: disable=invalid-name

  def _getTempDir(self) -> Text:
    return tempfile.mkdtemp()

  def _makeExample(self, **kwargs) -> tf.train.Example:
    return util.make_example(**kwargs)

  def assertHasKeyWithTDistributionAlmostEqual(
      self,
      d: Dict[Text, types.ValueWithTDistribution],
      key: Text,
      value: float,
      places: int = 5) -> None:

    self.assertIn(key, d)
    self.assertIsInstance(d[key], types.ValueWithTDistribution)
    self.assertAlmostEqual(
        d[key].unsampled_value, value, places=places, msg='key {}'.format(key))

  def assertHasKeyWithValueAlmostEqual(self,
                                       d: Dict[Text, float],
                                       key: Text,
                                       value: float,
                                       places: int = 5) -> None:
    self.assertIn(key, d)
    self.assertAlmostEqual(
        d[key], value, places=places, msg='key {}'.format(key))

  def assertDictElementsAlmostEqual(self,
                                    got_values_dict: Dict[Text, float],
                                    expected_values_dict: Dict[Text, float],
                                    places: int = 5) -> None:
    for key, expected_value in expected_values_dict.items():
      self.assertHasKeyWithValueAlmostEqual(got_values_dict, key,
                                            expected_value, places)

  def assertDictElementsWithTDistributionAlmostEqual(
      self,
      got_values_dict: Dict[Text, types.ValueWithTDistribution],
      expected_values_dict: Dict[Text, float],
      places: int = 5) -> None:
    for key, expected_value in expected_values_dict.items():
      self.assertHasKeyWithTDistributionAlmostEqual(got_values_dict, key,
                                                    expected_value, places)

  def assertDictMatrixRowsAlmostEqual(
      self,
      got_values_dict: Dict[Text, Sequence[Iterable[Union[float, int]]]],
      expected_values_dict: Dict[Text, Iterable[Tuple[int,
                                                      Iterable[Union[float,
                                                                     int]]]]],
      places: int = 5) -> None:
    """Fails if got_values_dict does not match values in expected_values_dict.

    For each entry, expected_values_dict provides the row index and the values
    of that row to be compared to the bucketing result in got_values_dict. For
    example:
      got_values_dict={'key', [[1,2,3],[4,5,6],[7,8,9]]}
    you can check the first and last row of got_values_dict[key] by setting
      expected_values_dict={'key', [(0,[1,2,3]), (2,[7,8,9])]}

    Args:
      got_values_dict: The dict got, where each value represents a full
        bucketing result.
      expected_values_dict: The expected dict. It may contain a subset of keys
        in got_values_dict. The value is of type "Iterable[Tuple[int,
        Iterable[scalar]]]", where each Tuple contains the index of a row to be
        checked and the expected values of that row.
      places: The number of decimal places to compare.
    """
    for key, expected_value in expected_values_dict.items():
      self.assertIn(key, got_values_dict)
      for (row, values) in expected_value:
        self.assertSequenceAlmostEqual(
            got_values_dict[key][row],
            values,
            places=places,
            msg_prefix='for key %s, row %d: ' % (key, row))

  def assertSequenceAlmostEqual(self,
                                got_seq: Iterable[Union[float, int]],
                                expected_seq: Iterable[Union[float, int]],
                                places: int = 5,
                                delta: float = 0,
                                msg_prefix='') -> None:
    got = list(got_seq)
    expected = list(expected_seq)
    self.assertEqual(
        len(got), len(expected), msg=msg_prefix + 'lengths do not match')
    for index, (a, b) in enumerate(zip(got, expected)):
      msg = msg_prefix + 'at index %d. sequences were: %s and %s' % (index, got,
                                                                     expected)
      if math.isnan(a) or math.isnan(b):
        self.assertEqual(math.isnan(a), math.isnan(b), msg=msg)
      else:
        if delta:
          self.assertAlmostEqual(a, b, msg=msg, delta=delta)
        else:
          self.assertAlmostEqual(a, b, msg=msg, places=places)

  def assertSparseTensorValueEqual(
      self, expected_sparse_tensor_value: tf.compat.v1.SparseTensorValue,
      got_sparse_tensor_value: tf.compat.v1.SparseTensorValue) -> None:
    self.assertAllEqual(expected_sparse_tensor_value.indices,
                        got_sparse_tensor_value.indices)
    self.assertAllEqual(expected_sparse_tensor_value.values,
                        got_sparse_tensor_value.values)
    self.assertAllEqual(expected_sparse_tensor_value.dense_shape,
                        got_sparse_tensor_value.dense_shape)
    # Check dtypes too
    self.assertEqual(expected_sparse_tensor_value.indices.dtype,
                     got_sparse_tensor_value.indices.dtype)
    self.assertEqual(expected_sparse_tensor_value.values.dtype,
                     got_sparse_tensor_value.values.dtype)
    self.assertEqual(expected_sparse_tensor_value.dense_shape.dtype,
                     got_sparse_tensor_value.dense_shape.dtype)

  def createTestEvalSharedModel(
      self,
      eval_saved_model_path: Optional[Text] = None,
      add_metrics_callbacks: Optional[List[
          types.AddMetricsCallbackType]] = None,
      include_default_metrics: Optional[bool] = True,
      example_weight_key: Optional[Union[Text, Dict[Text, Text]]] = None,
      additional_fetches: Optional[List[Text]] = None,
      tags: Optional[Text] = None,
      model_type: Optional[Text] = None,
      model_name: Text = '',
      rubber_stamp: Optional[bool] = False,
      is_baseline: Optional[bool] = False) -> types.EvalSharedModel:

    if not model_type:
      model_type = model_util.get_model_type(None, eval_saved_model_path, tags)
    if not tags:
      if model_type in (constants.TF_GENERIC, constants.TF_ESTIMATOR):
        model_type = constants.TF_ESTIMATOR
        tags = [eval_constants.EVAL_TAG]
      else:
        tags = [tf.saved_model.SERVING]

    return types.EvalSharedModel(
        model_name=model_name,
        model_type=model_type,
        model_path=eval_saved_model_path,
        add_metrics_callbacks=add_metrics_callbacks,
        example_weight_key=example_weight_key,
        rubber_stamp=rubber_stamp,
        is_baseline=is_baseline,
        model_loader=types.ModelLoader(
            tags=tags,
            construct_fn=model_util.model_construct_fn(
                eval_saved_model_path=eval_saved_model_path,
                model_type=model_type,
                add_metrics_callbacks=add_metrics_callbacks,
                include_default_metrics=include_default_metrics,
                additional_fetches=additional_fetches,
                tags=tags)))

  def predict_injective_single_example(
      self, eval_saved_model: load.EvalSavedModel,
      raw_example_bytes: bytes) -> types.FeaturesPredictionsLabels:
    """Run predict for a single example for a injective model.

    Args:
      eval_saved_model: EvalSavedModel
      raw_example_bytes: Raw example bytes for the example

    Returns:
      The singular FPL returned by eval_saved_model.predict on the given
      raw_example_bytes.
    """
    fetched_list = eval_saved_model.predict(raw_example_bytes)
    self.assertEqual(1, len(fetched_list))
    self.assertEqual(0, fetched_list[0].input_ref)
    return eval_saved_model.as_features_predictions_labels(fetched_list)[0]

  def predict_injective_example_list(
      self, eval_saved_model: load.EvalSavedModel,
      raw_example_bytes_list: List[bytes]
  ) -> List[types.FeaturesPredictionsLabels]:
    """Run predict_list for a list of examples for a injective model.

    Args:
      eval_saved_model: EvalSavedModel
      raw_example_bytes_list: List of raw example bytes

    Returns:
      The list of FPLs returned by eval_saved_model.predict on the given
      raw_example_bytes.
    """
    fetched_list = eval_saved_model.predict_list(raw_example_bytes_list)

    # Check that each FPL corresponds to one example.
    self.assertSequenceEqual(
        range(0, len(raw_example_bytes_list)),
        [fetched.input_ref for fetched in fetched_list])

    return eval_saved_model.as_features_predictions_labels(fetched_list)
