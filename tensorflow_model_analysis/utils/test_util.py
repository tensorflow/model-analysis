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

import math
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import model_util

from tensorflow.core.example import example_pb2


class TensorflowModelAnalysisTest(tf.test.TestCase):
  """Test class that extends tf.test.TestCase with extra functionality."""

  def setUp(self) -> None:  # pylint: disable=invalid-name
    super().setUp()
    self.longMessage = True  # pylint: disable=invalid-name

  def _getTempDir(self) -> str:  # pylint: disable=invalid-name
    return tempfile.mkdtemp()

  def _makeExample(self, **kwargs) -> tf.train.Example:  # pylint: disable=invalid-name
    """Make a TensorFlow Example with the given fields.

    The arguments can be singleton values, or a list of values, e.g.
    makeExample(age=3.0, fruits=['apples', 'pears', 'oranges']).
    Empty lists are not allowed, since we won't be able to deduce the type.

    Args:
      **kwargs: Each key=value pair defines a field in the example to be
        constructed. The name of the field will be key, and the value will be
        value. The type will be deduced from the type of the value. Care must be
        taken for numeric types: 0 will be interpreted as an int, and 0.0 as a
        float.

    Returns:
      TensorFlow.Example with the corresponding fields set to the corresponding
      values.

    Raises:
      ValueError: One of the arguments was an empty list.
      TypeError: One of the elements (or one of the elements in a list) had an
        unsupported type.
    """

    result = example_pb2.Example()
    for key, value in kwargs.items():
      if isinstance(value, float):
        result.features.feature[key].float_list.value[:] = [value]
      elif isinstance(value, int):
        result.features.feature[key].int64_list.value[:] = [value]
      elif isinstance(value, bytes):
        result.features.feature[key].bytes_list.value[:] = [value]
      elif isinstance(value, str):
        result.features.feature[key].bytes_list.value[:] = [
            value.encode('utf8')
        ]
      elif isinstance(value, list):
        if len(value) == 0:  # pylint: disable=g-explicit-length-test
          raise ValueError(
              'empty lists not allowed, but field %s was an empty list' % key
          )
        if isinstance(value[0], float):
          result.features.feature[key].float_list.value[:] = value
        elif isinstance(value[0], int):
          result.features.feature[key].int64_list.value[:] = value
        elif isinstance(value[0], bytes):
          result.features.feature[key].bytes_list.value[:] = value
        elif isinstance(value[0], str):
          result.features.feature[key].bytes_list.value[:] = [
              v.encode('utf8') for v in value
          ]
        else:
          raise TypeError(
              'field %s was a list, but the first element had unknown type %s'
              % key,
              type(value[0]),
          )
      else:
        raise TypeError(
            'unrecognised type for field %s: type %s' % (key, type(value))
        )
    return result

  def assertHasKeyWithTDistributionAlmostEqual(  # pylint: disable=invalid-name
      self,
      d: Dict[str, types.ValueWithTDistribution],
      key: str,
      value: float,
      places: int = 5,
  ) -> None:

    self.assertIn(key, d)
    self.assertIsInstance(d[key], types.ValueWithTDistribution)
    self.assertAlmostEqual(
        d[key].unsampled_value, value, places=places, msg='key {}'.format(key))

  def assertHasKeyWithValueAlmostEqual(  # pylint: disable=invalid-name
      self,
      d: Dict[str, float],
      key: str,
      value: float,
      places: int = 5,
  ) -> None:
    self.assertIn(key, d)
    self.assertAlmostEqual(
        d[key], value, places=places, msg='key {}'.format(key))

  def assertDictElementsAlmostEqual(  # pylint: disable=invalid-name
      self,
      got_values_dict: Dict[str, float],
      expected_values_dict: Dict[str, float],
      places: int = 5,
  ) -> None:
    for key, expected_value in expected_values_dict.items():
      self.assertHasKeyWithValueAlmostEqual(got_values_dict, key,
                                            expected_value, places)

  def assertDictElementsWithTDistributionAlmostEqual(  # pylint: disable=invalid-name
      self,
      got_values_dict: Dict[str, types.ValueWithTDistribution],
      expected_values_dict: Dict[str, float],
      places: int = 5,
  ) -> None:
    for key, expected_value in expected_values_dict.items():
      self.assertHasKeyWithTDistributionAlmostEqual(got_values_dict, key,
                                                    expected_value, places)

  def assertDictMatrixRowsAlmostEqual(  # pylint: disable=invalid-name
      self,
      got_values_dict: Dict[str, Sequence[Iterable[Union[float, int]]]],
      expected_values_dict: Dict[
          str, Iterable[Tuple[int, Iterable[Union[float, int]]]]
      ],
      places: int = 5,
  ) -> None:
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

  def createKerasTestEvalSharedModel(  # pylint: disable=invalid-name
      self,
      eval_saved_model_path: str,
      eval_config: config_pb2.EvalConfig,
  ) -> types.EvalSharedModel:
    """Create a test Keras EvalSharedModel."""
    return model_eval_lib.default_eval_shared_model(
        eval_saved_model_path=eval_saved_model_path, eval_config=eval_config
    )

  def assertSequenceAlmostEqual(  # pylint: disable=invalid-name
      self,
      got_seq: Iterable[Union[float, int]],
      expected_seq: Iterable[Union[float, int]],
      places: int = 5,
      delta: float = 0,
      msg_prefix='',
  ) -> None:
    """Assert that two sequences are almost equal."""
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

  def assertSparseTensorValueEqual(  # pylint: disable=invalid-name
      self, expected_sparse_tensor_value: Union[types.SparseTensorValue,
                                                tf.compat.v1.SparseTensorValue],
      got_sparse_tensor_value: Union[types.SparseTensorValue,
                                     tf.compat.v1.SparseTensorValue]
  ) -> None:
    """Assert that two SparseTensorValues are equal."""
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

  def createTestEvalSharedModel(  # pylint: disable=invalid-name
      self,
      model_path: Optional[str] = None,
      add_metrics_callbacks: Optional[
          List[types.AddMetricsCallbackType]
      ] = None,
      include_default_metrics: Optional[bool] = True,
      example_weight_key: Optional[Union[str, Dict[str, str]]] = None,
      additional_fetches: Optional[List[str]] = None,
      tags: Optional[str] = None,
      model_type: Optional[str] = None,
      model_name: str = '',
      rubber_stamp: Optional[bool] = False,
      is_baseline: Optional[bool] = False,
  ) -> types.EvalSharedModel:
    """Create a test EvalSharedModel."""

    if not model_type:
      model_type = model_util.get_model_type(None, model_path, tags)
    if model_type == constants.TFMA_EVAL:
      raise ValueError(
          f'Models of type {model_type} are deprecated. Please do not use it'
          'for testing.'
      )
    if not tags:
      tags = [tf.saved_model.SERVING]

    return types.EvalSharedModel(
        model_name=model_name,
        model_type=model_type,
        model_path=model_path,
        add_metrics_callbacks=add_metrics_callbacks,
        example_weight_key=example_weight_key,
        rubber_stamp=rubber_stamp,
        is_baseline=is_baseline,
        model_loader=types.ModelLoader(
            tags=tags,
            construct_fn=model_util.model_construct_fn(
                eval_saved_model_path=model_path,
                model_type=model_type,
                add_metrics_callbacks=add_metrics_callbacks,
                include_default_metrics=include_default_metrics,
                additional_fetches=additional_fetches,
                tags=tags,
            ),
        ),
    )
