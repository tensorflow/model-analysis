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
"""General utilities."""

import collections
import inspect
import sys
import traceback

from typing import Any, List, Mapping, MutableMapping, Optional, Union

import numpy as np
import six
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types

# Separator used when combining multiple layers of Extracts keys into a single
# string. Normally we would like to use '.' or '/' as a separator, but the
# output gets written to a table backed by a proto based schema which limits the
# characters that can be used to [a-zA-Z_].
KEY_SEPARATOR = '__'
# Suffix for keys representing the top k keys associated with a sparse item.
KEYS_SUFFIX = 'keys'
# Suffix for keys representing the top k values associated with a sparse item.
VALUES_SUFFIX = 'values'


def is_sparse_or_ragged_tensor_value(tensor: Any) -> bool:
  """Returns true if sparse or ragged tensor."""
  return (isinstance(tensor, types.SparseTensorValue) or
          isinstance(tensor, types.RaggedTensorValue) or
          isinstance(tensor, tf.compat.v1.SparseTensorValue))


def to_numpy(tensor: Any) -> np.ndarray:
  """Converts tensor type (list, etc) to np.ndarray if not already."""
  if isinstance(tensor, np.ndarray):
    return tensor
  elif hasattr(tensor, 'numpy') and callable(tensor.numpy):
    return tensor.numpy()
  else:
    return np.array(tensor)


def to_sparse_tensor_value(
    sparse_tensor: Union[tf.SparseTensor, tf.compat.v1.SparseTensorValue]
) -> types.SparseTensorValue:
  """Converts sparse tensor to SparseTensorValue."""
  return types.SparseTensorValue(
      to_numpy(sparse_tensor.values), to_numpy(sparse_tensor.indices),
      to_numpy(sparse_tensor.dense_shape))


def to_ragged_tensor_value(
    ragged_tensor: tf.RaggedTensor) -> types.RaggedTensorValue:
  """Converts ragged tensor to RaggedTensorValue."""
  nested_row_splits = []
  for splits in ragged_tensor.nested_row_splits:
    nested_row_splits.append(to_numpy(splits))
  return types.RaggedTensorValue(
      to_numpy(ragged_tensor.flat_values), nested_row_splits)


def to_tensor_value(tensor: Any) -> types.TensorValue:
  """Converts tensor to a tensor value."""
  if isinstance(tensor, tf.SparseTensor):
    return to_sparse_tensor_value(tensor)
  elif isinstance(tensor, tf.compat.v1.SparseTensorValue):
    return to_sparse_tensor_value(tensor)
  elif isinstance(tensor, tf.RaggedTensor):
    return to_ragged_tensor_value(tensor)
  elif isinstance(tensor, tf.compat.v1.ragged.RaggedTensorValue):
    return to_ragged_tensor_value(tensor)
  else:
    return to_numpy(tensor)


def to_tensor_values(tensors: Any) -> types.TensorValueMaybeMultiLevelDict:
  """Converts multi-level dict of tensors to tensor values."""
  if isinstance(tensors, Mapping):
    result = {}
    for k, v in tensors.items():
      result[k] = to_tensor_values(v)
    return result
  else:
    return to_tensor_value(tensors)


def is_tensorflow_tensor(obj):
  return (isinstance(obj, tf.Tensor) or isinstance(obj, tf.SparseTensor) or
          isinstance(obj, tf.RaggedTensor))


def to_tensorflow_tensor(tensor_value: types.TensorValue) -> types.TensorType:
  """Converts a tensor value to a tensorflow tensor."""
  if isinstance(tensor_value, types.RaggedTensorValue):
    return tf.RaggedTensor.from_nested_row_splits(
        tensor_value.values, tensor_value.nested_row_splits)
  elif (isinstance(tensor_value, types.SparseTensorValue) or
        isinstance(tensor_value, tf.compat.v1.SparseTensorValue)):
    return tf.SparseTensor(
        values=tensor_value.values,
        indices=tensor_value.indices,
        dense_shape=tensor_value.dense_shape)
  else:
    return tf.convert_to_tensor(tensor_value)


def to_tensorflow_tensors(
    values: types.TensorValueMaybeMultiLevelDict,
    specs: Optional[types.TypeSpecMaybeMultiLevelDict] = None
) -> types.TensorTypeMaybeMultiLevelDict:
  """Converts multi-level dict of tensor values to tensorflow tensors.

  Args:
    values: Tensor values (np.ndarray, etc) to convert to tensorflow tensors.
    specs: If provided, filters output to include only tensors that are also in
      specs. Returned tensors will also be validated against their associated
      spec.

  Returns:
    Tensorflow tensors optionally filtered by specs.

  Raises:
    ValueError: If specs is provided and a tensor value does not match a
      provided spec.
  """
  # Wrap in inner function so we can hold onto contextual info for logging
  all_values = values
  all_specs = specs

  def to_tensors(values: types.TensorValueMaybeMultiLevelDict,
                 specs: Optional[types.TypeSpecMaybeMultiLevelDict],
                 keys: List[str]) -> types.TensorTypeMaybeMultiLevelDict:
    if (specs is not None and
        ((isinstance(specs, Mapping) and not isinstance(values, Mapping)) or
         (not isinstance(specs, Mapping) and isinstance(values, Mapping)))):
      raise ValueError(
          f'incompatible types: values = {values} and specs = {specs} \n\n'
          f'Failed at key {keys} in {all_values} and specs {all_specs}')

    if isinstance(values, Mapping):
      result = {}
      for key in specs.keys() if specs else values.keys():
        if key not in values:
          raise ValueError(
              f'Tensor value for {key} not found in {values}: \n\n'
              f'Failed at key {keys} in {all_values} and specs {all_specs}')
        result[key] = to_tensors(values[key], specs[key] if specs else None,
                                 keys + [key])
      return result
    else:
      tensor = to_tensorflow_tensor(values)
      if (isinstance(specs, tf.TypeSpec) and
          not specs.is_compatible_with(tensor)):
        raise ValueError(
            f'Tensor {tensor} is not compatible with {specs} \n\n'
            f'Failed at key {keys} in {all_values} and specs {all_specs}')
      return tensor

  return to_tensors(values, specs, [])


def unique_key(key: str,
               current_keys: List[str],
               update_keys: Optional[bool] = False) -> str:
  """Returns a unique key given a list of current keys.

  If the key exists in current_keys then a new key with _1, _2, ..., etc
  appended will be returned, otherwise the key will be returned as passed.

  Args:
    key: desired key name.
    current_keys: List of current key names.
    update_keys: True to append the new key to current_keys.
  """
  index = 1
  k = key
  while k in current_keys:
    k = '%s_%d' % (key, index)
    index += 1
  if update_keys:
    current_keys.append(k)
  return k


def compound_key(keys: List[str], separator: str = KEY_SEPARATOR) -> str:
  """Returns a compound key based on a list of keys.

  Args:
    keys: Keys used to make up compound key.
    separator: Separator between keys. To ensure the keys can be parsed out of
      any compound key created, any use of a separator within a key will be
      replaced by two separators.
  """
  return separator.join([key.replace(separator, separator * 2) for key in keys])


def create_keys_key(key: str) -> str:
  """Creates secondary key representing the sparse keys associated with key."""
  return '_'.join([key, KEYS_SUFFIX])


def create_values_key(key: str) -> str:
  """Creates secondary key representing sparse values associated with key."""
  return '_'.join([key, VALUES_SUFFIX])


def get_by_keys(data: Mapping[str, Any],
                keys: List[Any],
                default_value=None,
                optional: bool = False) -> Any:
  """Returns value with given key(s) in (possibly multi-level) dict.

  The keys represent multiple levels of indirection into the data. For example
  if 3 keys are passed then the data is expected to be a dict of dict of dict.
  For compatibily with data that uses prefixing to create separate the keys in a
  single dict, lookups will also be searched for under the keys separated by
  '/'. For example, the keys 'head1' and 'probabilities' could be stored in a
  a single dict as 'head1/probabilties'.

  Args:
    data: Dict to get value from.
    keys: Sequence of keys to lookup in data. None keys will be ignored.
    default_value: Default value if not found.
    optional: Whether the key is optional or not. If default value is None and
      optional is False then a ValueError will be raised if key not found.

  Raises:
    ValueError: If (non-optional) key is not found.
  """
  if not keys:
    raise ValueError('no keys provided to get_by_keys: %d' % data)

  format_keys = lambda keys: '->'.join([str(k) for k in keys if k is not None])

  value = data
  keys_matched = 0
  for i, key in enumerate(keys):
    if key is None:
      keys_matched += 1
      continue

    if not isinstance(value, Mapping):
      raise ValueError('expected dict for "%s" but found %s: %s' %
                       (format_keys(keys[:i + 1]), type(value), data))

    if key in value:
      value = value[key]
      keys_matched += 1
      continue

    # If values have prefixes matching the key, return those values (stripped
    # of the prefix) instead.
    prefix_matches = {}
    for k, v in value.items():
      if k.startswith(key + '/'):
        prefix_matches[k[len(key) + 1:]] = v
    if prefix_matches:
      value = prefix_matches
      keys_matched += 1
      continue

    break

  if keys_matched < len(keys) or isinstance(value, Mapping) and not value:
    if default_value is not None:
      return default_value
    if optional:
      return None
    raise ValueError('"%s" key not found (or value is empty dict): %s' %
                     (format_keys(keys[:keys_matched + 1]), data))
  return value


def include_filter(
    include: MutableMapping[Any, Any],
    target: MutableMapping[Any, Any]) -> MutableMapping[Any, Any]:
  """Filters target by tree structure in include.

  Args:
    include: Dict of keys from target to include. An empty dict matches all
      values.
    target: Target dict to apply filter to.

  Returns:
    A new dict with values from target filtered out. If a filter key is passed
    that did not match any values, then an empty dict will be returned for that
    key.
  """
  if not include:
    return target

  result = {}
  for key, subkeys in include.items():
    if key in target:
      if subkeys:
        result[key] = include_filter(subkeys, target[key])
      else:
        result[key] = target[key]
  return result


def exclude_filter(
    exclude: MutableMapping[Any, Any],
    target: MutableMapping[Any, Any]) -> MutableMapping[Any, Any]:
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
        value = exclude_filter(exclude[key], target[key])
        if value:
          result[key] = value
    else:
      result[key] = value
  return result


def merge_filters(
    filter1: MutableMapping[Any, Any],
    filter2: MutableMapping[Any, Any]) -> MutableMapping[Any, Any]:
  """Merges two filters together.

  Args:
    filter1: Filter 1.
    filter2: Filter 2

  Returns:
    A new filter with merged values from both filter1 and filter2.
  """
  if (not isinstance(filter1, MutableMapping) or
      not isinstance(filter2, MutableMapping)):
    raise ValueError('invalid filter, non-dict type used as a value: {}'.format(
        [filter1, filter2]))
  if not filter1:
    return filter1
  if not filter2:
    return filter2
  result = {}
  for k in set(filter1.keys()) | set(filter2.keys()):
    if k in filter1 and k in filter2:
      result[k] = merge_filters(filter1[k], filter2[k])
    elif k in filter1:
      result[k] = filter1[k]
    else:
      result[k] = filter2[k]
  return result


def reraise_augmented(exception: Exception, additional_message: str) -> None:
  """Reraise a given exception with additional information.

  Based on _reraise_augmented in Apache Beam.

  Args:
    exception: Original exception.
    additional_message: Additional message to append to original exception's
      message.
  """
  # To emulate exception chaining (not available in Python 2).
  original_traceback = sys.exc_info()[2]
  try:
    # Attempt to construct the same kind of exception
    # with an augmented message.
    #
    # pytype: disable=attribute-error
    # PyType doesn't know that Exception has the args attribute.
    new_exception = type(exception)(
        exception.args[0] + ' additional message: ' + additional_message,
        *exception.args[1:])
    # pytype: enable=attribute-error
  except:  # pylint: disable=bare-except
    # If anything goes wrong, construct a RuntimeError whose message
    # records the original exception's type and message.
    new_exception = RuntimeError(
        traceback.format_exception_only(type(exception), exception)[-1].strip()
        + ' additional message: ' + additional_message)

  six.reraise(type(new_exception), new_exception, original_traceback)


def kwargs_only(fn):
  """Wraps function so that callers must call it using keyword-arguments only.

  Args:
    fn: fn to wrap.

  Returns:
    Wrapped function that may only be called using keyword-arguments.
  """

  if hasattr(inspect, 'getfullargspec'):
    # For Python 3
    args = inspect.getfullargspec(fn)
    varargs = args.varargs
    keywords = args.varkw
  else:
    # For Python 2
    args = inspect.getargspec(fn)  # pylint: disable=deprecated-method
    varargs = args.varargs
    keywords = args.keywords
  if varargs is not None:
    raise TypeError('function to wrap should not have *args parameter')
  if keywords is not None:
    raise TypeError('function to wrap should not have **kwargs parameter')

  arg_list = args.args
  has_default = [False] * len(arg_list)
  default_values = [None] * len(arg_list)
  has_self = arg_list[0] == 'self'
  if args.defaults:
    has_default[-len(args.defaults):] = [True] * len(args.defaults)
    default_values[-len(args.defaults):] = args.defaults

  def wrapped_fn(*args, **kwargs):
    """Wrapped function."""
    if args:
      if not has_self or (has_self and len(args) != 1):
        raise TypeError('function %s must be called using keyword-arguments '
                        'only.' % fn.__name__)

    if has_self:
      if len(args) != 1:
        raise TypeError('function %s has self argument but not called with '
                        'exactly 1 positional argument' % fn.__name__)
      kwargs['self'] = args[0]

    kwargs_to_pass = {}
    for arg_name, arg_has_default, arg_default_value in zip(
        arg_list, has_default, default_values):
      if not arg_has_default and arg_name not in kwargs:
        raise TypeError('function %s must be called with %s specified' %
                        (fn.__name__, arg_name))
      kwargs_to_pass[arg_name] = kwargs.pop(arg_name, arg_default_value)

    if kwargs:
      raise TypeError('function %s called with extraneous kwargs: %s' %
                      (fn.__name__, kwargs.keys()))

    return fn(**kwargs_to_pass)

  return wrapped_fn


def get_features_from_extracts(
    element: types.Extracts
) -> Union[types.DictOfTensorValue, types.DictOfFetchedTensorValues]:
  """Fetch features from the extracts."""
  features = {}
  if constants.FEATURES_PREDICTIONS_LABELS_KEY in element:
    fpl = element[constants.FEATURES_PREDICTIONS_LABELS_KEY]
    if not isinstance(fpl, types.FeaturesPredictionsLabels):
      raise TypeError(
          'Expected FPL to be instance of FeaturesPredictionsLabel. FPL was: '
          '%s of type %s' % (str(fpl), type(fpl)))
    features = fpl.features
  elif constants.FEATURES_KEY in element:
    features = element[constants.FEATURES_KEY]
  return features


def merge_extracts(extracts: List[types.Extracts]) -> types.Extracts:
  """Merges list of extracts into single extract with multi-dimentional data."""

  def merge_with_lists(target: types.Extracts, key: str, value: Any):
    """Merges key and value into the target extracts as a list of values."""
    if isinstance(value, Mapping):
      if key not in target:
        target[key] = {}
      target = target[key]
      for k, v in value.items():
        merge_with_lists(target, k, v)
    else:
      if key not in target:
        target[key] = []
      if isinstance(value, np.ndarray):
        value = value.tolist()
      target[key].append(value)

  def merge_lists(target: types.Extracts) -> types.Extracts:
    """Converts target's leaves which are lists to batched np.array's, etc."""
    if isinstance(target, Mapping):
      result = {}
      for key, value in target.items():
        try:
          result[key] = merge_lists(value)
        except Exception as e:
          raise RuntimeError(
              'Failed to convert value for key "{}"'.format(key)) from e
      return {k: merge_lists(v) for k, v in target.items()}
    elif target and (isinstance(target[0], tf.compat.v1.SparseTensorValue) or
                     isinstance(target[0], types.SparseTensorValue)):
      t = tf.sparse.concat(
          0,
          [tf.sparse.expand_dims(to_tensorflow_tensor(t), 0) for t in target])
      return to_tensor_value(t)
    elif target and isinstance(target[0], types.RaggedTensorValue):
      t = tf.concat(
          [tf.expand_dims(to_tensorflow_tensor(t), 0) for t in target], 0)
      return to_tensor_value(t)
    else:
      arr = np.array(target)
      # Flatten values that were originally single item lists into a single list
      # e.g. [[1], [2], [3]] -> [1, 2, 3]
      if len(arr.shape) == 2 and arr.shape[1] == 1:
        return arr.squeeze(axis=1)
      # Special case for empty slice arrays since numpy treats empty tuples as
      # arrays with dimension 0.
      # e.g. [[()], [()], [()]] -> [(), (), ()]
      elif len(arr.shape) == 3 and arr.shape[1] == 1 and arr.shape[2] == 0:
        return arr.squeeze(axis=1)
      else:
        return arr

  result = {}
  for x in extracts:
    for k, v in x.items():
      merge_with_lists(result, k, v)
  return merge_lists(result)


def split_extracts(extracts: types.Extracts) -> List[types.Extracts]:
  """Splits extracts into a list of extracts along the batch dimension."""
  results = []

  def add_to_results(keys: List[str], values: Any):
    # Use TF to split SparseTensor and RaggedTensorValue
    if isinstance(values, types.SparseTensorValue):
      values = to_tensorflow_tensor(values)
      values = tf.sparse.split(
          sp_input=values, num_split=values.dense_shape[0], axis=0)
      # Drop the batch dimension
      values = [tf.sparse.reshape(v, v.dense_shape[1:]) for v in values]
    elif isinstance(values, types.RaggedTensorValue):
      values = to_tensorflow_tensor(values)
    size = len(values) if hasattr(values, '__len__') else values.shape[0]
    for i in range(size):
      if len(results) <= i:
        results.append({})
      parent = results[i]
      # Values are stored in the leaves of a multi-level dict. Navigate to the
      # parent dict storing the final leaf value that is to be updated.
      for key in keys[:-1]:
        if key not in parent:
          parent[key] = {}
        parent = parent[key]
      value = to_tensor_value(values[i])
      # Scalars should be in array form (e.g. np.array([0.0]) vs np.array(0.0)).
      # The overall slice '()' also needs special handling since numpy encodes
      # it as dimension of 0 size (e.g. compare np.array([()]) vs np.array([0]))
      if (isinstance(value, np.ndarray) and
          (not value.shape or value.shape == (0,))):
        value = np.expand_dims(value, 0)
      parent[keys[-1]] = value

  def visit(subtree: types.Extracts, keys: List[str]):
    for key, value in subtree.items():
      if isinstance(value, Mapping):
        visit(value, keys + [key])
      else:
        add_to_results(keys + [key], value)

  visit(extracts, [])
  return results


class StandardExtracts(collections.abc.MutableMapping):
  """Standard extracts wrap extracts with helpers for accessing common keys.

  Note that the extract values returned may be multi-level dicts depending on
  whether or not multi-model and/or multi-output evalutions were performed.
  """

  def __init__(self, extracts: Optional[types.Extracts] = None, **kwargs):
    """Initializes StandardExtracts.

    Args:
      extracts: Reference to existing extracts to use.
      **kwargs: Name/value pairs to create new extracts from. Only one of either
        extracts or kwargs should be used.
    """
    if extracts is not None and kwargs:
      raise ValueError('only one of extracts or kwargs should be used')
    if extracts is not None:
      self.extracts = extracts
    else:
      self.extracts = kwargs

  def __repr__(self) -> str:
    return repr(self.extracts)

  def __str__(self) -> str:
    return str(self.extracts)

  def __getitem__(self, key):
    return self.extracts[key]

  def __setitem__(self, key, value):
    self.extracts[key] = value

  def __delitem__(self, key):
    del self.extracts[key]

  def __iter__(self):
    return iter(self.extracts)

  def __len__(self):
    return len(self.extracts)

  def get_inputs(self) -> Any:
    """Returns tfma.INPUT_KEY extract."""
    return self[constants.INPUT_KEY]

  inputs = property(get_inputs)

  def get_labels(
      self,
      model_name: Optional[str] = None,
      output_name: Optional[str] = None
  ) -> Optional[types.TensorValueMaybeMultiLevelDict]:
    """Returns tfma.LABELS_KEY extract."""
    return self.get_by_key(constants.LABELS_KEY, model_name, output_name)

  labels = property(get_labels)

  def get_predictions(
      self,
      model_name: Optional[str] = None,
      output_name: Optional[str] = None
  ) -> Optional[types.TensorValueMaybeMultiLevelDict]:
    """Returns tfma.PREDICTIONS_KEY extract."""
    return self.get_by_key(constants.PREDICTIONS_KEY, model_name, output_name)

  predictions = property(get_predictions)

  def get_example_weights(
      self,
      model_name: Optional[str] = None,
      output_name: Optional[str] = None
  ) -> Optional[types.TensorValueMaybeMultiLevelDict]:
    """Returns tfma.EXAMPLE_WEIGHTS_KEY extract."""
    return self.get_by_key(constants.EXAMPLE_WEIGHTS_KEY, model_name,
                           output_name)

  example_weights = property(get_example_weights)

  def get_features(self) -> Optional[types.DictOfTensorValueMaybeDict]:
    """Returns tfma.FEATURES_KEY extract."""
    return self.get_by_key(constants.FEATURES_KEY)

  features = property(get_features)

  def get_transformed_features(
      self,
      model_name: Optional[str] = None
  ) -> Optional[types.DictOfTensorValueMaybeDict]:
    """Returns tfma.TRANSFORMED_FEATURES_KEY extract."""
    return self.get_by_key(constants.TRANSFORMED_FEATURES_KEY, model_name)

  transformed_features = property(get_transformed_features)

  def get_attributions(
      self,
      model_name: Optional[str] = None,
      output_name: Optional[str] = None
  ) -> Optional[types.DictOfTensorValueMaybeDict]:
    """Returns tfma.ATTRIBUTIONS_KEY extract."""
    return self.get_by_key(constants.ATTRIBUTIONS_KEY, model_name, output_name)

  attributions = property(get_attributions)

  def get_by_key(self,
                 key: str,
                 model_name: Optional[str] = None,
                 output_name: Optional[str] = None) -> Any:
    """Returns item for key possibly filtered by model and/or output names."""

    def optionally_get_by_keys(value: Any, keys: List[Any]) -> Any:
      """Returns item in dict (if value is dict and path exists) else value."""
      if isinstance(value, Mapping):
        new_value = get_by_keys(value, keys, optional=True)
        if new_value is not None:
          return new_value
      return value

    value = self[key] if key in self else None
    if model_name is not None:
      value = optionally_get_by_keys(value, [model_name])
    if output_name is not None:
      value = optionally_get_by_keys(value, [output_name])
    return value
