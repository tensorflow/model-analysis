# Lint as: python3
# Copyright 2019 Google LLC
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
"""Utils for metrics."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import inspect
import math

from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Text, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis import util
from tensorflow_model_analysis.metrics import metric_types

from tensorflow_metadata.proto.v0 import schema_pb2

_ALL_CLASSES = 'all_classes'
_PREDICTIONS = 'predictions'
_LOGISTIC = 'logistic'
_PROBABILITIES = 'probabilities'
_LOGITS = 'logits'

_MEAN_METRIC_WRAPPER = 'MeanMetricWrapper'
_LOSS_FUNCTION_WRAPPER = 'LossFunctionWrapper'

_EPSILON = 1e-7


def within_interval(value: float, left: float, right: float) -> bool:
  """Returns true if value is within [left, right]."""
  # EPSILON is used to handle rounding errors that may occur if the value was
  # created using floating point operations.
  return value >= left - _EPSILON and value <= right + _EPSILON


def serialize_metric(metric: tf.keras.metrics.Metric) -> Dict[Text, Any]:
  """Serializes keras metric."""
  cfg = tf.keras.metrics.serialize(metric)
  # If a metric function (vs a class) is passed directly to compile, it
  # will be wrapped in a MeanMetricWrapper which is not deserializable.
  # If this happens, set the class name to the CamelCase from of the
  # function name since most keras metric functions have both forms.
  if ('class_name' in cfg and cfg['class_name'] == _MEAN_METRIC_WRAPPER and
      'config' in cfg and 'name' in cfg['config']):
    cfg['class_name'] = _camel_case(cfg['config']['name'])
  return cfg


def serialize_loss(loss: tf.keras.losses.Loss) -> Dict[Text, Any]:
  """Serializes keras loss."""
  cfg = tf.keras.losses.serialize(loss)
  # If a metric function (vs a class) is passed directly to compile, it
  # will be wrapped in a LossFunctionWrapper which is not deserializable.
  # If this happens, set the class name to the CamelCase from of the
  # function name since most keras loss functions have both forms.
  if ('class_name' in cfg and cfg['class_name'] == _LOSS_FUNCTION_WRAPPER and
      'config' in cfg and 'name' in cfg['config']):
    cfg['class_name'] = _camel_case(cfg['config']['name'])
  return cfg


def _camel_case(txt: Text) -> Text:
  return ''.join(s.capitalize() for s in txt.split('_'))


def to_numpy(tensor: Any) -> np.ndarray:
  """Converts tensor type (list, etc) to np.ndarray if not already."""
  if isinstance(tensor, np.ndarray):
    return tensor
  elif hasattr(tensor, 'numpy'):
    return tensor.numpy()
  else:
    return np.array(tensor)


def to_scalar(
    tensor: Optional[Union[np.ndarray, tf.compat.v1.SparseTensorValue]],
    tensor_name: Text = 'unknown') -> Optional[Union[float, int, Text]]:
  """Returns value as a scalar or raises ValueError."""
  if tensor is None:
    return None
  if isinstance(tensor, tf.compat.v1.SparseTensorValue):
    tensor = tensor.values
  if tensor.size != 1:
    raise ValueError('"{}" should have exactly 1 value, but found {} instead: '
                     'values={}'.format(tensor_name, tensor.size(), tensor))
  return tensor.item()


def pad(arr: np.ndarray, last_dim: int, value: float) -> np.ndarray:
  """Pads the given array with value until last dim is of size last_dim."""
  if arr.shape[-1] == last_dim:
    return arr
  pad_width = []
  for _ in arr.shape[:-1]:
    pad_width.append((0, 0))  # Don't pad inner dimensions
  pad_width.append((0, last_dim - arr.shape[-1]))  # Pad up to last_dim
  return np.pad(
      arr, pad_width=pad_width, mode='constant', constant_values=value)


def to_standard_metric_inputs(
    extracts: types.Extracts,
    include_features: bool = False) -> metric_types.StandardMetricInputs:
  """Filters and converts extracts to StandardMetricInputs."""
  if constants.LABELS_KEY not in extracts:
    raise ValueError('"{}" key not found in extracts. Check that the '
                     'configuration is setup properly to specify the name of '
                     'label input and that the proper extractor has been '
                     'configured to extract the labels from the inputs.'.format(
                         constants.LABELS_KEY))
  if constants.PREDICTIONS_KEY not in extracts:
    raise ValueError('"{}" key not found in extracts. Check that the proper '
                     'extractor has been configured to perform model '
                     'inference.'.format(constants.PREDICTIONS_KEY))
  example_weights = None
  if constants.EXAMPLE_WEIGHTS_KEY in extracts:
    example_weights = extracts[constants.EXAMPLE_WEIGHTS_KEY]
  features = None
  if include_features:
    if constants.FEATURES_KEY not in extracts:
      raise ValueError('"{}" key not found in extracts. Check that the proper '
                       'extractor has been configured to extract the features '
                       'from the inputs.'.format(constants.FEATURES_KEY))
    features = extracts[constants.FEATURES_KEY]
  return metric_types.StandardMetricInputs(extracts[constants.LABELS_KEY],
                                           extracts[constants.PREDICTIONS_KEY],
                                           example_weights, features)


def top_k_indices(
    top_k: int,
    scores: Any,
    sort: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
  """Returns top_k indices into a list of scores.

  Note that the indices are returned in a form that is useful for assigning
  values to the array. If using to select values from an array you may need to
  reshape the output. Examples:

     # Assigning values to scores based on indices
     indices = top_k_indices(1, scores)
     scores[indices] = 0.0

     # Selecting top_k
     indices = top_k_indices(scores)
     scores[indices].reshape(scores.shape[:-1] + (top_k,))

  Args:
    top_k: Number of top k values to return.
    scores: Array or list of scores for computing the top_k indices.
    sort: True if the indices should be sorted (in descending order).

  Returns:
    An array of indices into scores that can be used with either 1D or 2D
    arrays. If sort was True the indices will be returned in descending order of
    score (i.e. top score first).

  Raises:
    ValueError: If top_k doesn't match scores or input has more than 2 dims.
  """
  scores = to_numpy(scores)
  if scores.shape[-1] < top_k:
    raise ValueError(
        'not enough values were provided to perform the requested '
        'calcuations for top k. The requested value for k is {}, but the '
        'values are {}\n\nThis may be caused by a metric configuration error '
        'or an error in the pipeline.'.format(top_k, scores))

  if len(scores.shape) == 1:
    # 1D data
    indices = np.argpartition(scores, -top_k)[-top_k:]
    if sort:
      indices = indices[np.argsort(-scores[indices])]
    return indices
  elif len(scores.shape) == 2:
    # 2D data
    indices = np.argpartition(scores, -top_k, axis=-1)[:, -top_k:]
    # The above creates an n x top_k matrix where each row in indices matches
    # the corresponding row in scores. For example:
    #   [
    #      [<row1_top_k_index_1>, <row_1_top_k_index_2>, ...],
    #      [<row2_top_k_index_1>, <row_2_top_k_index_2>, ...],
    #      ...
    #   ]
    # However numpy indexing wants the index to be be a 2-tuple of where the
    # first tuple value contains the row indices (repeated top k times for each
    # row) and the second tuple value contains the column values.
    #   (row1, row1, ..., row2, ...), (row1_top_k_index1, row1_top_index_2,...)
    if sort:
      for i in range(indices.shape[0]):
        indices[i] = indices[i][np.argsort(-scores[i][indices[i]])]
    return np.arange(indices.shape[0]).repeat(top_k), indices.flatten()
  else:
    raise NotImplementedError(
        'top_k not supported for shapes > 2: scores = {}'.format(scores))


def select_indices(
    arr: np.ndarray,
    indices: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
  """Selects values from tensor at given indices.

  Args:
    arr: Array to select values from.
    indices: Indices that are given either by an np.ndarray (1D) or a tuple of
      np.ndarray's where the first value identifies the rows and the second the
      columns (2D).

  Returns:
    Values with the same shape as tensor except the last dimension will match
    the number of indices selected.
  """
  values = arr[indices]
  if len(arr.shape) == 1:
    return values
  elif len(arr.shape) == 2:
    # The indices[0] contains rows of the form [row1, row1, ..., row2, ...]
    # the rows are repeated for each column. Since the first dimension of the
    # array tells us the number of rows, dividing the length of indices[0] by
    # the number of rows tells us the number of columns we are returning (i.e.
    # the size of the last dim).
    last_dim = int(len(indices[0]) / arr.shape[0])
    values = values.reshape(arr.shape[:-1] + (last_dim,))
    return values
  else:
    raise NotImplementedError('select_indices not supported for shapes > 2: '
                              'arr={}, indices={}'.format(arr, indices))


def to_label_prediction_example_weight(
    inputs: metric_types.StandardMetricInputs,
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    output_name: Text = '',
    sub_key: Optional[metric_types.SubKey] = None,
    aggregation_type: Optional[metric_types.AggregationType] = None,
    class_weights: Optional[Dict[int, float]] = None,
    fractional_labels: bool = False,
    flatten: bool = True,
    squeeze: bool = True,
    allow_none: bool = False,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """Yields label, prediction, and example weights for use in calculations.

  Where applicable this function will perform model and output name lookups as
  well as any required class ID, top K, etc conversions. It will also apply
  prediction keys and label vocabularies given the necessary information is
  provided as part of the EvalConfig (or standard estimator based naming is
  used). The sparseness of labels will be inferred from the shapes of the labels
  and predictions (i.e. if the shapes are different then the labels will be
  assumed to be sparse).

  If successful, the final output of calling this function will be a tuple of
  numpy arrays representing the label, prediction, and example weight
  respectively. Labels and predictions will be returned in the same shape
  provided (default behavior) unless (1) flatten is True in which case a series
  of values (one per class ID) will be returned with last dimension of size 1 or
  (2) a sub_key is used in which case the last dimension may be re-shaped to
  match the new number of outputs (1 for class_id or k, top_k for top k with
  aggregation).

  Note that for top_k without aggregation, the non-top_k prediction values will
  be set to float('-inf'), but for top_k with aggregation the values will be
  truncated to only return the top k values.

  Examples:

    # default behavior
    #
    # Binary classification
    Input  : labels=[1] predictions=[0.6]
    Output : (np.array([1]), np.array([0.6]), np.array([1.0]))
    # Multi-class classification w/ sparse labels
    Input : labels=[2] predictions=[0.3, 0.6, 0.1]
    Output: (np.array([2]), np.array([0.3, 0.6, 0.1]), np.array([1.0]))
    # Multi-class / multi-label classification w/ dense labels
    Input  : labels=[0, 1, 1] predictions=[0.3, 0.6, 0.1]
    Output : (np.array([0, 1, 1]), np.array([0.3, 0.6, 0.1]), np.array([1.0]))

    # flatten=True
    #
    # Multi-class classification w/ sparse labels
    Input  : labels=[2], predictions=[0.3, 0.6, 0.1]
    Output : (np.array([0]), np.array([0.3]), np.array([1.0])),
             (np.array([0]), np.array([0.6]), np.array([1.0])),
             (np.array([1]), np.array([0.1]), np.array([1.0]))
    # Multi-class/multi-label classification w/ dense labels
    Input  : labels=[0, 0, 1], predictions=[0.3, 0.6, 0.1]
    Output : (np.array([0]), np.array([0.3]), np.array([1.0])),
             (np.array([0]), np.array([0.6]), np.array([1.0])),
             (np.array([1]), np.array([0.1]), np.array([1.0]))

    # sub_key.class_id=[2]
    #
    # Multi-class classification w/ sparse labels
    Input  : labels=[2] predictions=[0.3, 0.6, 0.1]
    Output : (np.array([1]), np.array([0.1]), np.array([1.0]))
    # Multi-class classification w/ dense labels
    Input  : labels=[0, 0, 1] predictions=[0.3, 0.6, 0.1]
    Output : (np.array([1]), np.array([0.1]), np.array([1.0]))

    # sub_key.top_k=2 and aggregation_type is None (i.e. binarization of top 2).
    #
    # Multi-class classification w/ sparse labels
    Input  : labels=[2] predictions=[0.3, 0.6, 0.1]
    Output : (np.array([0, 0, 1]), np.array([0.3, 0.6, -inf]), np.array([1.0]))
    # Multi-class classification w/ dense labels
    Input  : labels=[0, 0, 1] predictions=[0.3, 0.1, 0.6]
    Output : (np.array([0, 0, 1]), np.array([0.3, -inf, 0.6]), np.array([1.0]))

    # sub_key.top_k=2 and aggregation_type is not None (i.e. aggregate top 2).
    #
    # Multi-class classification w/ sparse labels
    Input  : labels=[2] predictions=[0.3, 0.6, 0.1]
    Output : (np.array([0, 1]), np.array([0.3, 0.6]), np.array([1.0]))
    # Multi-class classification w/ dense labels
    Input  : labels=[0, 0, 1] predictions=[0.3, 0.1, 0.6]
    Output : (np.array([0, 0]), np.array([0.3, 0.6]), np.array([1.0]))

    # sub_key.k=2 (i.e. binarization by choosing 2nd largest predicted value).
    #
    # Multi-class classification w/ sparse labels
    Input  : labels=[0] predictions=[0.3, 0.6, 0.1]
    Output : (np.array([1]), np.array([0.3]), np.array([1.0]))
    # Multi-class classification w/ dense labels
    Input  : labels=[0] predictions=[0.3]
    Output : (np.array([0]), np.array([0.3]), np.array([1.0]))

  Args:
    inputs: Standard metric inputs.
    eval_config: Eval config
    model_name: Optional model name (if multi-model evaluation).
    output_name: Optional output name (if multi-output model type).
    sub_key: Optional sub key.
    aggregation_type: Optional aggregation type.
    class_weights: Optional class weights to apply to multi-class / multi-label
      labels and predictions. If used, flatten must also be True.
    fractional_labels: If true, each incoming tuple of (label, prediction, and
      example weight) will be split into two tuples as follows (where l, p, w
      represent the resulting label, prediction, and example weight values):
        (1) l = 0.0, p = prediction, and w = example_weight * (1.0 - label)
        (2) l = 1.0, p = prediction, and w = example_weight * label
      If enabled, an exception will be raised if labels are not within [0, 1].
      The implementation is such that tuples associated with a weight of zero
      are not yielded. This means it is safe to enable fractional_labels even
      when the labels only take on the values of 0.0 or 1.0.
    flatten: True to flatten the final label and prediction outputs so that the
      yielded values are always arrays of size 1. For example, multi-class /
      multi-label outputs would be converted into label and prediction pairs
      that could then be processed by a binary classification metric in order to
      compute a micro average over all classes.
    squeeze: True to squeeze any outputs that have rank > 1. This transforms
      outputs such as np.array([[1]]) to np.array([1]).
    allow_none: True to allow labels or predictions with None values to be
      returned. When used, the values will be returned as empty np.ndarrays. The
      example weight will always be non-empty.

  Yields:
    Tuple of (label, prediction, example_weight).
  """

  def optionally_get_by_keys(value: Any, keys: List[Text]) -> Any:
    if isinstance(value, dict):
      new_value = util.get_by_keys(value, keys, optional=True)
      if new_value is not None:
        return new_value
    return value

  prediction_key = ''
  label_key = ''
  if eval_config and eval_config.model_specs:
    for spec in eval_config.model_specs:
      # To maintain consistency between settings where single models are used,
      # always use '' as the model name regardless of whether a name is passed.
      spec_name = spec.name if len(eval_config.model_specs) > 1 else ''
      if spec_name == model_name:
        prediction_key = spec.prediction_key
        label_key = spec.label_key
        break

  label = inputs.label
  if label_key:
    # This is to support a custom EvalSavedModel where the labels are a dict
    # but the keys are not output_names.
    label = optionally_get_by_keys(label, [label_key])
  prediction = inputs.prediction
  example_weight = inputs.example_weight
  if example_weight is None:
    example_weight = np.array(1.0, dtype=np.float32)  # tf-ranking needs float32
  if model_name:
    prediction = util.get_by_keys(prediction, [model_name])
    # Labels and weights can optionally be keyed by model name.
    label = optionally_get_by_keys(label, [model_name])
    example_weight = optionally_get_by_keys(example_weight, [model_name])
  if output_name:
    prediction = util.get_by_keys(prediction, [output_name])
    # Labels and example weights can optionally be keyed by output name.
    label = optionally_get_by_keys(label, [output_name])
    example_weight = optionally_get_by_keys(example_weight, [output_name])
  label, prediction = prepare_labels_and_predictions(label, prediction,
                                                     prediction_key)

  if not allow_none:
    for txt, value in zip(('label', 'prediction'), (label, prediction)):
      if value is None:
        raise ValueError(
            'no value provided for {}: model_name={}, output_name={}, '
            'sub_key={}, aggregation_type={}, StandardMetricInputs={}\n\n'
            'This may be caused by a configuration error (i.e. label, '
            'and/or prediction keys were not specified) or an '
            'error in the pipeline.'.format(txt, model_name, output_name,
                                            sub_key, aggregation_type, inputs))

  example_weight = to_numpy(example_weight)

  # Query based metrics group by a query_id which will result in the
  # example_weight being replicated once for each matching example in the group.
  # When this happens convert the example_weight back to a single value.
  if example_weight.size > 1:
    example_weight = example_weight.flatten()
    if not np.all(example_weight == example_weight[0]):
      raise ValueError(
          'if example_weight size > 0, the values must all be the same: '
          'example_weight={} model_name={}, output_name={}, '
          'sub_key={}, aggregation_type={}, StandardMetricInputs={}'
          '\n\nThis is most likely a configuration error.'.format(
              example_weight, model_name, output_name, sub_key,
              aggregation_type, inputs))
    example_weight = np.array(example_weight[0])

  if sub_key is not None:
    if sub_key.class_id is not None:
      label, prediction = select_class_id(sub_key.class_id, label, prediction)
    elif sub_key.k is not None:
      indices = top_k_indices(sub_key.k, prediction)
      if len(prediction.shape) == 1:
        indices = indices[0]  # 1D
      else:
        # 2D, take kth values
        indices = (indices[0][0::sub_key.k], indices[1][0::sub_key.k])
      if label.shape != prediction.shape:
        label = one_hot(label, prediction)
      label = select_indices(label, indices)
      prediction = select_indices(prediction, indices)
    elif sub_key.top_k is not None:
      # Set all non-top-k predictions to -inf. Note that we do not sort.
      indices = top_k_indices(sub_key.top_k, prediction)
      if aggregation_type is None:
        top_k_predictions = np.full(prediction.shape, float('-inf'))
        top_k_predictions[indices] = prediction[indices]
        prediction = top_k_predictions
      else:
        if label.shape != prediction.shape:
          label = one_hot(label, prediction)
        label = select_indices(label, indices)
        prediction = select_indices(prediction, indices)

  # For consistency, make sure all outputs are arrays (i.e. convert scalars)
  if label is not None and not label.shape:
    label = label.reshape((1,))
  if prediction is not None and not prediction.shape:
    prediction = prediction.reshape((1,))
  if example_weight is not None and not example_weight.shape:
    example_weight = example_weight.reshape((1,))

  if class_weights:
    if not flatten:
      raise ValueError(
          'class_weights can only be used when flatten is also used. This is '
          'likely caused by a configuration error (i.e. micro averaging being '
          "applied to metrics that don't support micro averaging): "
          'class_weights={}, flatten={}, StandardMetricInputs={}'.format(
              class_weights, flatten, inputs))
    example_weight = np.array([
        float(example_weight) * class_weights[i] if i in class_weights else 0.0
        for i in range(prediction.shape[-1] or label.shape[-1])
    ])
  elif flatten:
    example_weight = np.array([
        float(example_weight)
        for i in range(prediction.shape[-1] or label.shape[-1])
    ])

  label = label if label is not None else np.array([])
  prediction = prediction if prediction is not None else np.array([])

  def yield_results(label, prediction, example_weight):
    if (not flatten or (label.size == 0 and prediction.size == 0) or
        (label.size == 1 and prediction.size == 1)):
      if squeeze:
        yield _squeeze(label), _squeeze(prediction), _squeeze(example_weight)
      else:
        yield label, prediction, example_weight
    elif label.size == 0:
      for p, w in (prediction.flatten(), example_weight.flatten()):
        yield label, np.array([p]), np.array([w])
    elif prediction.size == 0:
      for l, w in (label.flatten(), example_weight.flatten()):
        yield np.array([l]), prediction, np.array([w])
    elif label.size == prediction.size:
      for l, p, w in zip(label.flatten(), prediction.flatten(),
                         example_weight.flatten()):
        yield np.array([l]), np.array([p]), np.array([w])
    elif label.shape[-1] == 1:
      label = one_hot(label, prediction)
      for l, p, w in zip(label.flatten(), prediction.flatten(),
                         example_weight.flatten()):
        yield np.array([l]), np.array([p]), np.array([w])
    else:
      raise ValueError(
          'unable to pair labels with predictions: labels={}, predictions={}, '
          'model_name={}, output_name={}, sub_key={}, aggregation_type={}, '
          'StandardMetricInputs={}\n\nThis is most likely a configuration '
          'error.'.format(label, prediction, model_name, output_name, sub_key,
                          aggregation_type, inputs))

  for result in yield_results(label, prediction, example_weight):
    if fractional_labels and label.size:
      for new_result in _yield_fractional_labels(*result):
        yield new_result
    else:
      yield result


def _yield_fractional_labels(
    label: np.ndarray, prediction: np.ndarray, example_weight: np.ndarray
) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
  """Yields (label, prediction, example_weight) applying fractional labels.

  The incoming label, prediction, and example weights will be split into two
  tuples such that if l, p, w represent the resulting tuple values we will get:
    (1) l = 0.0, p = prediction, and w = example_weight * (1.0 - label)
    (2) l = 1.0, p = prediction, and w = example_weight * label

  Args:
    label: Label.
    prediction: Prediction.
    example_weight: Example weight.

  Raises:
    ValueError: If labels are not within [0, 1].
  """
  # Verify that labels are also within [0, 1]
  if not within_interval(float(label), 0.0, 1.0):
    raise ValueError('label must be within [0, 1]: '
                     'label={}, prediction={}, example_weight={}'.format(
                         label, prediction, example_weight))
  for l, w in ((np.array([0], dtype=label.dtype), example_weight * (1 - label)),
               (np.array([1], dtype=label.dtype), example_weight * label)):
    if not math.isclose(w, 0.0):
      yield (l, prediction, w)


def _squeeze(arr: np.ndarray):
  """Squeezes arr while aways returning an array unless 'arr' is a scalar."""
  if arr.shape not in ((), (1,)):
    arr = arr.squeeze()
    if not arr.shape:
      arr = np.expand_dims(arr, axis=0)
  return arr


def prepare_labels_and_predictions(
    labels: Any,
    predictions: Any,
    prediction_key: Optional[Text] = None,
    label_vocabulary: Optional[Union[np.ndarray, List[Text]]] = None
) -> Tuple[np.ndarray, np.ndarray]:
  """Prepares labels and predictions for use in calculations.

  If the predictions are a dict (i.e. estimator based output) this function will
  apply the necessary lookup based on the prediction_key provided (or using a
  default set of common keys such as 'probabilities', etc). Note that the
  predictions passed as args must be AFTER the model_name and/or output_name
  lookups have been performed. This function also applies any label vocabulary
  transformations where possible.

  If successful, the final output of calling this function will be a pair of
  numpy arrays representing the labels and predictions.

  Args:
    labels: List, np.ndarray, or SparseTensorValue of values (1D, 2D, or 3D).
    predictions: List or np.ndarray of prediction values (1D, 2D, or 3D) or a
      dict of prediction values keyed by prediction_key or common estimator keys
      (logistic, probabilties, etc).
    prediction_key: Optional predictions key. Used when the predict output is a
      dict.
    label_vocabulary: Optional label vocabulary to convert label values to ints
      (if prediction is a dict containing an 'all_classes' key that will be used
      if label_vocabulary is None).

  Returns:
    A (labels, predictions) tuple suitable for metric calculations.

  Raises:
    ValueError: If the labels or predictions are in an invalid format.
  """
  if isinstance(predictions, dict):
    if label_vocabulary is None:
      if _ALL_CLASSES in predictions:
        # Check for use of estimator label vocab under ALL_CLASSES. This was
        # added in 06/2019 for eval signatures because the CLASSES only contains
        # the label for the chosen class.
        label_vocabulary = to_numpy(predictions[_ALL_CLASSES])
      elif (tf.saved_model.CLASSIFY_OUTPUT_SCORES in predictions and
            tf.saved_model.CLASSIFY_OUTPUT_CLASSES in predictions):
        # For classification model using the default serving signature, the
        # CLASSES contains the full vocabulary. The check for scores is needed
        # here to avoid matching CLASSES in the eval case (scores are not used
        # in eval).
        label_vocabulary = to_numpy(
            predictions[tf.saved_model.CLASSIFY_OUTPUT_CLASSES])
      if label_vocabulary is not None:
        while len(label_vocabulary.shape) > 1:
          label_vocabulary = label_vocabulary[0]  # Remove the bach dimensions
    if not prediction_key:
      # Estimator predictions use dicts of scores, probabilities, classes, etc.
      for k in (tf.saved_model.CLASSIFY_OUTPUT_SCORES,
                tf.saved_model.REGRESS_OUTPUTS, _PREDICTIONS, _LOGISTIC,
                _PROBABILITIES, _LOGITS):
        if k in predictions:
          predictions = predictions[k]
          prediction_key = k
          break
    elif prediction_key in predictions:
      predictions = predictions[prediction_key]

  if isinstance(labels, dict) or isinstance(predictions, dict):
    raise ValueError('unable to prepare labels and predictions because the '
                     'labels and/or predictions are dicts with unrecognized '
                     'keys. If a multi-output keras model (or estimator) was '
                     'used check that an output_name was provided. If an '
                     'estimator was used check that common prediction keys '
                     'were provided (e.g. logistic, probabilities, etc): '
                     'labels={}, predictions={}, prediction_key={}'.format(
                         labels, predictions, prediction_key))

  if predictions is not None:
    predictions = to_numpy(predictions)

  if labels is not None:
    if isinstance(labels, tf.compat.v1.SparseTensorValue):
      if predictions is None or predictions.size == 0:
        raise ValueError('predictions must also be used if labels are of type '
                         'SparseTensorValue: labels={}'.format(labels))
      values = labels.values if labels.values is not None else np.array([])
      indices = labels.indices if labels.indices is not None else np.array([])
      if label_vocabulary is not None and values.dtype.kind in ('U', 'S', 'O'):
        values = _string_labels_to_class_ids(label_vocabulary, values)
        # If vocab is used then the values will be the indices into the vocab
        # and we should use multi-hot encoding to store the output. We can
        # accomplish this by passing 1's for the values and using the values
        # converted from the vocab as the indices to insert the 1's at the
        # proper offsets in the resulting multi-hot vector.
        labels = _to_dense_tensor(
            np.ones(values.shape), values, predictions.shape)
      else:
        labels = _to_dense_tensor(values, indices, predictions.shape)
    else:
      labels = to_numpy(labels)
      if label_vocabulary is not None and labels.dtype.kind in ('U', 'S', 'O'):
        labels = _string_labels_to_class_ids(label_vocabulary, labels)

  return (labels, predictions)


def _to_dense_tensor(values: np.ndarray, indices: np.ndarray,
                     dense_shape: Tuple[int, ...]) -> np.ndarray:
  """Converts sparse tensor to dense given values, indices, and dense shape."""
  # Squeeze is used on the values, indices, and result to ensure that single
  # value inputs that still have the batch dimension such as [1, n_classes] can
  # still be indexed properly from SparseTensorValues that don't use batching.
  result = _squeeze(np.zeros(dense_shape))
  for value, index in zip(_squeeze(values), _squeeze(indices)):
    result[index] = value
  return result.reshape(dense_shape)


def _string_labels_to_class_ids(label_vocabulary: Union[np.ndarray, List[Text]],
                                labels: np.ndarray) -> np.ndarray:
  """Returns class ID for given string label using given classes or -1."""

  def lookup(label):
    for i, c in enumerate(label_vocabulary):
      if c == label:
        return i
    return -1

  return np.array([lookup(l) for l in labels.flatten()]).reshape(labels.shape)


def select_class_id(
    class_id: int,
    labels: Any,
    predictions: Any,
    sparse_labels: bool = None,
) -> Tuple[np.ndarray, np.ndarray]:
  """Selects values for given class ID from multi-class labels and predictions.

  Args:
    class_id: Class ID to filter the labels and predictions by.
    labels: Array or list of processed labels (1D, 2D, or 3D).
    predictions: Array or list of processed predictions (1D, 2D, or 3D).
    sparse_labels: True if sparse labels are being used. If None then the
      sparseness will be inferred from the shapes of the labels and predictions
      (i.e. if the shapes are different then the labels will be assumed to be
      sparse).

  Returns:
    A (labels, predictions) tuple with the predictions returned in the same form
    as the originals (except for the last dimension which will be 1).

  Raises:
    ValueError: If the labels or predictions cannot be formatted properly.
  """
  labels = to_numpy(labels)
  predictions = to_numpy(predictions)
  if labels.size == 0 or predictions.size == 0:
    return (labels, predictions)

  def lookup(arr, target):
    if class_id < 0 or class_id >= len(arr):
      raise ValueError('class_id "{}" out of range of {}: {}'.format(
          class_id, target, arr))
    return arr[class_id]

  # Convert scalars to arrays
  if not labels.shape:
    labels = labels.reshape((1,))
  if not predictions.shape:
    predictions = predictions.reshape((1,))

  sparse_labels = _verify_sparse_labels(
      labels, predictions, sparse_labels=sparse_labels)
  if sparse_labels and labels.shape[-1] != 1:
    # Convert to [[class_id1], ...]
    labels = labels.reshape((-1, 1))

  labels_out_shape = list(labels.shape)
  labels_out_shape[-1] = 1
  predictions_out_shape = list(predictions.shape)
  predictions_out_shape[-1] = 1

  # Convert labels and predictions into the form ([[...], [...]])
  if len(labels.shape) > 1:
    # Flatten all but the last dim (a, b, c) -> (a * b, c)
    labels = labels.reshape((-1, labels.shape[-1]))
  else:
    labels = labels.reshape((1, labels.shape[0]))
  if len(predictions.shape) > 1:
    predictions = predictions.reshape((-1, predictions.shape[-1]))
  else:
    predictions = predictions.reshape((1, predictions.shape[0]))

  if sparse_labels:
    # Labels are of the form [[class_id1], [class_id2], ...]
    labels = np.array([int(l[0] == class_id) for l in labels])
  else:
    # Labels are of the form [[0, 0, 1, ...], [0, 0, 0, ...], ...]
    labels = np.array([lookup(l, 'labels') for l in labels])
  predictions = np.array([lookup(p, 'predictions') for p in predictions])

  return (labels.reshape(labels_out_shape),
          predictions.reshape(predictions_out_shape))


def _verify_sparse_labels(labels: np.ndarray,
                          predictions: np.ndarray,
                          sparse_labels: bool = None) -> bool:
  """Checks if labels are sparse or not.

  Args:
    labels: Numpy array of labels.
    predictions: Numpy array of predictions.
    sparse_labels: True if sparse labels should be used. If None then the
      sparseness will be inferred from the shapes of the labels and predictions
      (i.e. if the shapes are different then the labels will be assumed to be
      sparse).

  Returns:
    True if sparse.

  Raises:
    ValueError: If the sparse_labels setting does not match labels and
      predictions.
  """
  if (len(labels.shape) != len(predictions.shape) or
      labels.shape[-1] != predictions.shape[-1]):
    # Labels are of the form [class_id1, ...]
    #
    # Note that it is possible that the labels could be multi-label of the form
    # [[class_id1, class_id2], [class_id3, ...]]. However, this would require a
    # ragged or sparse tensor input to support. Ragged data in np.array is
    # encoded as a list object which doesn't accurately reflect the shape (i.e.
    # np.array([[1, 2], [3]]) has shape (2,). As such we will assume that all
    # multi-label use cases will use one-hot encodings (e.g. [0, 1. 1, ...]) and
    # will update this if/when RaggedTensorValue and SparseTensorValue value
    # types are supported in addition to np.ndarray.
    if sparse_labels is not None and not sparse_labels:
      raise ValueError(
          'The number of labels = 1, but sparse labels are not being used\n\n'
          'This is likley caused by a metric configuration error. Change to '
          'use a non-sparse versions of the metrics or ensure that sparse '
          'labels are passed as input: labels={}, predictions={}'.format(
              labels, predictions))
    return True
  else:
    # Labels are of the form [0, 0, 1, ...] (i.e. one-hot). This includes
    # regression and binary classification.
    #
    # Similar to the note above, this is only true if multi-label inputs are
    # always encoded as one-hot since it is possible for a multi-label encoding
    # using [class_id1, ...] to use all the classes and therefore match the
    # prediction's last dimension in length.
    if sparse_labels is not None and sparse_labels:
      raise ValueError(
          'The number of labels > 1, but sparse labels are being used\n\n'
          'This is likley caused by a metric configuration error. Change to '
          'use sparse versions of the metrics or ensure that non-sparse labels '
          'are passed as input: labels={}, predictions={}'.format(
              labels, predictions))
    return False


def one_hot(tensor: np.ndarray, target: np.ndarray) -> np.ndarray:
  """Convert tensor's last dimension into a one-hot vector of target's shape.

  Args:
    tensor: Tensor to convert to one-hot vector. Must have no shape or a final
      dimension of size 1.
    target: Target tensor to reshape the tensor to.

  Returns:
    Tensor with last dimension encoded as a one-hot vector with the overall
    shape the same as that of target.
  """
  # For values that are OOV (i.e. set to -1) we will use a vector of all 0's.
  # When np.eye is indexed by -1, a value of all 0's followed by 1 is used for
  # the row. The following handles -1 values by adding an additional column for
  # indexing the -1 and then removing it after.
  tensor = np.delete(np.eye(target.shape[-1] + 1)[tensor], -1, axis=-1)
  return tensor.reshape(target.shape)


def merge_per_key_computations(
    create_computations_fn: Callable[..., metric_types.MetricComputations],
) -> Callable[..., metric_types.MetricComputations]:
  """Wraps create_computations_fn to be called separately for each key."""

  def merge_computations_fn(
      eval_config: Optional[config.EvalConfig] = None,
      schema: Optional[schema_pb2.Schema] = None,
      model_names: Optional[List[Text]] = None,
      output_names: Optional[List[Text]] = None,
      sub_keys: Optional[List[Optional[metric_types.SubKey]]] = None,
      aggregation_type: Optional[metric_types.AggregationType] = None,
      class_weights: Optional[Dict[int, float]] = None,
      query_key: Optional[Text] = None,
      **kwargs) -> metric_types.MetricComputations:
    """Merge computations function."""
    if model_names is None:
      model_names = ['']
    if output_names is None:
      output_names = ['']
    if sub_keys is None:
      sub_keys = [None]
    computations = []
    for model_name in model_names:
      for output_name in output_names:
        for sub_key in sub_keys:
          if hasattr(inspect, 'getfullargspec'):
            args = inspect.getfullargspec(create_computations_fn).args
          else:
            args = inspect.getargspec(create_computations_fn).args  # pylint: disable=deprecated-method
          updated_kwargs = metric_types.update_create_computations_fn_kwargs(
              args, kwargs.copy(), eval_config, schema, model_names, sub_keys,
              aggregation_type, class_weights, query_key)
          if 'model_name' in args:
            updated_kwargs['model_name'] = model_name
          if 'output_name' in args:
            updated_kwargs['output_name'] = output_name
          if 'sub_key' in args:
            updated_kwargs['sub_key'] = sub_key
          computations.extend(create_computations_fn(**updated_kwargs))
    return computations

  return merge_computations_fn
