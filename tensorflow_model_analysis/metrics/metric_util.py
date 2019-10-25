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
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis import util
from tensorflow_model_analysis.metrics import metric_types
from typing import Any, Callable, List, Optional, Text, Tuple, Union

_ALL_CLASSES = 'all_classes'
_PREDICTIONS = 'predictions'
_LOGISTIC = 'logistic'
_PROBABILITIES = 'probabilities'
_LOGITS = 'logits'


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
  return np.asscalar(tensor)


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


def to_label_prediction_example_weight(
    inputs: metric_types.StandardMetricInputs,
    eval_config: Optional[config.EvalConfig] = None,
    model_name: Text = '',
    output_name: Text = '',
    sub_key: Optional[metric_types.SubKey] = None,
    allow_none: bool = False,
    array_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns label, prediction, and example weight for use in calculations.

  Where applicable this function will perform model and output name lookups as
  well as any required class ID, top K, etc conversions. It will also apply
  prediction keys and label vocabularies given the necessary information is
  provided as part of the EvalConfig (or standard estimator based naming is
  used).

  If successful, the final output of calling this function will be a tuple of
  numpy arrays representing the label, prediction, and example weight
  respectively.

  Args:
    inputs: Standard metric inputs.
    eval_config: Eval config
    model_name: Optional model name (if multi-model evaluation).
    output_name: Optional output name (if multi-output model type).
    sub_key: Optional sub key.
    allow_none: True to allow labels or predictions with None values to be
      returned. The example weight will always be non-None.
    array_size: Verifies the prediction and labels are of the given size. If
      both array_size and sub_key.top_k is set, then array_size will be ignored
      and the size will be verified based on the top_k setting. The example
      weight will always be size 1.
  """

  def optionally_get_by_keys(value: Any, keys: List[Text]) -> Any:
    if isinstance(value, dict):
      new_value = util.get_by_keys(value, keys, optional=True)
      if new_value is not None:
        return new_value
    return value

  label = inputs.label
  prediction = inputs.prediction
  example_weight = inputs.example_weight
  if example_weight is None:
    example_weight = np.array(1.0)
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
  prediction_key = ''
  if eval_config and eval_config.model_specs:
    for spec in eval_config.model_specs:
      if spec.name == model_name:
        prediction_key = spec.prediction_key
        break
  label, prediction = prepare_labels_and_predictions(label, prediction,
                                                     prediction_key)

  if sub_key is not None:
    if sub_key.class_id is not None:
      label, prediction = select_class_id(sub_key.class_id, label, prediction)
    elif sub_key.k is not None:
      label, prediction = select_top_k(sub_key.k, label, prediction)
      label = np.array([label[sub_key.k - 1]])
      prediction = np.array([prediction[sub_key.k - 1]])
    elif sub_key.top_k is not None:
      label, prediction = select_top_k(sub_key.top_k, label, prediction)

  example_weight = to_numpy(example_weight)

  if not allow_none:
    for txt, value in zip(('label', 'prediction', 'example_weight'),
                          (label, prediction, example_weight)):
      if value is None:
        raise ValueError(
            'no value provided for {}: model_name={}, output_name={}, '
            'sub_key={}, StandardMetricInputs={}\n\n'
            'This may be caused by a configuration error (i.e. label, '
            'prediction, and/or example weight keys were not specified) or an '
            'error in the pipeline.'.format(txt, model_name, output_name,
                                            sub_key, inputs))

  for txt, value in zip(('label', 'prediction', 'example_weight'),
                        (label, prediction, example_weight)):
    if value is None:
      continue
    if txt == 'example_weight':
      size = 1
    elif array_size is None:
      continue
    elif sub_key and sub_key.top_k is not None:
      size = sub_key.top_k
    else:
      size = array_size
    if value.size != size:
      raise ValueError(
          'expected {} to be size = {}, but instead it has size = {}: '
          '{}={}, model_name={}, output_name={}, sub_key={}, '
          'StandardMetricInputs={}\n\nThis is most likely a configuration '
          'error (for multi-class models using binary classification '
          'metrics, a sub_key must be set).'.format(txt, size, value.size, txt,
                                                    value, model_name,
                                                    output_name, sub_key,
                                                    inputs))

  # For consistency, make sure all outputs are arrays (i.e. convert scalars)
  if label is not None and not label.shape:
    label = label.reshape((1,))
  if prediction is not None and not prediction.shape:
    prediction = prediction.reshape((1,))
  if example_weight is not None and not example_weight.shape:
    example_weight = example_weight.reshape((1,))
  return label, prediction, example_weight


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
    labels: List or np.ndarray of label values (1D, 2D, or 3D).
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

  if labels is not None:
    labels = to_numpy(labels)
  if predictions is not None:
    predictions = to_numpy(predictions)

  if (labels is None or predictions is None or labels.size == 0 or
      predictions.size == 0):
    return (labels, predictions)

  if label_vocabulary is not None and labels.dtype.kind in ('U', 'S'):
    labels = _string_labels_to_class_ids(label_vocabulary, labels)

  # Classify scores contain two values intead of one for binary classification
  # problems, choose top prediction.
  if (prediction_key == tf.saved_model.CLASSIFY_OUTPUT_SCORES and
      predictions.shape[-1] == 2):
    predictions = np.amax(predictions, -1, keepdims=True)

  return (labels, predictions)


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
) -> Tuple[np.ndarray, np.ndarray]:
  """Selects values for given class ID from multi-class labels and predictions.

  Args:
    class_id: Class ID to filter the labels and predictions by.
    labels: Array or list of processed labels (1D, 2D, or 3D).
    predictions: Array or list of processed predictions (1D, 2D, or 3D).

  Returns:
    A (labels, predictions) tuple with the predictions returned in the same form
    as the labels.

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

  out_shape = list(labels.shape)
  if len(out_shape) > 1 and out_shape[-1] > 1:
    # Labels are of the form [[0, 0, 1, ...], ..] so the last dim will be
    # reduced to 1 later, so squeeze this out of the output.
    out_shape = out_shape[:-1]

  # Flatten all but the last dim (a, b, c) -> (a * b, c)  (e.g. [[...], [...]])
  if len(labels.shape) == 1:
    # Labels are of the form [class_id1, class_id2, ...]. Assume this is for a
    # batch of labels and convert to the form [[class_id1], [class_id2], ...]
    # for looking up the class IDs. This will be reshaped back to single
    # dimension later.
    labels = labels.reshape((-1, 1))
  else:
    labels = labels.reshape((-1, labels.shape[-1]))
  if len(predictions.shape) == 1:
    # Predictions are of the form [p1, p2, ...]. Assume this is for a single
    # multi-class prediction and convert to the form [[p1, p2, ...]] for looking
    # up the class IDs. This will be reshaped to match the labels later.
    predictions = predictions.reshape((1, predictions.shape[0]))
  predictions = predictions.reshape((-1, predictions.shape[-1]))

  if labels.shape[1] == 1:
    # Labels are of the form [[class_id1], [class_id2], ...]
    labels = np.array([int(l[0] == class_id) for l in labels])
  else:
    # Labels are of the form [[0, 0, 1, ...], [0, 0, 0, ...], ...]
    labels = np.array([lookup(l, 'labels') for l in labels])
  predictions = np.array([lookup(p, 'predictions') for p in predictions])

  return (labels.reshape(out_shape), predictions.reshape(out_shape))


def select_top_k(top_k: int,
                 labels: Any,
                 predictions: Any,
                 scores: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
  """Selects top_k values from multi-class labels and predictions.

  Args:
    top_k: Number of top k values to return.
    labels: Array or list of processed labels (1D, 2D, or 3D).
    predictions: Array or list of processed predictions (1D, 2D, or 3D).
    scores: Optional array or list of scores for computing the top_k values. If
      scores is not set then predictions will be used.

  Returns:
    A (labels, predictions) tuple. Both values will have the same shape as the
    predictions but with the last dimension of size top_k. The prediction values
    will be returned in descending order of score (i.e. top prediction first).

  Raises:
    ValueError: If the labels or predictions cannot be formatted properly.
  """
  labels = to_numpy(labels)
  predictions = to_numpy(predictions)
  if labels.size == 0 or predictions.size == 0:
    return (labels, predictions)

  if predictions.shape[-1] <= 2:
    raise ValueError('select_top_k requires predictions.shape > 2: '
                     'predictions = {}'.format(predictions))

  if scores is not None:
    scores = to_numpy(scores)
    if scores.shape != predictions.shape:
      raise ValueError(
          'predictions and scores must have the same shape {} != {}: '
          'predictions={}, scores={}'.format(predictions.shape, scores.shape,
                                             predictions, scores))

  if not labels.shape or labels.shape[-1] == 1:
    # Convert labels into one-hot vectors. For labels that are OOV (i.e. set to
    # -1) we will use a vector of all 0's. When np.eye is indexed by -1, a
    # value of all 0's followed by 1 is used for the row. The following handles
    # -1 values by adding an additional column for indexing the -1 and then
    # removing it after.
    labels = np.delete(np.eye(predictions.shape[-1] + 1)[labels], -1, axis=-1)
    labels = labels.reshape(predictions.shape)

  if scores is None:
    scores = predictions

  if scores.shape[-1] < top_k:
    raise ValueError(
        'not enough predictions were provided to perform the requested '
        'calcuations for top k. The requested value for k is {}, but the '
        'values are {}\n\nThis may be caused by a metric configuration error '
        'or an error in the pipeline.'.format(top_k, scores))

  if len(predictions.shape) == 1:
    # 1D data
    indices = np.argpartition(scores, -top_k)[-top_k:]
    indices = indices[np.argsort(-scores[indices])]
    return (labels[indices], predictions[indices])
  elif len(predictions.shape) == 2:
    # Batched 2D data
    out_shape = predictions.shape[:-1] + (top_k,)
    out_labels = np.empty(out_shape)
    out_predictions = np.empty(out_shape)
    indices = np.argpartition(scores, -top_k, axis=-1)[:, -top_k:]
    for i in range(predictions.shape[0]):
      for j, idx in enumerate(indices[i][np.argsort(-scores[i][indices[i]])]):
        out_labels[i][j] = labels[i][idx]
        out_predictions[i][j] = predictions[i][idx]
    return (out_labels, out_predictions)
  else:
    raise NotImplementedError(
        'select_top_k not supported for shapes > 2: predictions = {}'.format(
            predictions))


def merge_per_key_computations(
    create_computations_fn: Callable[..., metric_types.MetricComputations],
) -> metric_types.MetricComputations:
  """Wraps create_computations_fn to be called separately for each key."""

  def merge_computations_fn(eval_config: Optional[config.EvalConfig] = None,
                            model_names: Optional[List[Text]] = None,
                            output_names: Optional[List[Text]] = None,
                            sub_keys: Optional[List[Text]] = None,
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
            args = inspect.getargspec(create_computations_fn).args
          updated_kwargs = kwargs.copy()
          if 'eval_config' in args:
            updated_kwargs['eval_config'] = eval_config
          if 'model_name' in args:
            updated_kwargs['model_name'] = model_name
          if 'output_name' in args:
            updated_kwargs['output_name'] = output_name
          if 'sub_key' in args:
            updated_kwargs['sub_key'] = sub_key
          if 'query_key' in args:
            updated_kwargs['query_key'] = query_key
          computations.extend(create_computations_fn(**updated_kwargs))
    return computations

  return merge_computations_fn
