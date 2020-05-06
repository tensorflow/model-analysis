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
"""Utils for working with models."""

# Standard __future__ imports

import collections
import datetime
import os

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Text

from absl import logging
import apache_beam as beam
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import constants as eval_constants
from tensorflow_model_analysis.eval_saved_model import load

KERAS_INPUT_SUFFIX = '_input'

_TFLITE_FILE_NAME = 'tflite'


class ModelContents(object):
  """Class for storing model contents.

  This class exists because weak references to bytes are not allowed.
  """
  __slots__ = ['contents', '__weakref__']

  def __init__(self, contents: bytes):
    self.contents = contents


def get_baseline_model_spec(
    eval_config: config.EvalConfig) -> Optional[config.ModelSpec]:
  """Returns baseline model spec."""
  for spec in eval_config.model_specs:
    if spec.is_baseline:
      return spec
  return None


def get_model_spec(eval_config: config.EvalConfig,
                   model_name: Text) -> Optional[config.ModelSpec]:
  """Returns model spec with given model name."""
  if len(eval_config.model_specs) == 1 and not model_name:
    return eval_config.model_specs[0]
  for spec in eval_config.model_specs:
    if spec.name == model_name:
      return spec
  return None


def get_label_key(model_spec: config.ModelSpec,
                  output_name: Text) -> Optional[Text]:
  """Returns the label_key corresponding to a given output name."""
  if output_name:
    if model_spec.label_key:
      return model_spec.label_key
    elif model_spec.label_keys:
      return model_spec.label_keys[output_name]
    else:
      return None
  else:
    if model_spec.label_key:
      return model_spec.label_key
    elif model_spec.label_keys:
      raise ValueError('When setting label_keys in a model spec, all metrics '
                       'specs for that model must specify an output_name.')
    else:
      return None


def get_model_type(model_spec: config.ModelSpec,
                   model_path: Optional[Text] = '',
                   tags: Optional[List[Text]] = None) -> Text:
  """Returns model type for given model spec taking into account defaults.

  The defaults are chosen such that if a model_path is provided and the model
  can be loaded as a keras model then TF_KERAS is assumed. Next, if tags
  are provided and the tags contains 'eval' then TF_ESTIMATOR is assumed.
  Lastly, if the model spec contains an 'eval' signature TF_ESTIMATOR is assumed
  otherwise TF_GENERIC is assumed.

  Args:
    model_spec: Model spec.
    model_path: Optional model path to verify if keras model.
    tags: Options tags to verify if eval is used.
  """
  if model_spec and model_spec.model_type:
    return model_spec.model_type

  if model_path:
    try:
      keras_model = tf.keras.models.load_model(model_path)
      # In some cases, tf.keras.models.load_model can successfully load a
      # saved_model but it won't actually be a keras model.
      if isinstance(keras_model, tf.keras.models.Model):
        return constants.TF_KERAS
    except Exception:  # pylint: disable=broad-except
      pass

  if tags:
    if tags and eval_constants.EVAL_TAG in tags:
      return constants.TF_ESTIMATOR
    else:
      return constants.TF_GENERIC

  signature_names = []
  if model_spec:
    if model_spec.signature_name:
      signature_names.append(model_spec.signature_name)
    elif model_spec.signature_names:
      for signature_name in model_spec.signature_names:
        signature_names.append(signature_name)
    else:
      signature_names.append(tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY)

  # Default to serving unless estimator is used and eval signature is used.
  if eval_constants.EVAL_TAG in signature_names:
    return constants.TF_ESTIMATOR
  else:
    return constants.TF_GENERIC


def verify_and_update_eval_shared_models(
    eval_shared_model: Optional[types.MaybeMultipleEvalSharedModels]
) -> Optional[List[types.EvalSharedModel]]:
  """Verifies eval shared models and normnalizes to produce a single list.

  The output is normalized such that if a list or dict contains a single entry,
  the model name will always be empty.

  Args:
    eval_shared_model: None, a single model, a list of models, or a dict of
      models keyed by model name.

  Returns:
    A list of models.

  Raises:
    ValueError if dict is passed and keys don't match model names or a
    multi-item list is passed without model names.
  """
  if not eval_shared_model:
    return None
  eval_shared_models = []
  if isinstance(eval_shared_model, dict):
    for k, v in eval_shared_model.items():
      if v.model_name and k and k != v.model_name:
        raise ValueError('keys for EvalSharedModel dict do not match '
                         'model_names: dict={}'.format(eval_shared_model))
      if not v.model_name and k:
        v = v._replace(model_name=k)
      eval_shared_models.append(v)
  elif isinstance(eval_shared_model, list):
    eval_shared_models = eval_shared_model
  else:
    eval_shared_models = [eval_shared_model]
  if len(eval_shared_models) > 1:
    for v in eval_shared_models:
      if not v.model_name:
        raise ValueError(
            'model_name is required when passing multiple EvalSharedModels: '
            'eval_shared_models={}'.format(eval_shared_models))
  # To maintain consistency between settings where single models are used,
  # always use '' as the model name regardless of whether a name is passed.
  elif len(eval_shared_models) == 1 and eval_shared_models[0].model_name:
    eval_shared_models[0] = eval_shared_models[0]._replace(model_name='')
  return eval_shared_models


def rebatch_by_input_names(
    batch_of_extracts: List[types.Extracts],
    input_names: List[Text],
    input_specs: Optional[Dict[Text, tf.TypeSpec]] = None) -> Dict[Text, Any]:
  """Converts a batch of extracts into multiple batches keyed by input names.

  Args:
    batch_of_extracts: Batch of extracts (one per example).
    input_names: List of input names to search for features under.
    input_specs: Optional list of type specs associated with inputs.

  Returns:
    Dict of batch aligned features keyed by input (feature) name.
  """
  # TODO(b/138474171): Make this code more efficient.
  if input_specs is None:
    input_specs = {}
  inputs = collections.defaultdict(list)
  found = {}
  for name in input_names:
    for extract in batch_of_extracts:
      # If features key exist, use that for features, else use input_key
      if constants.FEATURES_KEY in extract:
        input_features = extract[constants.FEATURES_KEY]
      else:
        input_features = extract[constants.INPUT_KEY]
      if isinstance(input_features, dict):
        value = None
        if name in input_features:
          found[name] = True
          value = input_features[name]
        # Some keras models prepend '_input' to the names of the inputs
        # so try under '<name>_input' as well.
        elif (name.endswith(KERAS_INPUT_SUFFIX) and
              name[:-len(KERAS_INPUT_SUFFIX)] in input_features):
          found[name] = True
          value = input_features[name[:-len(KERAS_INPUT_SUFFIX)]]
        if value is not None:
          # If the expected input shape contains only the batch dimension
          # then we need to flatten the np.array. This it to handle tf_hub
          # cases where the inputs can have a single dimension.
          if name in input_specs and len(input_specs[name].shape) == 1:
            if value.size != 1:
              raise ValueError(
                  'model expects inputs with shape (?,), but shape is '
                  '{}: input_names={} input_specs={}, extract={}'.format(
                      value.shape, input_names, input_specs, extract))
            inputs[name].append(value.item())
          else:
            inputs[name].append(value)
      else:
        # Check that we have not previously added inputs before.
        if inputs:
          raise ValueError(
              'only a single input was passed, but model expects multiple: '
              'input_names = {}, extract={}'.format(input_names, extract))
        found[name] = True
        inputs[name].append(input_features)
  if len(found) != len(input_names):
    logging.log_first_n(
        logging.WARNING,
        'inputs do not match those expected by the model: input_names=%s, '
        'found in extracts=%s', 1, input_names, found)
  return inputs


def find_input_name_in_features(features: Set[Text],
                                input_name: Text) -> Optional[Text]:
  """Maps input name to an entry in features. Returns None if not found."""
  if input_name in features:
    return input_name
  # Some keras models prepend '_input' to the names of the inputs
  # so try under '<name>_input' as well.
  elif (input_name.endswith(KERAS_INPUT_SUFFIX) and
        input_name[:-len(KERAS_INPUT_SUFFIX)] in features):
    return input_name[:-len(KERAS_INPUT_SUFFIX)]
  return None


def filter_tensors_by_input_names(
    tensors: Dict[Text,
                  Any], input_names: List[Text]) -> Optional[Dict[Text, Any]]:
  """Filter tensors by input names.

  In case we don't find the specified input name in the tensors and there
  exists only one input name, we assume we are feeding serialized examples to
  the model and return None.

  Args:
    tensors: Dict of tensors.
    input_names: List of input names.

  Returns:
    Filtered tensors.

  Raises:
    RuntimeError: When the specified input tensor cannot be found.
  """
  if not input_names:
    return None
  result = {}
  tensor_keys = set(tensors.keys())
  for name in input_names:
    tensor_name = find_input_name_in_features(tensor_keys, name)
    if tensor_name is None:
      # This should happen only in the case where the model takes serialized
      # examples as input. Else raise an exception.
      if len(input_names) == 1:
        return None
      raise RuntimeError(
          'Input tensor not found: {}. Existing keys: {}.'.format(
              name, ','.join(tensors.keys())))
    result[name] = tensors[tensor_name]
  return result


def model_construct_fn(  # pylint: disable=invalid-name
    eval_saved_model_path: Optional[Text] = None,
    add_metrics_callbacks: Optional[List[types.AddMetricsCallbackType]] = None,
    include_default_metrics: Optional[bool] = None,
    additional_fetches: Optional[List[Text]] = None,
    blacklist_feature_fetches: Optional[List[Text]] = None,
    tags: Optional[List[Text]] = None,
    model_type: Optional[Text] = constants.TF_ESTIMATOR):
  """Returns function for constructing shared models."""
  if tags is None:
    tags = [eval_constants.EVAL_TAG]

  def construct_fn(model_load_seconds_callback: Callable[[int], None]):
    """Thin wrapper for the actual construct to allow for load time metrics."""

    def construct():  # pylint: disable=invalid-name
      """Function for constructing shared models."""
      start_time = datetime.datetime.now()
      # If we are evaluating on TPU, initialize the TPU.
      # TODO(b/143484017): Add model warmup for TPU.
      if tf.saved_model.TPU in tags:
        tf.tpu.experimental.initialize_tpu_system()
      if (model_type == constants.TF_ESTIMATOR and
          eval_constants.EVAL_TAG in tags):
        model = load.EvalSavedModel(
            eval_saved_model_path,
            include_default_metrics,
            additional_fetches=additional_fetches,
            blacklist_feature_fetches=blacklist_feature_fetches,
            tags=tags)
        if add_metrics_callbacks:
          model.register_add_metric_callbacks(add_metrics_callbacks)
        model.graph_finalize()
      elif model_type == constants.TF_KERAS:
        # TODO(b/141524386, b/141566408): TPU Inference is not supported
        # for Keras saved_model yet.
        model = tf.keras.models.load_model(eval_saved_model_path)
      elif model_type == constants.TF_LITE:
        # The tf.lite.Interpreter is not thread-safe so we only load the model
        # file's contents and leave construction of the Interpreter up to the
        # PTransform using it.
        model_filename = os.path.join(eval_saved_model_path, _TFLITE_FILE_NAME)
        with tf.io.gfile.GFile(model_filename, 'rb') as model_file:
          model = ModelContents(model_file.read())
      else:
        model = tf.compat.v1.saved_model.load_v2(
            eval_saved_model_path, tags=tags)
      end_time = datetime.datetime.now()
      model_load_seconds_callback(int((end_time - start_time).total_seconds()))
      return model

    return construct

  return construct_fn


class DoFnWithModels(beam.DoFn):
  """Abstract class for DoFns that need the shared models."""

  def __init__(self, model_loaders: Dict[Text, types.ModelLoader]):
    """Initializes DoFn using dict of model loaders keyed by model location."""
    self._model_loaders = model_loaders
    self._loaded_models = None
    self._model_load_seconds = None
    self._model_load_seconds_distribution = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'model_load_seconds')

  def _set_model_load_seconds(self, model_load_seconds):
    self._model_load_seconds = model_load_seconds

  def setup(self):
    self._loaded_models = {}
    for model_name, model_loader in self._model_loaders.items():
      self._loaded_models[model_name] = model_loader.shared_handle.acquire(
          model_loader.construct_fn(self._set_model_load_seconds))

  def process(self, elem):
    raise NotImplementedError('Subclasses are expected to override this.')

  def finish_bundle(self):
    # Must update distribution in finish_bundle instead of setup
    # because Beam metrics are not supported in setup.
    if self._model_load_seconds is not None:
      self._model_load_seconds_distribution.update(self._model_load_seconds)
      self._model_load_seconds = None


# TODO(pachristopher): Remove this class once non-batched predict extractor v2
# is deleted and override the process method directly in predict extractor v1.
@beam.typehints.with_input_types(beam.typehints.List[types.Extracts])
@beam.typehints.with_output_types(types.Extracts)
class BatchReducibleDoFnWithModels(DoFnWithModels):
  """Abstract class for DoFns that need the shared models.

  This DoFn will try to use large batch size at first. If a functional failure
  is caught, an attempt will be made to process the elements serially
  at batch size 1.
  """

  def __init__(self, model_loaders: Dict[Text, types.ModelLoader]):
    super(BatchReducibleDoFnWithModels, self).__init__(model_loaders)
    self._batch_size = (
        beam.metrics.Metrics.distribution(constants.METRICS_NAMESPACE,
                                          'batch_size'))
    self._batch_size_failed = (
        beam.metrics.Metrics.distribution(constants.METRICS_NAMESPACE,
                                          'batch_size_failed'))
    self._num_instances = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_instances')

  def _batch_reducible_process(
      self, elements: List[types.Extracts]) -> Sequence[types.Extracts]:
    raise NotImplementedError('Subclasses are expected to override this.')

  def process(self, elements: List[types.Extracts]) -> Sequence[types.Extracts]:
    batch_size = len(elements)
    try:
      result = self._batch_reducible_process(elements)
      self._batch_size.update(batch_size)
      self._num_instances.inc(batch_size)
      return result
    except (ValueError, tf.errors.InvalidArgumentError) as e:
      tf.compat.v1.logging.warning(
          'Large batch_size %s failed with error %s. '
          'Attempting to run batch through serially.', batch_size, e)
      self._batch_size_failed.update(batch_size)
      result = []
      for element in elements:
        self._batch_size.update(1)
        result.extend(self._batch_reducible_process([element]))
      self._num_instances.inc(len(result))
      return result


@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
class BatchReducibleBatchedDoFnWithModels(DoFnWithModels):
  """Abstract class for DoFns that need the shared models.

  This DoFn operates on batched Arrow RecordBatch as input. This DoFn will try
  to use a large batch size at first. If a functional failure is caught, an
  attempt will be made to process the elements serially at batch size 1.
  """

  def __init__(self, model_loaders: Dict[Text, types.ModelLoader]):
    super(BatchReducibleBatchedDoFnWithModels, self).__init__(model_loaders)
    self._batch_size = (
        beam.metrics.Metrics.distribution(constants.METRICS_NAMESPACE,
                                          'batch_size'))
    self._batch_size_failed = (
        beam.metrics.Metrics.distribution(constants.METRICS_NAMESPACE,
                                          'batch_size_failed'))
    self._num_instances = beam.metrics.Metrics.counter(
        constants.METRICS_NAMESPACE, 'num_instances')

  def _batch_reducible_process(
      self, batched_extract: types.Extracts) -> Sequence[types.Extracts]:
    raise NotImplementedError('Subclasses are expected to override this.')

  def process(self, element: types.Extracts) -> Sequence[types.Extracts]:
    batch_size = element[constants.ARROW_RECORD_BATCH_KEY].num_rows
    try:
      result = self._batch_reducible_process(element)
      self._batch_size.update(batch_size)
      self._num_instances.inc(batch_size)
      return result
    except (ValueError, tf.errors.InvalidArgumentError) as e:
      logging.warning(
          'Large batch_size %s failed with error %s. '
          'Attempting to run batch through serially. Note that this will '
          'significantly affect the performance.', batch_size, e)
      self._batch_size_failed.update(batch_size)
      result = []
      record_batch = element[constants.ARROW_RECORD_BATCH_KEY]
      for i in range(batch_size):
        self._batch_size.update(1)
        unbatched_element = {}
        for key in element.keys():
          if key == constants.ARROW_RECORD_BATCH_KEY:
            unbatched_element[key] = record_batch.slice(i, 1)
          else:
            unbatched_element[key] = [element[key][i]]
        result.extend(self._batch_reducible_process(unbatched_element))
      self._num_instances.inc(len(result))
      return result


class CombineFnWithModels(beam.CombineFn):
  """Abstract class for CombineFns that need the shared models.

  Until BEAM-3736 (Add SetUp() and TearDown() for CombineFns) is implemented
  users of this class are responsible for calling _setup_if_needed manually.
  """

  def __init__(self, model_loaders: Dict[Text, types.ModelLoader]):
    """Initializes CombineFn using dict of loaders keyed by model location."""
    self._model_loaders = model_loaders
    self._loaded_models = None
    self._model_load_seconds = None
    self._model_load_seconds_distribution = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'model_load_seconds')

  def _set_model_load_seconds(self, model_load_seconds):
    self._model_load_seconds = model_load_seconds

  # TODO(yifanmai): Merge _setup_if_needed into setup
  # There's no initialisation method for CombineFns.
  # See BEAM-3736: Add SetUp() and TearDown() for CombineFns.
  def _setup_if_needed(self) -> None:
    if self._loaded_models is None:
      self._loaded_models = {}
      for model_name, model_loader in self._model_loaders.items():
        self._loaded_models[model_name] = model_loader.shared_handle.acquire(
            model_loader.construct_fn(self._set_model_load_seconds))
      if self._model_load_seconds is not None:
        self._model_load_seconds_distribution.update(self._model_load_seconds)
        self._model_load_seconds = None
