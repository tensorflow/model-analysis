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

import collections
import copy
import importlib
import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from absl import logging
import apache_beam as beam
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis import types
from tensorflow_model_analysis.eval_saved_model import constants as eval_constants
from tensorflow_model_analysis.eval_saved_model import load
from tensorflow_model_analysis.experimental import preprocessing_functions
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import util
from tfx_bsl.tfxio import tensor_adapter

from tensorflow_metadata.proto.v0 import schema_pb2

# TODO(b/162075791): Need to load tensorflow_ranking, tensorflow_text,
# tensorflow_decision_forests, and struct2tensor for models that use those ops.
# pylint: disable=g-import-not-at-top
# LINT.IfChange
try:
  # Needed to load SavedModel on s3://
  importlib.import_module('tensorflow_io')
  logging.info('imported tensorflow_io')
except Exception as e:  # pylint: disable=broad-except
  logging.info('tensorflow_io is not available: %s', e)
try:
  importlib.import_module('tensorflow_ranking')
  logging.info('imported tensorflow_ranking')
# tensorflow_ranking may not be available, or it may fail to be imported
# (because it does not support TF 1.x).
except Exception as e:  # pylint: disable=broad-except
  logging.info('tensorflow_ranking is not available: %s', e)
try:
  importlib.import_module('tensorflow_text')
  logging.info('imported tensorflow_text')
except (ImportError, tf.errors.NotFoundError) as e:
  logging.info('tensorflow_text is not available: %s', e)
try:
  importlib.import_module('tensorflow_decision_forests')
  logging.info('imported tensorflow_decision_forests')
except Exception as e:  # pylint: disable=broad-except
  logging.info('tensorflow_decision_forests is not available: %s', e)
try:
  importlib.import_module('struct2tensor')
  logging.info('imported struct2tensor')
except Exception as e:  # pylint: disable=broad-except
  logging.info('struct2tensor is not available: %s', e)
# LINT.ThenChange(tensorflow_transform/saved/saved_transform_io.py)
# pylint: enable=g-import-not-at-top

_TF_MAJOR_VERSION = int(tf.version.VERSION.split('.')[0])

KERAS_INPUT_SUFFIX = '_input'

_TFLITE_FILE_NAME = 'tflite'

_PREDICT_SIGNATURE_DEF_KEY = 'predict'


class ModelContents:
  """Class for storing model contents.

  This class exists because weak references to bytes are not allowed.
  """
  __slots__ = ['contents', '__weakref__']

  def __init__(self, contents: bytes):
    self.contents = contents


def get_preprocessing_signature(signature_name: str) -> Tuple[str, List[str]]:
  """Returns the preprocessing function name and its feature name."""
  signature_name, *input_names = signature_name.split('@')
  if len(input_names) > 1:
    raise NotImplementedError(
        'Transforming on multiple features is not '
        f'supported. signature {signature_name} has input names: '
        f'{input_names}.')
  return signature_name, input_names


def get_baseline_model_spec(
    eval_config: config_pb2.EvalConfig) -> Optional[config_pb2.ModelSpec]:
  """Returns baseline model spec."""
  for spec in eval_config.model_specs:
    if spec.is_baseline:
      return spec
  return None


def get_non_baseline_model_specs(
    eval_config: config_pb2.EvalConfig) -> Iterable[config_pb2.ModelSpec]:
  """Returns non-baseline model specs."""
  return [spec for spec in eval_config.model_specs if not spec.is_baseline]


def get_model_spec(eval_config: config_pb2.EvalConfig,
                   model_name: str) -> Optional[config_pb2.ModelSpec]:
  """Returns model spec with given model name."""
  if len(eval_config.model_specs) == 1 and not model_name:
    return eval_config.model_specs[0]
  for spec in eval_config.model_specs:
    if spec.name == model_name:
      return spec
  return None


def get_label_key(model_spec: config_pb2.ModelSpec,
                  output_name: str) -> Optional[str]:
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


def get_model_type(model_spec: Optional[config_pb2.ModelSpec],
                   model_path: Optional[str] = '',
                   tags: Optional[List[str]] = None) -> str:
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

  signature_name = None
  if model_spec:
    if model_spec.signature_name:
      signature_name = model_spec.signature_name
    else:
      signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY

  # Default to serving unless estimator is used and eval signature is used.
  if signature_name == eval_constants.EVAL_TAG:
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


def get_feature_values_for_model_spec_field(
    model_specs: List[config_pb2.ModelSpec],
    field: str,
    multi_output_field: Optional[str],
    batched_extracts: types.Extracts,
    allow_missing: bool = False) -> Optional[Any]:
  """Gets feature values associated with given model spec fields from extracts.

  Args:
    model_specs: List of model specs from EvalConfig.
    field: Name of field used to determine the feature(s) to extract. This
      should be an attribute on the ModelSpec such as "label_key",
      "example_weight_key", or "prediction_key".
    multi_output_field: Optional name of field used to store multi-output
      versions of the features. This should be an attribute on the ModelSpec
      such as "label_keys", "example_weight_keys", or "prediction_keys". This
      field is only used if a value at field is not found.
    batched_extracts: Extracts containing batched features keyed by
      tfma.FEATURES_KEY and optionally tfma.TRANSFORMED_FEATURES_KEY.
    allow_missing: True if the feature may be missing (in which case None will
      be used as the value).

  Returns:
    Feature values stored at given key (or feature values stored at each output
    keyed by output name if field containing map of feature keys was used). If
    multiple models are used the value(s) will be stored in a dict keyed by
    model name. If no values are found and allow_missing is False then None
    will be returned.
  """
  values = {}
  if (constants.FEATURES_KEY in batched_extracts and
      batched_extracts[constants.FEATURES_KEY]):
    features = batched_extracts[constants.FEATURES_KEY]
  else:
    features = {}
  for spec in model_specs:
    # Get transformed features (if any) for this model.
    if (constants.TRANSFORMED_FEATURES_KEY in batched_extracts and
        batched_extracts[constants.TRANSFORMED_FEATURES_KEY]):
      transformed_features = batched_extracts[
          constants.TRANSFORMED_FEATURES_KEY]
      if len(model_specs) > 1 and transformed_features:
        if spec.name in transformed_features:
          transformed_features = transformed_features[spec.name]
      transformed_features = transformed_features or {}
    else:
      transformed_features = {}
    # Lookup first in transformed_features and then in features.
    if hasattr(spec, field) and getattr(spec, field):
      key = getattr(spec, field)
      if key in transformed_features:
        values[spec.name] = transformed_features[key]
      elif key in features:
        values[spec.name] = features[key]
      elif allow_missing:
        values[spec.name] = None
    elif (multi_output_field and hasattr(spec, multi_output_field) and
          getattr(spec, multi_output_field)):
      output_values = {}
      for output_name, key in getattr(spec, multi_output_field).items():
        if key in transformed_features:
          output_values[output_name] = transformed_features[key]
        elif key in features:
          output_values[output_name] = features[key]
        elif allow_missing:
          output_values[output_name] = None
      if output_values:
        values[spec.name] = output_values
    elif allow_missing:
      values[spec.name] = None
  if values:
    # If only one model, the output is stored without using a dict
    if len(model_specs) == 1:
      values = next(iter(values.values()))
  else:
    values = None
  return values


def get_default_signature_name(model: Any) -> str:
  """Returns default signature name for given model."""
  # First try 'predict' then try 'serving_default'. The estimator output
  # for the 'serving_default' key does not include all the heads in a
  # multi-head model. However, keras only uses the 'serving_default' for
  # its outputs. Note that the 'predict' key only exists for estimators
  # for multi-head models, for single-head models only 'serving_default'
  # is used.
  if (hasattr(model, 'signatures') and
      _PREDICT_SIGNATURE_DEF_KEY in model.signatures):
    return _PREDICT_SIGNATURE_DEF_KEY
  return tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY


# TODO(b/175357313): Remove _get_save_spec check when the save_spec changes
# have been released.
def _get_model_input_spec(model: Any) -> Optional[Any]:
  """Returns the model input `TensorSpec`s."""
  if hasattr(model, 'save_spec'):
    if model.save_spec() is None:
      return None
    # The inputs TensorSpec is the first element of the (args, kwargs) tuple.
    return model.save_spec()[0][0]
  elif hasattr(model, '_get_save_spec'):
    # In versions of TF released before `save_spec`, `_get_save_spec` returns
    # the input save spec.
    return model._get_save_spec()  # pylint: disable=protected-access
  return None


def is_callable_fn(fn: Any) -> bool:
  """Returns true if function is callable."""
  if _TF_MAJOR_VERSION >= 2:
    if isinstance(_get_model_input_spec(fn), dict):
      return True
    if (hasattr(fn, 'input_names') and fn.input_names and
        hasattr(fn, 'inputs') and fn.inputs):
      return True
  return False


def get_callable(model: Any,
                 signature_name: Optional[str] = None,
                 required: bool = True) -> Optional[Callable[..., Any]]:
  """Returns callable associated with given signature or None if not callable.

  The available callables are defined by the model.signatures attribute which
  are defined at the time the model is saved. For keras based models, the
  model itself can also be used as can a callable attribute on the model named
  after the signature_name.

  Args:
    model: A model that is callable or contains a `signatures` attribute. If
      neither of these conditions are met, then None will be returned.
    signature_name: Optional name of signature to use. If not provided then
      either the default serving signature will be used (if model is not
      callable) or the model itself will be used (if the model is callable). If
      provided then model.signatures will be used regardless of whether the
      model is callable or not.
    required: True if signature_name is required to exist if provided.

  Returns:
    Callable associated with given signature (or the model itself) or None if
    no callable could be found.

  Raises:
    ValueError: If signature_name not found in model.signatures.
  """
  if not hasattr(model, 'signatures') and not is_callable_fn(model):
    return None

  if not signature_name:
    if is_callable_fn(model):
      return model
    signature_name = get_default_signature_name(model)

  if signature_name not in model.signatures:
    if hasattr(model, signature_name):
      fn = getattr(model, signature_name)
      if is_callable_fn(fn):
        return fn
    if required:
      raise ValueError('{} not found in model signatures: {}'.format(
          signature_name, model.signatures))
    return None

  return model.signatures[signature_name]


def get_input_specs(model: Any,
                    signature_name: Optional[str] = None,
                    required: bool = True) -> Optional[Dict[str, tf.TypeSpec]]:
  """Returns the input names and tensor specs associated with callable or None.

  Args:
    model: A model that is callable or contains a `signatures` attribute. If
      neither of these conditions are met, then None will be returned.
    signature_name: Optional name of signature to use. If not provided then
      either the default serving signature will be used (if model is not
      callable) or the model itself will be used (if the model is callable). If
      provided then model.signatures will be used regardless of whether the
      model is callable or not.
    required: True if signature_name is required to exist if provided.

  Returns:
    Dict mapping input names to their associated tensor specs or None if no
    callable could be found.

  Raises:
    ValueError: If signature_name not found in model.signatures.
  """
  if not hasattr(model, 'signatures') and not is_callable_fn(model):
    return None

  def get_callable_input_specs(fn):
    if isinstance(_get_model_input_spec(fn), dict):
      return _get_model_input_spec(fn)
    else:
      input_specs = {}
      for input_name, input_tensor in zip(fn.input_names, fn.inputs):
        if hasattr(input_tensor, 'type_spec'):
          # "KerasTensor" types have type_spec attributes.
          type_spec = input_tensor.type_spec
        else:
          type_spec = tf.type_spec_from_value(input_tensor)
        input_specs[input_name] = type_spec
      return input_specs

  if not signature_name:
    # Special support for keras-based models.
    if is_callable_fn(model):
      return get_callable_input_specs(model)
    signature_name = get_default_signature_name(model)

  if signature_name in model.signatures:
    signature = model.signatures[signature_name]
    # First arg of structured_input_signature tuple is shape, second is spec
    # (we currently only support named params passed as a dict)
    if (signature.structured_input_signature and
        len(signature.structured_input_signature) == 2 and
        isinstance(signature.structured_input_signature[1], dict)):
      return signature.structured_input_signature[1]
    else:
      return None
  elif hasattr(model, signature_name):
    fn = getattr(model, signature_name)
    if is_callable_fn(fn):
      return get_callable_input_specs(fn)

  if required:
    raise ValueError('{} not found in model signatures: {}'.format(
        signature_name, model.signatures))

  return None


def input_specs_to_tensor_representations(
    input_specs: Dict[str,
                      tf.TypeSpec]) -> tensor_adapter.TensorRepresentations:
  """Converts input specs into tensor representations."""
  tensor_representations = {}
  for name, type_spec in input_specs.items():
    tensor_representation = schema_pb2.TensorRepresentation()
    if isinstance(type_spec, tf.SparseTensorSpec):
      tensor_representation.varlen_sparse_tensor.column_name = name
    elif isinstance(type_spec, tf.RaggedTensorSpec):
      tensor_representation.ragged_tensor.feature_path.step.append(name)
    else:
      tensor_representation.dense_tensor.column_name = name
      for dim in type_spec.shape[1:] if len(type_spec.shape) > 1 else []:
        if dim is None:
          raise ValueError(
              'input {} contains unknown dimensions which are not supported: '
              'type_spec={}, input_specs={}'.format(name, type_spec,
                                                    input_specs))
        tensor_representation.dense_tensor.shape.dim.append(
            schema_pb2.FixedShape.Dim(size=dim))
    tensor_representations[name] = tensor_representation
  return tensor_representations


def find_input_name_in_features(features: Set[str],
                                input_name: str) -> Optional[str]:
  """Maps input name to an entry in features. Returns None if not found."""
  if input_name in features:
    return input_name
  # Some keras models prepend '_input' to the names of the inputs
  # so try under '<name>_input' as well.
  elif (input_name.endswith(KERAS_INPUT_SUFFIX) and
        input_name[:-len(KERAS_INPUT_SUFFIX)] in features):
    return input_name[:-len(KERAS_INPUT_SUFFIX)]
  return None


def filter_by_input_names(
    d: Mapping[str, types.TensorType],
    input_names: List[str]) -> Optional[Mapping[str, types.TensorType]]:
  """Filters dict by input names.

  In case we don't find the specified input name in the dict, we assume we are
  feeding serialized examples to the model and return None.

  Args:
    d: Dict to filter.
    input_names: List of input names.

  Returns:
    Dict with keys matching input_names or None if not all keys could be found.
  """
  if not input_names:
    return None
  result = {}
  dict_keys = set(d.keys())
  for name in input_names:
    input_name = find_input_name_in_features(dict_keys, name)
    if input_name is None:
      return None
    result[name] = d[input_name]
  return result


def get_inputs(
    features: types.DictOfTensorValue,
    input_specs: types.DictOfTypeSpec,
) -> Optional[types.TensorTypeMaybeMultiLevelDict]:
  """Returns inputs from features for given input specs.

  Args:
    features: Dict of feature tensors.
    input_specs: Input specs keyed by input name.

  Returns:
    Input tensors keyed by input name.
  """
  inputs = None
  input_names = list(input_specs.keys())
  # Avoid getting the tensors if we appear to be feeding serialized examples to
  # the callable.
  single_input = (
      next(iter(input_specs.values())) if len(input_specs) == 1 else None)
  single_input_name = input_names[0] if single_input else None
  if not (single_input and
          single_input.dtype == tf.string and find_input_name_in_features(
              set(features.keys()), single_input_name) is None):
    # If filtering is not successful (i.e. None is returned) fallback to feeding
    # serialized examples.
    features = filter_by_input_names(features, input_names)
    if features:
      inputs = util.to_tensorflow_tensors(features, input_specs)
  return inputs


def model_construct_fn(  # pylint: disable=invalid-name
    eval_saved_model_path: Optional[str] = None,
    add_metrics_callbacks: Optional[List[types.AddMetricsCallbackType]] = None,
    include_default_metrics: Optional[bool] = None,
    additional_fetches: Optional[List[str]] = None,
    blacklist_feature_fetches: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    model_type: Optional[str] = constants.TF_ESTIMATOR) -> Callable[[], Any]:
  """Returns function for constructing shared models."""
  if tags is None:
    tags = [eval_constants.EVAL_TAG]

  def construct_fn():  # pylint: disable=invalid-name
    """Function for constructing shared models."""
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
      model = tf.keras.models.load_model(eval_saved_model_path)
    elif model_type == constants.TF_LITE:
      # The tf.lite.Interpreter is not thread-safe so we only load the model
      # file's contents and leave construction of the Interpreter up to the
      # PTransform using it.
      model_filename = os.path.join(eval_saved_model_path, _TFLITE_FILE_NAME)
      with tf.io.gfile.GFile(model_filename, 'rb') as model_file:
        model_bytes = model_file.read()

      # If a SavedModel is present in the same directory, load it as well.
      # This allows the SavedModel to be used for computing the
      # Transformed Features and Labels.
      if (tf.io.gfile.exists(
          os.path.join(eval_saved_model_path,
                       tf.saved_model.SAVED_MODEL_FILENAME_PB)) or
          tf.io.gfile.exists(
              os.path.join(eval_saved_model_path,
                           tf.saved_model.SAVED_MODEL_FILENAME_PBTXT))):
        model = tf.compat.v1.saved_model.load_v2(
            eval_saved_model_path, tags=tags)
        model.contents = model_bytes
      else:
        model = ModelContents(model_bytes)

    elif model_type == constants.TF_JS:
      # We invoke TFJS models via a subprocess call. So this call is no-op.
      return None
    else:
      model = tf.compat.v1.saved_model.load_v2(eval_saved_model_path, tags=tags)
    return model

  return construct_fn


class DoFnWithModels(beam.DoFn):
  """Abstract class for DoFns that need the shared models."""

  def __init__(self, model_loaders: Dict[str, types.ModelLoader]):
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
      self._loaded_models[model_name] = model_loader.load(
          model_load_time_callback=self._set_model_load_seconds)

  def process(self, elem):
    raise NotImplementedError('Subclasses are expected to override this.')

  def finish_bundle(self):
    # Must update distribution in finish_bundle instead of setup
    # because Beam metrics are not supported in setup.
    if self._model_load_seconds is not None:
      self._model_load_seconds_distribution.update(self._model_load_seconds)
      self._model_load_seconds = None


# TODO(b/178158073): Remove this class once non-batched predict extractor v2
# is deleted and override the process method directly in predict extractor v1.
@beam.typehints.with_input_types(beam.typehints.List[types.Extracts])
@beam.typehints.with_output_types(types.Extracts)
class BatchReducibleDoFnWithModels(DoFnWithModels):
  """Abstract class for DoFns that need the shared models.

  This DoFn will try to use large batch size at first. If a functional failure
  is caught, an attempt will be made to process the elements serially
  at batch size 1.
  """

  def __init__(self, model_loaders: Dict[str, types.ModelLoader]):
    super().__init__(model_loaders)
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
    except (ValueError, tf.errors.InvalidArgumentError,
            tf.errors.ResourceExhaustedError) as e:
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

  This DoFn operates on batched features as input. This DoFn will try to use a
  large batch size at first. If a functional failure is caught, an attempt will
  be made to process the elements serially at batch size 1.
  """

  def __init__(self, model_loaders: Dict[str, types.ModelLoader]):
    super().__init__(model_loaders)
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
    batch_size = util.batch_size(element)
    try:
      result = self._batch_reducible_process(element)
      self._batch_size.update(batch_size)
      self._num_instances.inc(batch_size)
      return result
    except (ValueError, tf.errors.InvalidArgumentError,
            tf.errors.ResourceExhaustedError, RuntimeError) as e:
      logging.warning(
          'Large batch_size %s failed with error %s. '
          'Attempting to run batch through serially. Note that this will '
          'significantly affect the performance.', batch_size, e)
      self._batch_size_failed.update(batch_size)
      result = []
      for unbatched_element in util.split_extracts(
          element, keep_batch_dim=True):
        self._batch_size.update(1)
        result.extend(self._batch_reducible_process(unbatched_element))
      self._num_instances.inc(len(result))
      return result


@beam.typehints.with_input_types(types.Extracts)
@beam.typehints.with_output_types(types.Extracts)
class ModelSignaturesDoFn(BatchReducibleBatchedDoFnWithModels):
  """Updates extracts by calling specified model signature functions."""

  def __init__(self,
               eval_config: config_pb2.EvalConfig,
               eval_shared_models: Dict[str, types.EvalSharedModel],
               signature_names: Dict[str, Dict[str, List[str]]],
               default_signature_names: Optional[List[str]] = None,
               prefer_dict_outputs: bool = True):
    """Initializes DoFn.

    Examples of combinations of signature_names and default_signatures that
    might be used:

    1) Update 'predictions' using default callable on a single model.

      signature_names: {'predictions': {'': [None]}}

    2) Update 'predictions' using custom callables

      signature_names: {'predictions': {'model1': ['fn1'], 'model2': ['fn2']}}

    3) Update 'features' using 'tft_layer' callable

      signature_names: {'features': {'': ['tft_layer']}}

    4) Updates 'features' using a specific setting for one model, but using
       defaults signatures for another

      signature_names: {'features': {'model1': ['tft_layer'], 'model2': []}}
      default_signature_names: ['transformed_features', 'transformed_labels']

    Args:
      eval_config: Eval config.
      eval_shared_models: Shared model parameters keyed by model name.
      signature_names: Names of signature functions to call keyed by the
        associated extracts that should be updated and the name of the model
        they are associated with. The signature functions may be stored either
        in a dict under a `signatures` attribute or directly as separate named
        attributes of the model. If a signature name list is empty then the
        default_signatures will be used. If a list entry is empty (None or ''),
        then the model itself (or a common default signature for the model -
        e.g. 'serving_default') will be used.
      default_signature_names: One or more signature names to use by default
        when an empty list is used in signature_names. All defaults will be
        tried, but unlike signature_names it is not an error if a signature is
        not found.
      prefer_dict_outputs: True to convert results from calling a signature
        function are are not dicts into dicts by using the signature_name as the
        key. If False, dict outputs that have only one entry will be converted
        into single output values. For example, it is preferable to store
        predictions as single output values (unless a multi-output model is
        used) whereas it is preferrable to always store features as a dict where
        the output keys represent the feature names.
    """
    super().__init__({k: v.model_loader for k, v in eval_shared_models.items()})
    self._eval_config = eval_config
    self._signature_names = signature_names
    self._default_signature_names = default_signature_names
    self._prefer_dict_outputs = prefer_dict_outputs

  def setup(self):
    super().setup()
    # Verify and filter models to only those used in ModelSpecs.
    loaded_models = {}
    for spec in self._eval_config.model_specs:
      # To maintain consistency between settings where single models are used,
      # always use '' as the model name regardless of whether a name is passed.
      model_name = spec.name if len(self._eval_config.model_specs) > 1 else ''
      if model_name not in self._loaded_models:
        raise ValueError(
            'loaded model for "{}" not found: eval_config={}'.format(
                spec.name, self._eval_config))
      loaded_models[model_name] = self._loaded_models[model_name]
    self._loaded_models = loaded_models

  def _batch_reducible_process(
      self, batched_extract: types.Extracts) -> List[types.Extracts]:

    def maybe_expand_dims(arr):
      if not hasattr(arr, 'shape') or not arr.shape:
        return np.expand_dims(arr, axis=0)
      else:
        return arr

    def to_dense(t):
      if isinstance(t, tf.SparseTensor):
        return tf.sparse.to_dense(t)
      elif isinstance(t, tf.RaggedTensor):
        return t.to_tensor()
      else:
        return t

    def check_shape(t, batch_size, key=None):
      if t.shape[0] != batch_size:
        raise ValueError(
            'First dimension does not correspond with batch size. '
            f'Batch size: {batch_size}, Dimensions: {t.shape}, Key: {key}.')

    result = copy.copy(batched_extract)
    batch_size = util.batch_size(batched_extract)
    features = util.get_features_from_extracts(batched_extract)
    serialized_examples = batched_extract[constants.INPUT_KEY]
    if isinstance(serialized_examples, np.ndarray):
      # Most models only accept serialized examples as a 1-d tensor
      serialized_examples = serialized_examples.flatten()
    for extracts_key in self._signature_names.keys():
      if extracts_key not in result:
        result[extracts_key] = None
    for model_name, model in self._loaded_models.items():
      for extracts_key, signature_names in self._signature_names.items():
        for signature_name in (signature_names[model_name] or
                               self._default_signature_names):
          signature = None
          input_specs = None
          inputs = None
          positional_inputs = False
          required = bool(signature_names[model_name])
          if signature_name and '@' in signature_name:
            try:
              signature_name, input_names = get_preprocessing_signature(
                  signature_name)
              signature = getattr(preprocessing_functions, signature_name)
              input_specs = {
                  input_name: type_spec for input_name, type_spec in zip(
                      input_names, signature.input_signature)
              }
              inputs = get_inputs(features, input_specs)
              positional_inputs = True
            except AttributeError as e:
              logging.warning(
                  """Failed to get signature of %s or as TFMA
                  preprocessing function. Trying in-graph preprocessing
                  function.""", signature_name)

          if not input_specs:
            input_specs = get_input_specs(model, signature_name, required) or {}
            # If input_specs exist then try to filter the inputs by the input
            # names (unlike estimators, keras does not accept unknown inputs).
            if input_specs:
              inputs = get_inputs(features, input_specs)
          if not inputs:
            # Assume serialized examples
            assert serialized_examples is not None, 'Raw examples not found.'
            inputs = serialized_examples
            # If a signature name was not provided, default to using the serving
            # signature since parsing normally will be done outside model.
            if not signature_name:
              signature_name = get_default_signature_name(model)

          signature = signature or get_callable(model, signature_name, required)
          if signature is None:
            if not required:
              continue
            raise ValueError('Unable to find %s function needed to update %s' %
                             (signature_name, extracts_key))
          try:
            if isinstance(inputs, dict):
              if hasattr(signature, 'structured_input_signature'):
                outputs = signature(**inputs)
              elif positional_inputs:
                outputs = signature(*inputs.values())
              else:
                outputs = signature(inputs)
            else:
              outputs = signature(tf.constant(inputs, dtype=tf.string))
          except (TypeError, tf.errors.InvalidArgumentError) as e:
            raise ValueError(
                """Fail to call signature func with signature_name: {}.
                the inputs are:\n {}.
                The input_specs are:\n {}.""".format(signature_name, inputs,
                                                     input_specs)) from e

          dense_outputs = {}
          if isinstance(outputs, dict):
            for k, v in outputs.items():
              dense_outputs[k] = to_dense(v)
              check_shape(dense_outputs[k], batch_size, key=k)
          else:
            dense_outputs = to_dense(outputs)
            check_shape(dense_outputs, batch_size)

          if isinstance(dense_outputs, dict):
            output = {
                k: maybe_expand_dims(v.numpy())
                for k, v in dense_outputs.items()
            }
          else:
            output = {
                signature_name: maybe_expand_dims(np.asarray(dense_outputs))
            }
          if result[extracts_key] is None:
            result[extracts_key] = collections.defaultdict(dict)
          result[extracts_key][model_name].update(output)
    if isinstance(result[extracts_key], dict):
      for model_name, output in result[extracts_key].items():
        if not self._prefer_dict_outputs and len(output) == 1:
          result[extracts_key][model_name] = list(output.values())[0]
      # If only one model, the output is stored without using a dict
      if len(self._eval_config.model_specs) == 1:
        result[extracts_key] = list(result[extracts_key].values())[0]
    return [result]


class CombineFnWithModels(beam.CombineFn):
  """Abstract class for CombineFns that need the shared models."""

  def __init__(self, model_loaders: Dict[str, types.ModelLoader]):
    """Initializes CombineFn using dict of loaders keyed by model location."""
    self._model_loaders = model_loaders
    self._loaded_models = None
    self._model_load_seconds = None
    self._model_load_seconds_distribution = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'model_load_seconds')

  def _set_model_load_seconds(self, model_load_seconds):
    self._model_load_seconds = model_load_seconds

  def setup(self):
    if self._loaded_models is None:
      self._loaded_models = {}
      for model_name, model_loader in self._model_loaders.items():
        self._loaded_models[model_name] = model_loader.load(
            model_load_time_callback=self._set_model_load_seconds)
      if self._model_load_seconds is not None:
        self._model_load_seconds_distribution.update(self._model_load_seconds)
        self._model_load_seconds = None


# Need to run after verify_and_update_eval_shared_models.
def has_rubber_stamp(eval_shared_model: Optional[List[types.EvalSharedModel]]):
  """Check whether the candidate model is being rubber stamped."""
  # Model agnostic case, no baseline, change thresholds should not be
  # configured.
  if eval_shared_model is None:
    return False
  # In case of multiple candidate modules, all non baseline models need to have
  # rubber stamp.
  if isinstance(eval_shared_model, list):
    if (len(eval_shared_model) == 1 and eval_shared_model[0].is_baseline):
      raise ValueError('Only a baseline model is provided. '
                       'A candidate model is required for evaluation.')
    return all(m.rubber_stamp if not m.is_baseline else True
               for m in eval_shared_model)
  raise ValueError('Not supported eval_shared_model type: {}'.format(
      type(eval_shared_model)))
