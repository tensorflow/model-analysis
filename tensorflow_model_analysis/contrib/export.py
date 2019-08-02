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
"""Experimental export utilities for exporting feature column annotations."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import contextlib
import inspect
import os
import pickle

# Standard Imports
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis import types
from tensorflow_model_analysis import util as tfma_util

from typing import Any, Callable, Dict, Optional, Text

# pylint: disable=g-import-not-at-top,g-statement-before-imports
try:
  # Remove this after dense_features becomes available in TF.
  from tensorflow.python.feature_column import dense_features  # pylint: disable=g-direct-tensorflow-import
except ImportError:
  pass
from tensorflow.python.feature_column import feature_column_v2  # pylint: disable=g-direct-tensorflow-import
from tensorflow_estimator.python.estimator.canned import dnn


def _make_observing_layer_call(old_call_fn: Callable[..., Any],
                               key_prefix: Text, output_dict: Dict[Text, Any]):
  """Returns a a function that wraps <Layer>.__call__ and observes arguments.

  The returned function can be used to replace DenseFeatures.__call__ and
  LinearModelLayer.__call__ so we can observe the features dictionary and the
  cols_to_output_dict dictionary. Note that the wrapper function is
  "observer-only" - it wraps around the old_call_fn and observes its
  arguments.

  Args:
    old_call_fn: The old __call__ function.
    key_prefix: Prefix for the keys in output_dict.
    output_dict: Dictionary to write observed arguments to.

  Returns:
    A function that wraps old_call_fn and observes arguments, that can be used
    to replace DenseFeatures.__call__ and LinearModelLayer.__call__.
  """

  def observing_call(self, *args, **kwargs):
    """Wrapper around old_call_fn that observes arguments."""

    callargs = inspect.getcallargs(old_call_fn, self, *args, **kwargs)
    features = callargs['features']
    cols_to_output_tensors = callargs.get('cols_to_output_tensors')

    local_cols_to_output_tensors = {}
    if cols_to_output_tensors is not None:
      local_cols_to_output_tensors = cols_to_output_tensors
    result = old_call_fn(
        self,
        features=features,
        cols_to_output_tensors=local_cols_to_output_tensors)
    output_dict[key_prefix + '_features'] = dict(features)
    output_dict[key_prefix +
                '_cols_to_output_tensors'] = dict(local_cols_to_output_tensors)
    return result

  return observing_call


def _make_observing_model_call(old_call_fn: Callable[..., Any], key: Text,
                               output_dict: Dict[Text, Any]):
  """Returns a a function that wraps <Model>.__call__ and observes arguments.

  The returned function can be used to replace _DNNModel.__call__,
  _DNNModelV2.__call__, and LinearModel.__call__ so we can observe the Keras
  Model. Note that the wrapper function is "observer-only" - it wraps around the
  old_call_fn and observes its arguments.

  Args:
    old_call_fn: The old __call__ function.
    key: Key to write observed model to in output_dict.
    output_dict: Dictionary to write observed model to.

  Returns:
    A function that wraps old_call_fn and observes arguments, that can be used
    to replace DenseFeatures.__call__ and LinearModelLayer.__call__.
  """

  def observing_call(self, *args, **kwargs):
    output_dict[key] = self
    return old_call_fn(self, *args, **kwargs)

  return observing_call


def _get_dense_features_module():
  """Returns the module that contains DenseFeatures class."""
  try:
    return dense_features
  except:  # pylint: disable=bare-except
    return feature_column_v2


@contextlib.contextmanager
def _observe_dnn_model(output_dict: Dict[Text, Any]):
  """Observe feature and feature metadata from DNN models.

  We do this by monkey-patching the appropriate input feature layers and
  Keras model objects.

  Args:
    output_dict: Dictionary to write observed information to.

  Yields:
    Nothing.
  """

  dense_feature_module = _get_dense_features_module()
  old_dense_features_call = dense_feature_module.DenseFeatures.call
  dense_feature_module.DenseFeatures.call = _make_observing_layer_call(
      old_dense_features_call, 'dnn_model', output_dict)

  old_dnn_model_call = dnn._DNNModel.call  # pylint: disable=protected-access
  dnn._DNNModel.call = _make_observing_model_call(  # pylint: disable=protected-access
      old_dnn_model_call, 'dnn_model', output_dict)

  old_dnn_model_v2_call = dnn._DNNModelV2.call  # pylint: disable=protected-access
  dnn._DNNModelV2.call = _make_observing_model_call(  # pylint: disable=protected-access
      old_dnn_model_v2_call, 'dnn_model', output_dict)

  # This is a contextmanager, meant to be used in a with statement like
  # with _observe_dnn_model(output_dict):
  #   ...
  # The code before the yield will execute when the with statement is executed,
  # and the code after the yield will execute on exiting the with block.
  yield

  dnn._DNNModel.call = old_dnn_model_call  # pylint: disable=protected-access
  dnn._DNNModelV2.call = old_dnn_model_v2_call  # pylint: disable=protected-access
  dense_feature_module.DenseFeatures.call = old_dense_features_call


def _serialize_feature_column(feature_column: feature_column_v2.FeatureColumn
                             ) -> Dict[Text, Any]:
  """Serialize the given feature column into a dictionary."""
  if not hasattr(feature_column, '_is_v2_column'):
    raise ValueError('feature_column does not has _is_v2_column attribute')
  if not feature_column._is_v2_column:  # pylint: disable=protected-access
    raise ValueError('feature_column is not a v2 column')
  if isinstance(feature_column, feature_column_v2.NumericColumn):
    return {
        'type': 'NumericColumn',
        'key': feature_column.key,
    }
  elif isinstance(feature_column, feature_column_v2.BucketizedColumn):
    return {
        'type': 'BucketizedColumn',
        'key': feature_column.name,
        'boundaries': feature_column.boundaries,
    }
  elif isinstance(feature_column, feature_column_v2.CrossedColumn):
    return {
        'type': 'CrossedColumn',
        'key': feature_column.name,
        'keys': feature_column.keys,
    }
  elif isinstance(feature_column, feature_column_v2.HashedCategoricalColumn):
    return {
        'type': 'HashCategoricalColumn',
        'key': feature_column.key,
    }
  elif isinstance(feature_column, feature_column_v2.IdentityCategoricalColumn):
    return {'type': 'IdentityCategoricalColumn', 'key': feature_column.key}
  elif isinstance(feature_column,
                  feature_column_v2.VocabularyFileCategoricalColumn):
    return {
        'type': 'VocabularyFileCategoricalColumn',
        'key': feature_column.key,
    }
  elif isinstance(feature_column,
                  feature_column_v2.VocabularyListCategoricalColumn):
    return {
        'type': 'VocabularyListCategoricalColumn',
        'key': feature_column.key,
    }
  elif isinstance(feature_column, feature_column_v2.WeightedCategoricalColumn):
    return {
        'type': 'WeightedCategoricalColumn',
        'key': feature_column.categorical_column.key,
        'weight_feature_key': feature_column.weight_feature_key,
    }
  elif isinstance(feature_column, feature_column_v2.EmbeddingColumn):
    return {
        'type':
            'EmbeddingColumn',
        'key':
            feature_column.name,
        'combiner':
            feature_column.combiner,
        'categorical_column':
            _serialize_feature_column(feature_column.categorical_column)
    }
  elif isinstance(feature_column, feature_column_v2.SharedEmbeddingColumn):
    return {
        'type':
            'SharedEmbeddingColumn',
        'key':
            feature_column.name,
        'combiner':
            feature_column.combiner,
        'categorical_column':
            _serialize_feature_column(feature_column.categorical_column)
    }
  elif isinstance(feature_column, feature_column_v2.IndicatorColumn):
    return {
        'type':
            'IndicatorColumn',
        'key':
            feature_column.name,
        'categorical_column':
            _serialize_feature_column(feature_column.categorical_column)
    }
  else:
    raise ValueError('unknown feature column, type %s, value %s' %
                     (type(feature_column), str(feature_column)))


def _serialize_feature_metadata_for_model(
    cols_to_output_tensors: Dict[feature_column_v2.FeatureColumn, tf.Tensor],
    features: Dict[Text, types.TensorType]) -> Dict[Text, Any]:
  """Serialize feature metadata for a single model into a dictionary."""
  feature_columns = []
  associated_tensors = []
  for k, v in cols_to_output_tensors.items():
    feature_columns.append(_serialize_feature_column(k))
    associated_tensors.append(
        tf.compat.v1.saved_model.utils.build_tensor_info(v))

  serialized_features_dict = {
      k: tf.compat.v1.saved_model.utils.build_tensor_info(v)
      for k, v in features.items()
  }

  return {
      'feature_columns': feature_columns,
      'associated_tensors': associated_tensors,
      'features': serialized_features_dict,
  }


def feature_metadata_path(export_path: bytes) -> bytes:
  """Returns the path to the feature metadata for the given export dir."""
  return os.path.join(export_path, b'assets.extra', b'feature_metadata')


def serialize_feature_metadata(output_dict: Dict[Text, Any]) -> bytes:
  """Returns a blob of serialized feature metadata.

  Args:
    output_dict: Output dictionary in the format returned by _observe_dnn_model.

  Returns:
    Blob of serialized feature metdata.
  """
  dump_dict = {}
  if 'dnn_model' in output_dict:
    dump_dict['dnn_model'] = _serialize_feature_metadata_for_model(
        output_dict['dnn_model_cols_to_output_tensors'],
        output_dict['dnn_model_features'])

  return pickle.dumps(dump_dict)


@tfma_util.kwargs_only
def export_eval_savedmodel_with_feature_metadata(
    estimator,
    export_dir_base: Text,
    eval_input_receiver_fn: Callable[[], tfma.export.EvalInputReceiverType],
    serving_input_receiver_fn: Optional[
        Callable[[], tf.estimator.export.ServingInputReceiver]] = None,
    assets_extra: Optional[Dict[Text, Text]] = None,
    checkpoint_path: Optional[Text] = None) -> bytes:
  """Like tfma.export.export_eval_savedmodel, with extra feature metadata."""
  output_dict = {}
  with _observe_dnn_model(output_dict):
    export_path = tfma.export.export_eval_savedmodel(
        estimator=estimator,
        export_dir_base=export_dir_base,
        eval_input_receiver_fn=eval_input_receiver_fn,
        serving_input_receiver_fn=serving_input_receiver_fn,
        assets_extra=assets_extra,
        checkpoint_path=checkpoint_path)

  # Write feature metadata as an asset to model after export has been done.
  # It would be cleaner to write this as an asset during export, but there
  # doesn't seem to be a nice way to do this given how we are monkey-patching
  # to observe/capture the information we need.
  if not export_path:
    raise ValueError('export appears to have failed. export_path was: %s' %
                     export_path)

  output_path = feature_metadata_path(export_path)
  assets_extra_path = os.path.dirname(output_path)
  if not os.path.isdir(assets_extra_path):
    os.mkdir(assets_extra_path)

  with open(output_path, 'wb+') as f:
    f.write(serialize_feature_metadata(output_dict))

  return export_path


def deserialize_feature_metadata(serialized_feature_metadata: bytes
                                ) -> Dict[Text, Any]:
  """Deserialize serialized feature metadata blob.

  Args:
    serialized_feature_metadata: Feature metadata serialized using
      serialize_feature_metadata.

  Returns:
    Dictionary containing three keys:
    - feature_columns: light-weight serialized representation of FeatureColumns
    - associated_tensors: TensorInfo for the Tensor associated with each
        feature_column, in the same order.
    - features: dictionary mapping raw feature key to the TensorInfo for the
        associated Tensor.
  """

  output_dict = pickle.loads(serialized_feature_metadata)

  result = {'feature_columns': [], 'associated_tensors': [], 'features': {}}

  def merge_feature_metadata(into: Dict[Text, Any],
                             merge_from: Dict[Text, Any]) -> None:
    """Merge the second feature metadata dictionary into the first."""
    into['feature_columns'].extend(merge_from['feature_columns'])
    into['associated_tensors'].extend(merge_from['associated_tensors'])
    common_keys = (
        set(into['features'].keys()) & set(merge_from['features'].keys()))
    for key in common_keys:
      if into['features'][key] != merge_from['features'][key]:
        raise ValueError(
            'key %s was common to both into and merge_from, but '
            'had distinct values: into value was %s, merge_from '
            'value was %s' %
            (key, into['features'][key], merge_from['features'][key]))
    into['features'].update(merge_from['features'])

  if 'linear_model' in output_dict:
    merge_feature_metadata(result, output_dict['linear_model'])
  if 'dnn_model' in output_dict:
    merge_feature_metadata(result, output_dict['dnn_model'])

  return result


def load_feature_metadata(eval_saved_model_path: bytes):
  """Get feature data (feature columns, feature) from EvalSavedModel metadata.

  Args:
    eval_saved_model_path: Path to EvalSavedModel, for the purposes of loading
      the feature_metadata file.

  Returns:
    Described in deserialize_feature_metadata.
  """

  with open(feature_metadata_path(eval_saved_model_path), 'rb') as f:
    return deserialize_feature_metadata(f.read())


def load_and_resolve_feature_metadata(eval_saved_model_path: bytes,
                                      graph: tf.Graph):
  """Get feature data (feature columns, feature) from EvalSavedModel metadata.

  Like load_feature_metadata, but additionally resolves the Tensors in the given
  graph.

  Args:
    eval_saved_model_path: Path to EvalSavedModel, for the purposes of loading
      the feature_metadata file.
    graph: tf.Graph to resolve the Tensors in.

  Returns:
    Same as load_feature_metadata, except associated_tensors and features
    contain the Tensors resolved in the graph instead of TensorInfos.
  """
  result = load_feature_metadata(eval_saved_model_path=eval_saved_model_path)

  # Resolve Tensors in graph
  result['associated_tensors'] = [
      tf.compat.v1.saved_model.get_tensor_from_tensor_info(tensor_info, graph)
      for tensor_info in result['associated_tensors']
  ]
  result['features'] = {
      k: tf.compat.v1.saved_model.get_tensor_from_tensor_info(v, graph)
      for k, v in result['features'].items()
  }

  return result
